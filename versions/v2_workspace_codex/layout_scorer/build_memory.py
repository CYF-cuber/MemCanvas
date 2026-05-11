#!/usr/bin/env python3
"""
Phase 3a: Build memory library using SmartCanvas + LayoutScorer.

Replaces DynamicCanvas with SmartCanvasLayout + trained CLIP+MLP Scorer.
For each training sample:
    1. Create content blocks
    2. Generate 6 candidate layouts
    3. Score with LayoutScorer → pick best
    4. Render best layout
    5. Extract CLIP key embedding
    6. Store as MemoryEntry (same format as existing pipeline)

Outputs:
- memory_index_smart.pkl (compatible with existing eval pipeline)

Usage:
    python -m layout_scorer.build_memory [--max_samples None] [--device cuda]
"""

import io
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
sys.path.insert(0, "/home/cyf/memory")

from smart_canvas_layout import (
    ContentBlock,
    Layout,
    BlockType,
    layout_single_column,
    layout_two_column,
    layout_side_by_side,
    render_layout,
    measure_text,
    measure_image,
    measure_table,
)
from layout_scorer.model import LayoutScorer

# Data paths
SCIENCEQA_DIR = "/home/cyf/memory/ScienceQA"
PROBLEMS_PATH = f"{SCIENCEQA_DIR}/data/scienceqa/problems.json"
CAPTIONS_PATH = f"{SCIENCEQA_DIR}/data/captions.json"
IMAGES_DIR = f"{SCIENCEQA_DIR}/data/scienceqa/images"

SCORER_PATH = "/home/cyf/codex/layout_scorer_output/layout_scorer.pt"
OUTPUT_DIR = "/home/cyf/memory/experiments/scienceqa_smart_scorer"


# Reuse MemoryEntry format from existing pipeline
@dataclass
class MemoryEntry:
    pid: str
    subject: str = ""
    topic: str = ""
    has_image: bool = False
    num_choices: int = 0
    correct_choice_idx: int = 0
    key_embedding: Optional[np.ndarray] = None
    canvas_image_bytes: Optional[bytes] = None

    def get_canvas_image(self) -> Optional[Image.Image]:
        if self.canvas_image_bytes is None:
            return None
        return Image.open(io.BytesIO(self.canvas_image_bytes))


class MemoryIndex:
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.memories: List[MemoryEntry] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(self, memory: MemoryEntry):
        self.memories.append(memory)

    def build_index(self):
        if not self.memories:
            return
        embeddings = []
        for m in self.memories:
            if m.key_embedding is not None:
                embeddings.append(m.key_embedding)
            else:
                embeddings.append(np.zeros(self.embedding_dim))
        self.embeddings = np.array(embeddings)
        print(f"Index built: {len(self.memories)} memories, dim={self.embeddings.shape}")

    def save(self, path: str):
        data = {
            'memories': self.memories,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved: {path} ({len(self.memories)} memories)")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.memories = data['memories']
        self.embeddings = data['embeddings']
        self.embedding_dim = data.get('embedding_dim', 768)
        print(f"Index loaded: {len(self.memories)} memories")

    def search(self, query_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.1):
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append((self.memories[idx], float(sim)))
        return results


def problem_to_blocks(problem: Dict, img: Optional[Image.Image], caption: str = "") -> List[ContentBlock]:
    """Convert a ScienceQA problem into content blocks (enhanced with caption)."""
    blocks = []

    # Header
    subj = problem.get("subject", "")
    topic = problem.get("topic", "")
    pid = problem.get("pid", "")
    if subj or topic:
        blocks.append(measure_text(f"Q#{pid}: {subj} - {topic}", font_size=14))

    # Context/hint
    hint = problem.get("hint", "")
    if hint:
        blocks.append(measure_text(f"Context: {hint}", font_size=16))

    # Image
    if img is not None:
        blocks.append(measure_image(img, max_dim=500))

    # Caption
    if caption:
        blocks.append(measure_text(f"[Image] {caption}", font_size=16))

    # Question + choices
    question = problem.get("question", "")
    choices = problem.get("choices", [])
    answer_idx = problem.get("answer", -1)

    q_text = f"Question: {question}\n"
    for i, c in enumerate(choices):
        letter = chr(65 + i)
        marker = "✓" if i == answer_idx else "○"
        q_text += f"  {marker} {letter}. {c}\n"
    blocks.append(measure_text(q_text.strip(), font_size=18))

    # Background knowledge
    lecture = problem.get("lecture", "")
    if lecture:
        if "|" in lecture and lecture.count("|") >= 4:
            table_lines = [l.strip() for l in lecture.split("\n") if "|" in l]
            table_data = []
            for line in table_lines:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells and not all(set(c) <= {'-', ' '} for c in cells):
                    table_data.append(cells)
            if table_data:
                non_table = "\n".join(l for l in lecture.split("\n") if "|" not in l).strip()
                if non_table:
                    blocks.append(measure_text(f"Background: {non_table}", font_size=15))
                blocks.append(measure_table(table_data, font_size=15))
            else:
                blocks.append(measure_text(f"Background: {lecture}", font_size=15))
        else:
            blocks.append(measure_text(f"Background: {lecture}", font_size=15))

    # Solution
    solution = problem.get("solution", "")
    if solution:
        blocks.append(measure_text(f"Solution: {solution}", font_size=15))

    return blocks


def generate_all_candidates(blocks: List[ContentBlock]) -> List[Layout]:
    """Generate 6 candidate layouts."""
    total_area = sum(b.area for b in blocks) * 1.3
    target_side = int(math.sqrt(total_area))
    target_side = max(target_side, 350)
    target_side = min(target_side, 1200)

    candidates = [
        layout_single_column(blocks, target_side),
        layout_two_column(blocks, target_side),
        layout_side_by_side(blocks, target_side),
        layout_single_column(blocks, int(target_side * 1.3)),
        layout_two_column(blocks, int(target_side * 1.2)),
        layout_side_by_side(blocks, int(target_side * 1.2)),
    ]
    return candidates


class SmartCanvasMemoryBuilder:
    """Build memories using SmartCanvas + LayoutScorer."""

    def __init__(self, scorer_path: str, device: str = "cuda"):
        self.device = device
        self._load_scorer(scorer_path)
        self._load_clip()

    def _load_scorer(self, scorer_path: str):
        print(f"Loading LayoutScorer from {scorer_path}")
        ckpt = torch.load(scorer_path, map_location=self.device, weights_only=False)
        self.scorer = LayoutScorer(clip_dim=768, heuristic_dim=15).to(self.device)
        self.scorer.load_state_dict(ckpt["model_state_dict"])
        self.scorer.eval()
        print(f"  Scorer loaded (val pairwise={ckpt['val_metrics']['pairwise_acc']:.3f})")

    def _load_clip(self):
        from transformers import CLIPModel, CLIPProcessor
        import torch.nn.functional as F

        model_name = "openai/clip-vit-large-patch14"
        print(f"Loading CLIP model: {model_name}")
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_model.eval()
        print("  CLIP loaded")

    def extract_clip_features(self, image: Image.Image) -> torch.Tensor:
        """Extract CLIP visual features [768]."""
        import torch.nn.functional as F
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return F.normalize(features, dim=-1).squeeze(0)  # [768]

    def extract_key_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP key embedding for retrieval [768] numpy."""
        feat = self.extract_clip_features(image)
        return feat.cpu().numpy()

    def encode_text_query(self, text: str) -> np.ndarray:
        """Encode text query with CLIP [768] numpy."""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    def compute_heuristic_features(self, layout: Layout) -> List[float]:
        """Compute 15-dim handcrafted features from a Layout."""
        # Geometric [8]
        geo = [
            layout.squareness,
            layout.utilization,
            layout.width / 1000.0,
            layout.height / 1000.0,
            layout.aspect_ratio,
            (layout.width // 28) * (layout.height // 28) / 2000.0,
            len(layout.placements) / 10.0,
            1.0 if 400 <= layout.width <= 900 and 400 <= layout.height <= 900 else 0.0,
        ]
        # Typographic [7]
        text_blocks = [p for p in layout.placements if p.block.type == BlockType.TEXT]
        image_blocks = [p for p in layout.placements if p.block.type == BlockType.IMAGE]
        table_blocks = [p for p in layout.placements if p.block.type == BlockType.TABLE]

        min_text_w = min((p.width for p in text_blocks), default=0)
        total_text_area = sum(p.width * p.height for p in text_blocks)
        total_image_area = sum(p.width * p.height for p in image_blocks)

        typo = [
            min_text_w / 600.0,
            1.0 if min_text_w >= 250 else min_text_w / 250.0,
            total_text_area / max(layout.total_area, 1),
            total_image_area / max(layout.total_area, 1),
            1.0 if image_blocks else 0.0,
            1.0 if table_blocks else 0.0,
            len(text_blocks) / 6.0,
        ]
        return geo + typo

    def select_best_layout(self, candidates: List[Layout], rendered_images: List[Image.Image]) -> Tuple[int, np.ndarray]:
        """Score all candidates with LayoutScorer, return (best_index, best_key_embedding)."""
        import torch.nn.functional as F

        # Extract CLIP features for all candidates (batch)
        clip_inputs = self.clip_processor(images=rendered_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_feats = self.clip_model.get_image_features(**clip_inputs)
        clip_feats = F.normalize(clip_feats, dim=-1)  # [6, 768]

        # Compute heuristic features
        heuristic_feats = torch.tensor(
            [self.compute_heuristic_features(c) for c in candidates],
            dtype=torch.float32,
        ).to(self.device)  # [6, 15]

        # Score
        with torch.no_grad():
            scores = self.scorer(clip_feats, heuristic_feats)  # [6]

        best_idx = scores.argmax().item()
        # Reuse CLIP feature as key embedding (avoids redundant forward pass)
        best_key_embedding = clip_feats[best_idx].cpu().numpy()
        return best_idx, best_key_embedding

    def build_memory(self, pid: str, problem: Dict, caption: str,
                     image_path: Optional[str]) -> MemoryEntry:
        """Build a single memory entry using SmartCanvas + Scorer."""
        # Load image
        img = None
        if image_path:
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                pass

        # Create content blocks
        problem_with_pid = dict(problem, pid=pid)
        blocks = problem_to_blocks(problem_with_pid, img, caption)

        # Generate 6 candidate layouts
        candidates = generate_all_candidates(blocks)

        # Render all candidates
        rendered = [render_layout(c) for c in candidates]

        # Score and select best (reuses CLIP features for key embedding)
        best_idx, key_embedding = self.select_best_layout(candidates, rendered)
        best_image = rendered[best_idx]

        # Compress canvas to bytes
        buffer = io.BytesIO()
        best_image.save(buffer, format='PNG', optimize=True)
        canvas_bytes = buffer.getvalue()

        return MemoryEntry(
            pid=pid,
            subject=problem.get('subject', ''),
            topic=problem.get('topic', ''),
            has_image=img is not None,
            num_choices=len(problem.get('choices', [])),
            correct_choice_idx=problem.get('answer', 0),
            key_embedding=key_embedding,
            canvas_image_bytes=canvas_bytes,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3a: Build Memory with SmartCanvas + LayoutScorer")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading ScienceQA data...")
    with open(PROBLEMS_PATH) as f:
        problems = json.load(f)
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f).get('captions', {})

    train_pids = [pid for pid, p in problems.items() if p.get('split') == 'train']
    if args.max_samples:
        train_pids = train_pids[:args.max_samples]
    print(f"Training samples: {len(train_pids)}")

    # Check for existing checkpoint
    index_path = f"{OUTPUT_DIR}/memory_index_smart.pkl"
    if os.path.exists(index_path):
        print(f"Memory index already exists: {index_path}")
        print("Delete it to rebuild, or proceed to evaluation.")
        return

    # Initialize builder
    builder = SmartCanvasMemoryBuilder(SCORER_PATH, device=args.device)
    memory_index = MemoryIndex()

    # Build memories
    layout_stats = defaultdict(int)
    t0 = time.time()

    for idx in tqdm(range(len(train_pids)), desc="Building memories"):
        pid = train_pids[idx]
        problem = problems[pid]
        caption = captions.get(pid, '')
        image_path = None
        if problem.get('image'):
            split = problem.get('split', 'train')
            img_path = f"{IMAGES_DIR}/{split}/{pid}/image.png"
            if os.path.exists(img_path):
                image_path = img_path

        try:
            memory = builder.build_memory(pid, problem, caption, image_path)
            memory_index.add(memory)
        except Exception as e:
            print(f"\n[WARN] pid={pid}: {e}")
            continue

        # Checkpoint
        if (idx + 1) % args.checkpoint_interval == 0:
            ckpt_path = f"{OUTPUT_DIR}/memory_index_checkpoint_{idx+1}.pkl"
            memory_index.build_index()
            memory_index.save(ckpt_path)
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{idx+1}/{len(train_pids)}] Rate: {rate:.1f}/s, "
                  f"ETA: {(len(train_pids) - idx - 1) / rate / 60:.0f}min")

    # Build and save final index
    memory_index.build_index()
    memory_index.save(index_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Built {len(memory_index.memories)} memories in {elapsed/60:.1f}min")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
