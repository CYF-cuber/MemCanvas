#!/usr/bin/env python3
"""
Retrieval Improvement Experiment for Memory Canvas on ScienceQA.

Two experiments:
  1. Hybrid key embedding: combine image + text features for memory keys
  2. Alternative encoders: SigLIP2, MetaCLIP, etc.

Two-phase evaluation:
  Phase 1 (Fast, no VLM): Re-encode memories, compute retrieval metrics
  Phase 2 (Slow, VLM):    Evaluate accuracy for best configurations

Usage:
  # Phase 1 only (fast — retrieval metrics)
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/retrieval_improvement_eval.py --phase1-only

  # Full evaluation
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/retrieval_improvement_eval.py

  # Resume
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/retrieval_improvement_eval.py --resume /path/to/output_dir
"""

import argparse
import gc
import io
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (  # noqa: E402
    ExperimentConfig,
    MemoryEntry,  # needed for pickle
    MemoryIndex,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
)

# Import prompt builder from prompt_improvement_eval
sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import (  # noqa: E402
    build_prompt_v2,
    extract_answer_last,
    predict_with_variant,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]
INTERNVL_MODEL_PATH = "/home/cyf/InternVL3.5"

# Hybrid alpha values to test
ALPHA_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Encoder configurations
ENCODER_CONFIGS = {
    "clip-l14": {
        "model_id": "openai/clip-vit-large-patch14",
        "type": "clip",
        "embedding_dim": 768,
    },
    "siglip2-base-384": {
        "model_id": "google/siglip2-base-patch16-384",
        "type": "siglip2",
        "embedding_dim": 768,
    },
    "siglip2-large-384": {
        "model_id": "google/siglip2-large-patch16-384",
        "type": "siglip2",
        "embedding_dim": 1024,
    },
    "siglip2-so400m-384": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "type": "siglip2",
        "embedding_dim": 1152,
    },
    "metaclip-h14": {
        "model_id": "facebook/metaclip-h14-fullcc2.5b",
        "type": "clip",  # uses standard CLIPModel API
        "embedding_dim": 1024,
    },
}


# ---------------------------------------------------------------------------
# InternVL3.5 image preprocessing (from official README)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_internvl_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_internvl_image(pil_image, input_size=448, max_num=6):
    """Preprocess a PIL Image for InternVL3.5. Returns (pixel_values, num_patches)."""
    transform = _build_internvl_transform(input_size)
    images = _dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values, len(images)


# ---------------------------------------------------------------------------
# InternVL3.5 Evaluator
# ---------------------------------------------------------------------------
class InternVL35Evaluator:
    """InternVL3.5-8B evaluator with same interface as Qwen25VLEvaluator."""

    def __init__(self, model_path: str = INTERNVL_MODEL_PATH, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer
        try:
            import flash_attn  # noqa: F401
            _has_flash_attn = True
        except ImportError:
            _has_flash_attn = False
        print(f"  Loading InternVL3.5-8B from {model_path} (flash_attn={_has_flash_attn}) ...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=_has_flash_attn,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.device = device
        print(f"    InternVL3.5-8B loaded on {device}")

    def free(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
def load_memory_index(path: str) -> Tuple[List, np.ndarray, int]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    return memories, embeddings, embedding_dim


# ---------------------------------------------------------------------------
# Unified encoder interface
# ---------------------------------------------------------------------------
class UnifiedEncoder:
    """Unified interface for different vision-language encoders."""

    def __init__(self, encoder_name: str, device: str = "cuda"):
        self.encoder_name = encoder_name
        self.config = ENCODER_CONFIGS[encoder_name]
        self.device = device
        self.model = None
        self.processor = None
        self.embedding_dim = self.config["embedding_dim"]
        self._load_model()

    def _load_model(self):
        model_id = self.config["model_id"]
        enc_type = self.config["type"]
        print(f"  Loading encoder: {self.encoder_name} ({model_id})")

        if enc_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)

        elif enc_type == "siglip2":
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(model_id)

        else:
            raise ValueError(f"Unknown encoder type: {enc_type}")

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"    Loaded on {self.device}, dim={self.embedding_dim}")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single image to normalized embedding."""
        enc_type = self.config["type"]

        if enc_type == "clip":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()

        elif enc_type == "siglip2":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            # SigLIP2 returns pooled features directly
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text to normalized embedding."""
        enc_type = self.config["type"]

        if enc_type == "clip":
            inputs = self.processor(
                text=text, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()

        elif enc_type == "siglip2":
            inputs = self.processor(
                text=text, return_tensors="pt",
                padding="max_length", max_length=64, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()

    def encode_images_batch(
        self, images: List[Image.Image], batch_size: int = 16
    ) -> np.ndarray:
        """Encode multiple images in batches."""
        all_features = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            enc_type = self.config["type"]

            if enc_type == "clip":
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)

            elif enc_type == "siglip2":
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                if hasattr(features, 'pooler_output'):
                    features = features.pooler_output

            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    def free(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Text extraction from memory/problem
# ---------------------------------------------------------------------------
def get_memory_text(pid: str, problems: Dict) -> str:
    """Extract key text from a problem for text-based key embedding."""
    prob = problems.get(pid, {})
    parts = []
    subject = prob.get("subject", "")
    topic = prob.get("topic", "")
    if subject:
        parts.append(subject)
    if topic:
        parts.append(topic)
    question = prob.get("question", "")
    if question:
        parts.append(question)
    hint = prob.get("hint", "")
    if hint:
        parts.append(hint)
    choices = prob.get("choices", [])
    if choices:
        parts.append(" ".join(choices))
    return " ".join(parts)


def get_query_text(problem: Dict) -> str:
    """Extract query text for a test problem."""
    return (problem.get("question", "") + " " + problem.get("hint", "")).strip()


# ---------------------------------------------------------------------------
# Unified VLM prediction (supports Qwen2.5-VL and InternVL3.5)
# ---------------------------------------------------------------------------
def predict_internvl(
    evaluator: InternVL35Evaluator,
    problem: Dict,
    retrieved_memories: Optional[List[Tuple]],
) -> Tuple[str, str]:
    """
    Run InternVL3.5 prediction with v2-style (guided+CoT) prompt.
    Returns (extracted_answer, raw_response).
    """
    question = problem.get("question", "")
    choices = problem.get("choices", [])
    hint = problem.get("hint", "")
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < len(CHOICE_LABELS)
    )

    prompt_parts = []
    all_pixel_values = []
    num_patches_list = []
    num_images = 0

    if retrieved_memories:
        prompt_parts.append(
            "Below are memory canvas images from similar problems you solved before."
        )
        prompt_parts.append(
            "Each canvas contains: the original question, answer choices "
            "(correct answer marked with \u2713), background knowledge, and solution reasoning."
        )
        prompt_parts.append(
            "Study these canvases carefully and apply the relevant knowledge to the new question."
        )
        prompt_parts.append("")
        for i, (mem, sim) in enumerate(retrieved_memories):
            canvas_img = mem.get_canvas_image()
            if canvas_img:
                pv, n_patches = _load_internvl_image(canvas_img, max_num=6)
                all_pixel_values.append(pv)
                num_patches_list.append(n_patches)
                num_images += 1
                prompt_parts.append(
                    f"<image>\n[Canvas {i+1}] ({mem.subject} - {mem.topic})"
                )
        prompt_parts.append("")
        prompt_parts.append("---")

    prompt_parts.append("Answer the following question by selecting the correct option.")
    if hint:
        prompt_parts.append(f"Context: {hint}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(f"Options:\n{choices_text}")
    prompt_parts.append(
        "Think step by step, then give your final answer as a single letter "
        "(A, B, C, D, etc.) on the last line."
    )

    user_question = "\n".join(prompt_parts)

    if all_pixel_values:
        pixel_values = torch.cat(all_pixel_values, dim=0).to(
            torch.bfloat16
        ).to(evaluator.device)
    else:
        pixel_values = None

    generation_config = dict(max_new_tokens=512, do_sample=False)

    try:
        response = evaluator.model.chat(
            evaluator.tokenizer,
            pixel_values,
            user_question,
            generation_config,
            num_patches_list=num_patches_list if num_patches_list else None,
        )
    except Exception as e:
        response = f"ERROR: {e}"

    answer = extract_answer_last(response)
    return answer, response


# ---------------------------------------------------------------------------
# Phase 1: Encoding and retrieval evaluation
# ---------------------------------------------------------------------------
def encode_all_memories(
    encoder: UnifiedEncoder,
    memories: List,
    batch_size: int = 16,
    checkpoint_path: Optional[Path] = None,
) -> np.ndarray:
    """Encode all memory canvas images with a given encoder."""
    # Check checkpoint
    if checkpoint_path and checkpoint_path.exists():
        data = np.load(checkpoint_path)
        if data.shape[0] == len(memories):
            print(f"    Image embeddings loaded from cache ({data.shape})")
            return data

    print(f"    Encoding {len(memories)} canvas images...")
    all_embeddings = []
    batch_images = []

    for i, mem in enumerate(tqdm(memories, desc="Encode images")):
        img = mem.get_canvas_image()
        if img is None:
            # Create a blank image placeholder
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        batch_images.append(img)

        if len(batch_images) == batch_size or i == len(memories) - 1:
            batch_embs = encoder.encode_images_batch(batch_images, batch_size)
            all_embeddings.append(batch_embs)
            batch_images = []

    result = np.vstack(all_embeddings)

    if checkpoint_path:
        np.save(checkpoint_path, result)
        print(f"    Saved image embeddings to {checkpoint_path}")

    return result


def encode_all_texts(
    encoder: UnifiedEncoder,
    pids: List[str],
    problems: Dict,
    is_memory: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> np.ndarray:
    """Encode all text (memory or query) with a given encoder."""
    if checkpoint_path and checkpoint_path.exists():
        data = np.load(checkpoint_path)
        if data.shape[0] == len(pids):
            print(f"    Text embeddings loaded from cache ({data.shape})")
            return data

    desc = "Encode memory texts" if is_memory else "Encode query texts"
    print(f"    Encoding {len(pids)} texts...")
    all_embeddings = []

    for pid in tqdm(pids, desc=desc):
        if is_memory:
            text = get_memory_text(pid, problems)
        else:
            text = get_query_text(problems[pid])
        emb = encoder.encode_text(text)
        all_embeddings.append(emb)

    result = np.array(all_embeddings)

    if checkpoint_path:
        np.save(checkpoint_path, result)
        print(f"    Saved text embeddings to {checkpoint_path}")

    return result


def compute_hybrid_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute hybrid embeddings: alpha * image + (1 - alpha) * text.
    Both inputs must have same dimension. Result is L2-normalized.
    """
    hybrid = alpha * image_embeddings + (1.0 - alpha) * text_embeddings
    norms = np.linalg.norm(hybrid, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return hybrid / norms


def retrieve_top_k(
    query_embeddings: np.ndarray,
    key_embeddings: np.ndarray,
    test_pids: List[str],
    memory_pids: List[str],
    top_k: int = 2,
    threshold: float = 0.1,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Retrieve top-k memories for each test query.
    Returns mapping: test_pid -> list of (memory_index, similarity).
    """
    # Normalize
    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    query_norm = query_embeddings / q_norms

    k_norms = np.linalg.norm(key_embeddings, axis=1, keepdims=True)
    k_norms[k_norms == 0] = 1.0
    key_norm = key_embeddings / k_norms

    # Batch similarity computation
    # (num_test, dim) @ (dim, num_memories) -> (num_test, num_memories)
    sims = query_norm @ key_norm.T

    # Build memory pid set for self-filtering
    mem_pid_to_idx = {}
    for i, pid in enumerate(memory_pids):
        mem_pid_to_idx.setdefault(pid, []).append(i)

    retrieved_map = {}
    for i, test_pid in enumerate(test_pids):
        row = sims[i]
        # Get top-(k+5) to allow filtering
        top_indices = np.argsort(row)[::-1][: top_k + 5]

        results = []
        for idx in top_indices:
            if row[idx] < threshold:
                break
            # Self-filter
            if memory_pids[idx] == test_pid:
                continue
            results.append((int(idx), float(row[idx])))
            if len(results) >= top_k:
                break

        retrieved_map[test_pid] = results

    return retrieved_map


def compute_retrieval_metrics(
    retrieved_map: Dict[str, List[Tuple[int, float]]],
    test_pids: List[str],
    test_data: Dict,
    memories: List,
) -> Dict[str, float]:
    """Compute retrieval quality metrics."""
    subject_match = 0
    topic_match = 0
    has_retrieval = 0
    total = len(test_pids)

    avg_sim = []
    retrieval_counts = []

    for pid in test_pids:
        results = retrieved_map.get(pid, [])
        retrieval_counts.append(len(results))

        if not results:
            continue

        has_retrieval += 1
        test_subject = test_data[pid].get("subject", "")
        test_topic = test_data[pid].get("topic", "")

        for mem_idx, sim in results:
            avg_sim.append(sim)
            mem = memories[mem_idx]
            if mem.subject == test_subject:
                subject_match += 1
            if mem.topic == test_topic:
                topic_match += 1

    total_retrieved = sum(retrieval_counts)

    return {
        "has_retrieval_pct": has_retrieval / total * 100 if total > 0 else 0,
        "avg_similarity": float(np.mean(avg_sim)) if avg_sim else 0,
        "std_similarity": float(np.std(avg_sim)) if avg_sim else 0,
        "subject_match_pct": subject_match / total_retrieved * 100
        if total_retrieved > 0
        else 0,
        "topic_match_pct": topic_match / total_retrieved * 100
        if total_retrieved > 0
        else 0,
        "avg_retrieval_count": float(np.mean(retrieval_counts)),
    }


def compute_overlap(
    map_a: Dict[str, List[Tuple[int, float]]],
    map_b: Dict[str, List[Tuple[int, float]]],
    test_pids: List[str],
) -> float:
    """Compute fraction of test samples with identical top-k retrieval."""
    same = 0
    for pid in test_pids:
        a_indices = set(idx for idx, _ in map_a.get(pid, []))
        b_indices = set(idx for idx, _ in map_b.get(pid, []))
        if a_indices == b_indices:
            same += 1
    return same / len(test_pids) * 100 if test_pids else 0


def run_phase1(
    memories: List,
    existing_embeddings: np.ndarray,
    test_pids: List[str],
    test_data: Dict,
    problems: Dict,
    encoder_names: List[str],
    alpha_values: List[float],
    output_dir: Path,
    top_k: int = 2,
    threshold: float = 0.1,
) -> Dict:
    """
    Phase 1: Encode with different encoders and compute retrieval metrics.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Retrieval Quality Evaluation")
    print("=" * 60)

    memory_pids = [m.pid for m in memories]
    results = {}

    # Baseline: current CLIP-L/14 image-only retrieval
    print("\n--- Baseline: CLIP-L/14 image-only ---")
    baseline_encoder = UnifiedEncoder("clip-l14")

    # Encode test queries with CLIP-L/14
    query_cache = output_dir / "clip-l14_query_embeddings.npy"
    query_embeddings_clip = encode_all_texts(
        baseline_encoder, test_pids, test_data,
        is_memory=False, checkpoint_path=query_cache,
    )

    # Baseline retrieval (image-only keys, as currently used)
    baseline_retrieved = retrieve_top_k(
        query_embeddings_clip, existing_embeddings,
        test_pids, memory_pids, top_k, threshold,
    )
    baseline_metrics = compute_retrieval_metrics(
        baseline_retrieved, test_pids, test_data, memories
    )
    results["baseline_clip-l14_alpha1.0"] = {
        "encoder": "clip-l14",
        "alpha": 1.0,
        "metrics": baseline_metrics,
        "description": "Current: CLIP-L/14 image-only keys",
    }
    print(f"    Baseline metrics: {baseline_metrics}")

    # --- Experiment 1: Hybrid embeddings with CLIP-L/14 ---
    print("\n--- Experiment 1: Hybrid key embeddings (CLIP-L/14) ---")

    # Encode memory texts
    mem_text_cache = output_dir / "clip-l14_memory_text_embeddings.npy"
    mem_text_embeddings = encode_all_texts(
        baseline_encoder, memory_pids, problems,
        is_memory=True, checkpoint_path=mem_text_cache,
    )

    for alpha in alpha_values:
        label = f"clip-l14_alpha{alpha:.2f}"
        print(f"\n  Alpha={alpha:.2f}: ", end="")

        if alpha == 1.0:
            # Pure image — same as baseline
            results[label] = results["baseline_clip-l14_alpha1.0"]
            print("(same as baseline)")
            continue

        if alpha == 0.0:
            # Pure text
            hybrid_keys = mem_text_embeddings
        else:
            hybrid_keys = compute_hybrid_embeddings(
                existing_embeddings, mem_text_embeddings, alpha
            )

        retrieved = retrieve_top_k(
            query_embeddings_clip, hybrid_keys,
            test_pids, memory_pids, top_k, threshold,
        )
        metrics = compute_retrieval_metrics(
            retrieved, test_pids, test_data, memories
        )
        overlap = compute_overlap(baseline_retrieved, retrieved, test_pids)
        metrics["overlap_with_baseline"] = overlap

        results[label] = {
            "encoder": "clip-l14",
            "alpha": alpha,
            "metrics": metrics,
            "description": f"CLIP-L/14 hybrid α={alpha:.2f}",
        }
        print(f"metrics={metrics}")

    baseline_encoder.free()

    # --- Experiment 2: Alternative encoders ---
    print("\n--- Experiment 2: Alternative encoders ---")

    alt_encoders = [e for e in encoder_names if e != "clip-l14"]
    for enc_name in alt_encoders:
        print(f"\n  Encoder: {enc_name}")
        try:
            encoder = UnifiedEncoder(enc_name)
        except Exception as e:
            print(f"    Failed to load {enc_name}: {e}")
            results[f"{enc_name}_alpha1.0"] = {
                "encoder": enc_name,
                "alpha": 1.0,
                "metrics": {"error": str(e)},
                "description": f"{enc_name} (failed to load)",
            }
            continue

        # Encode images
        img_cache = output_dir / f"{enc_name}_image_embeddings.npy"
        img_embeddings = encode_all_memories(
            encoder, memories, batch_size=16, checkpoint_path=img_cache,
        )

        # Encode query texts
        query_cache_alt = output_dir / f"{enc_name}_query_embeddings.npy"
        query_embeddings_alt = encode_all_texts(
            encoder, test_pids, test_data,
            is_memory=False, checkpoint_path=query_cache_alt,
        )

        # Image-only retrieval
        retrieved = retrieve_top_k(
            query_embeddings_alt, img_embeddings,
            test_pids, memory_pids, top_k, threshold,
        )
        metrics = compute_retrieval_metrics(
            retrieved, test_pids, test_data, memories
        )
        overlap = compute_overlap(baseline_retrieved, retrieved, test_pids)
        metrics["overlap_with_baseline"] = overlap

        results[f"{enc_name}_alpha1.0"] = {
            "encoder": enc_name,
            "alpha": 1.0,
            "metrics": metrics,
            "description": f"{enc_name} image-only keys",
        }
        print(f"    Image-only metrics: {metrics}")

        # Also test hybrid for alternative encoders
        mem_text_cache_alt = output_dir / f"{enc_name}_memory_text_embeddings.npy"
        mem_text_embs_alt = encode_all_texts(
            encoder, memory_pids, problems,
            is_memory=True, checkpoint_path=mem_text_cache_alt,
        )

        for alpha in [0.0, 0.5]:
            label = f"{enc_name}_alpha{alpha:.2f}"
            if alpha == 0.0:
                hybrid_keys = mem_text_embs_alt
            else:
                hybrid_keys = compute_hybrid_embeddings(
                    img_embeddings, mem_text_embs_alt, alpha
                )

            retrieved_h = retrieve_top_k(
                query_embeddings_alt, hybrid_keys,
                test_pids, memory_pids, top_k, threshold,
            )
            metrics_h = compute_retrieval_metrics(
                retrieved_h, test_pids, test_data, memories
            )
            overlap_h = compute_overlap(baseline_retrieved, retrieved_h, test_pids)
            metrics_h["overlap_with_baseline"] = overlap_h

            results[label] = {
                "encoder": enc_name,
                "alpha": alpha,
                "metrics": metrics_h,
                "description": f"{enc_name} hybrid α={alpha:.2f}",
            }
            print(f"    α={alpha:.2f} metrics: {metrics_h}")

        encoder.free()

    # Save Phase 1 results
    phase1_path = output_dir / "phase1_results.json"
    with open(phase1_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Phase 1 results saved: {phase1_path}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: VLM accuracy evaluation
# ---------------------------------------------------------------------------
def run_phase2(
    memories: List,
    test_pids: List[str],
    test_data: Dict,
    problems: Dict,
    conditions: List[Tuple[str, float, int]],  # (encoder_name, alpha, top_k)
    vlm_types: List[str],  # e.g. ["qwen25vl", "internvl35"]
    output_dir: Path,
    threshold: float = 0.1,
    save_interval: int = 100,
) -> Dict:
    """
    Phase 2: VLM accuracy evaluation for selected configurations.
    Uses v2 (guided+CoT) prompt. Tests each condition with each VLM.
    """
    print("\n" + "=" * 60)
    print("Phase 2: VLM Accuracy Evaluation")
    print("=" * 60)

    memory_pids = [m.pid for m in memories]
    phase2_results = {}

    # Load checkpoint
    checkpoint_path = output_dir / "phase2_checkpoint.json"
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint loaded: {len(checkpoint)} conditions done")

    # Pre-compute retrieval for each unique (encoder, alpha, top_k) once
    retrieval_cache = {}
    for enc_name, alpha, top_k in conditions:
        cache_key = f"{enc_name}_alpha{alpha:.2f}_topk{top_k}"
        if cache_key in retrieval_cache:
            continue

        # Load precomputed embeddings
        img_cache = output_dir / f"{enc_name}_image_embeddings.npy"
        query_cache = output_dir / f"{enc_name}_query_embeddings.npy"
        mem_text_cache = output_dir / f"{enc_name}_memory_text_embeddings.npy"

        if enc_name == "clip-l14":
            _, existing_embs, _ = load_memory_index(DEFAULT_MEMORY_INDEX)
            img_embeddings = existing_embs
        elif img_cache.exists():
            img_embeddings = np.load(img_cache)
        else:
            print(f"    ERROR: Image embeddings not found for {enc_name}")
            continue

        if query_cache.exists():
            query_embeddings = np.load(query_cache)
        else:
            print(f"    ERROR: Query embeddings not found for {enc_name}")
            continue

        if alpha == 1.0:
            key_embeddings = img_embeddings
        elif alpha == 0.0:
            if mem_text_cache.exists():
                key_embeddings = np.load(mem_text_cache)
            else:
                print(f"    ERROR: Text embeddings not found for {enc_name}")
                continue
        else:
            if mem_text_cache.exists():
                mem_text_embs = np.load(mem_text_cache)
            else:
                print(f"    ERROR: Text embeddings not found for {enc_name}")
                continue
            key_embeddings = compute_hybrid_embeddings(
                img_embeddings, mem_text_embs, alpha
            )

        retrieved_map_indices = retrieve_top_k(
            query_embeddings, key_embeddings,
            test_pids, memory_pids, top_k, threshold,
        )
        retrieved_map = {}
        for pid, entries in retrieved_map_indices.items():
            retrieved_map[pid] = [(memories[idx], sim) for idx, sim in entries]

        retrieval_cache[cache_key] = retrieved_map
        print(f"  Retrieval cached: {cache_key}")

    # Evaluate each VLM
    for vlm_type in vlm_types:
        print(f"\n{'='*50}")
        print(f"  Loading VLM: {vlm_type}")
        print(f"{'='*50}")

        if vlm_type == "qwen25vl":
            exp_config = ExperimentConfig()
            vlm = Qwen25VLEvaluator(exp_config)
        elif vlm_type == "internvl35":
            vlm = InternVL35Evaluator()
        else:
            print(f"  Unknown VLM type: {vlm_type}, skipping")
            continue

        for enc_name, alpha, top_k in conditions:
            retrieval_key = f"{enc_name}_alpha{alpha:.2f}_topk{top_k}"
            label = f"{retrieval_key}_{vlm_type}"

            if label in checkpoint and checkpoint[label].get("complete", False):
                phase2_results[label] = checkpoint[label]
                print(f"\n  [{label}] Cached: {checkpoint[label]['accuracy']:.2f}%")
                continue

            if retrieval_key not in retrieval_cache:
                print(f"\n  [{label}] Skipped: no retrieval data")
                continue

            print(f"\n  --- Evaluating: {label} ---")
            retrieved_map = retrieval_cache[retrieval_key]

            partial = checkpoint.get(label, {})
            predictions = partial.get("predictions", [])
            start_idx = len(predictions)
            correct = partial.get("correct", 0)
            total = partial.get("total", 0)

            if start_idx > 0:
                print(f"    Resuming from {start_idx}/{len(test_pids)}")

            for idx in tqdm(
                range(start_idx, len(test_pids)),
                desc=label, initial=start_idx, total=len(test_pids),
            ):
                pid = test_pids[idx]
                problem = test_data[pid]
                answer_idx = problem.get("answer", 0)
                correct_answer = (
                    CHOICE_LABELS[answer_idx]
                    if answer_idx < len(CHOICE_LABELS)
                    else "A"
                )

                mems = retrieved_map.get(pid)
                if not mems:
                    mems = None

                try:
                    if vlm_type == "qwen25vl":
                        pred, raw_response = predict_with_variant(
                            vlm, problem, mems, "v2"
                        )
                    elif vlm_type == "internvl35":
                        pred, raw_response = predict_internvl(
                            vlm, problem, mems
                        )
                except Exception as e:
                    print(f"\n    Warning: {pid} failed: {e}")
                    pred = "A"
                    raw_response = f"ERROR: {e}"

                is_correct = pred == correct_answer
                correct += int(is_correct)
                total += 1

                predictions.append({
                    "pid": pid,
                    "predicted": pred,
                    "correct": correct_answer,
                    "is_correct": is_correct,
                })

                # Periodic save
                if (idx + 1) % save_interval == 0 or idx == len(test_pids) - 1:
                    acc = correct / total * 100 if total > 0 else 0
                    checkpoint[label] = {
                        "correct": correct,
                        "total": total,
                        "accuracy": acc,
                        "predictions": predictions,
                        "complete": False,
                    }
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            accuracy = correct / total * 100 if total > 0 else 0
            result = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "predictions": predictions,
                "complete": True,
            }
            checkpoint[label] = result
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            phase2_results[label] = result
            print(f"    [{label}] Accuracy: {accuracy:.2f}%")

        # Free VLM memory before loading the next one
        if vlm_type == "qwen25vl":
            del vlm
            torch.cuda.empty_cache()
            gc.collect()
        elif vlm_type == "internvl35":
            vlm.free()

    return phase2_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    phase1_results: Dict,
    phase2_results: Optional[Dict],
    output_dir: Path,
) -> Path:
    """Generate Chinese markdown report."""
    report_path = output_dir / "retrieval_report.md"
    lines = []

    lines.append("# textexperiment report")
    lines.append("")
    lines.append(f"*text: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. text")
    lines.append("")
    lines.append("textMemory Canvastextquality，text：")
    lines.append("1. **textKey Embedding**：text")
    lines.append("2. **text**：textSigLIP2、MetaCLIPtext")
    lines.append("")

    # Phase 1 results
    lines.append("## 2. Phase 1: textqualitytext")
    lines.append("")
    lines.append(
        "| text | text | Subjecttext | Topictext | textBaselinetext |"
    )
    lines.append(
        "|------|:---------:|:------------:|:-----------:|:-------------:|"
    )

    for label in sorted(phase1_results.keys()):
        entry = phase1_results[label]
        m = entry.get("metrics", {})
        if "error" in m:
            lines.append(f"| {entry['description']} | ERROR | — | — | — |")
            continue
        avg_sim = m.get("avg_similarity", 0)
        subj = m.get("subject_match_pct", 0)
        topic = m.get("topic_match_pct", 0)
        overlap = m.get("overlap_with_baseline", 100)
        lines.append(
            f"| {entry['description']} | {avg_sim:.4f} | {subj:.1f}% "
            f"| {topic:.1f}% | {overlap:.1f}% |"
        )
    lines.append("")

    # Phase 2 results
    if phase2_results:
        lines.append("## 3. Phase 2: VLMaccuracy")
        lines.append("")
        lines.append("textv2 (guided+CoT)text，Oracletext。")
        lines.append("")

        # Reference results
        lines.append("### textbaseline (Qwen2.5-VL-7B)")
        lines.append("")
        lines.append("| text | accuracy |")
        lines.append("|------|:------:|")
        lines.append("| v2 baseline (no memory) | 81.70% |")
        lines.append("| v0 Oracle (textCLIP-L/14, top-2) | 80.71% |")
        lines.append("| v2 Oracle (textCLIP-L/14, top-2) | 82.32% |")
        lines.append("")

        # Group by VLM type, then by (retrieval_config, top_k)
        from collections import defaultdict
        vlm_groups = defaultdict(lambda: defaultdict(dict))
        # label format: enc_alphaX.XX_topkN_vlmtype
        for label in sorted(phase2_results.keys()):
            entry = phase2_results[label]
            # Parse: find last _ segment as vlm_type
            # Label: clip-l14_alpha0.50_topk2_qwen25vl
            for vt in ["qwen25vl", "internvl35"]:
                if label.endswith(f"_{vt}"):
                    vlm_name = vt
                    rest = label[: -(len(vt) + 1)]  # remove _vlmtype
                    break
            else:
                vlm_name = "unknown"
                rest = label

            # Split rest into config_key and topk
            parts = rest.rsplit("_topk", 1)
            config_key = parts[0]
            tk = int(parts[1]) if len(parts) > 1 else 2
            vlm_groups[vlm_name][config_key][tk] = entry

        # For each VLM, show top-k ablation table
        vlm_display = {"qwen25vl": "Qwen2.5-VL-7B", "internvl35": "InternVL3.5-8B"}
        for vlm_name in sorted(vlm_groups.keys()):
            display_name = vlm_display.get(vlm_name, vlm_name)
            lines.append(f"### {display_name}: text (top-k = 1, 2, 3)")
            lines.append("")

            config_groups = vlm_groups[vlm_name]
            all_ks = sorted(
                set(k for group in config_groups.values() for k in group.keys())
            )
            if not all_ks:
                continue

            header = "| text | " + " | ".join(f"top-{k}" for k in all_ks) + " |"
            sep = "|------|" + "|".join(":------:" for _ in all_ks) + "|"
            lines.append(header)
            lines.append(sep)

            for config_key in sorted(config_groups.keys()):
                group = config_groups[config_key]
                cells = []
                for k in all_ks:
                    if k in group:
                        acc = group[k].get("accuracy", 0)
                        diff = acc - 81.70
                        cells.append(f"{acc:.2f}% ({diff:+.2f}%)")
                    else:
                        cells.append("—")
                lines.append(f"| {config_key} | " + " | ".join(cells) + " |")
            lines.append("")

        # VLM comparison table (best config per VLM)
        if len(vlm_groups) > 1:
            lines.append("### VLMtext (text)")
            lines.append("")
            lines.append("| VLM | text | accuracy |")
            lines.append("|-----|---------|:------:|")

            for vlm_name in sorted(vlm_groups.keys()):
                display_name = vlm_display.get(vlm_name, vlm_name)
                best_acc = 0
                best_label = ""
                for config_key, group in vlm_groups[vlm_name].items():
                    for tk, entry in group.items():
                        acc = entry.get("accuracy", 0)
                        if acc > best_acc:
                            best_acc = acc
                            best_label = f"{config_key}_topk{tk}"
                lines.append(
                    f"| {display_name} | {best_label} | {best_acc:.2f}% |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*text: NVIDIA RTX A6000, Qwen2.5-VL-7B-Instruct / InternVL3.5-8B, "
        "ScienceQAtext(4,241text)*"
    )

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval Improvement Experiment for ScienceQA"
    )
    parser.add_argument(
        "--memory-index", default=DEFAULT_MEMORY_INDEX,
        help="Path to memory index pickle",
    )
    parser.add_argument(
        "--output-root", default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument(
        "--encoders", nargs="+",
        default=["clip-l14", "siglip2-base-384", "siglip2-so400m-384"],
        help="Encoders to test",
    )
    parser.add_argument(
        "--phase2-conditions", nargs="+", default=[],
        help='Conditions for Phase 2, e.g. "clip-l14:0.5:2 siglip2-so400m-384:1.0:3"'
             ' (format: encoder:alpha:top_k)',
    )
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--vlm", nargs="+",
        default=["qwen25vl", "internvl35"],
        choices=["qwen25vl", "internvl35"],
        help="VLMs to evaluate in Phase 2",
    )

    args = parser.parse_args()

    # Output directory
    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists()
        print(f"Resuming from: {output_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"retrieval_eval_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load memory index
    print(f"\nLoading memory index: {args.memory_index}")
    memories, embeddings, embedding_dim = load_memory_index(args.memory_index)
    print(f"  Loaded {len(memories)} memories, dim={embedding_dim}")

    # Load data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())

    if args.max_test_samples and args.max_test_samples > 0:
        test_pids = test_pids[: args.max_test_samples]
    print(f"  Test samples: {len(test_pids)}")

    # Load all problems (needed for text extraction)
    all_problems = loader.problems

    # Phase 1
    phase1_path = output_dir / "phase1_results.json"
    if phase1_path.exists() and args.resume:
        with open(phase1_path) as f:
            phase1_results = json.load(f)
        print(f"\n  Phase 1 results loaded from cache")
    else:
        phase1_results = run_phase1(
            memories, embeddings,
            test_pids, test_data, all_problems,
            args.encoders, ALPHA_VALUES, output_dir,
            top_k=args.top_k, threshold=args.threshold,
        )

    # Phase 2
    phase2_results = None
    if not args.phase1_only:
        # Parse phase2 conditions
        if args.phase2_conditions:
            conditions = []
            for c in args.phase2_conditions:
                parts = c.split(":")
                enc = parts[0]
                alpha = float(parts[1]) if len(parts) > 1 else 1.0
                top_k = int(parts[2]) if len(parts) > 2 else 2
                conditions.append((enc, alpha, top_k))
        else:
            # Auto-select: pick top-3 configs from Phase 1 by subject_match_pct
            # then cross with top_k=[1,2,3] for ablation
            ranked = sorted(
                phase1_results.items(),
                key=lambda x: x[1].get("metrics", {}).get("subject_match_pct", 0),
                reverse=True,
            )
            best_configs = []
            seen = set()
            for label, entry in ranked:
                key = (entry["encoder"], entry["alpha"])
                if key not in seen and len(best_configs) < 3:
                    best_configs.append(key)
                    seen.add(key)

            # Cross best configs with top_k ablation [1, 2, 3]
            conditions = []
            for enc, alpha in best_configs:
                for k in [1, 2, 3]:
                    conditions.append((enc, alpha, k))

        if conditions:
            print(f"\n  Phase 2 conditions: {conditions}")
            print(f"  VLMs: {args.vlm}")
            phase2_results = run_phase2(
                memories, test_pids, test_data, all_problems,
                conditions, args.vlm, output_dir,
                threshold=args.threshold,
                save_interval=args.save_interval,
            )

    # Generate report
    generate_report(phase1_results, phase2_results, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("Phase 1 Summary")
    print(f"{'='*60}")
    print(f"{'Config':<40} {'AvgSim':>8} {'Subj%':>8} {'Topic%':>8}")
    print("-" * 68)
    for label in sorted(phase1_results.keys()):
        m = phase1_results[label].get("metrics", {})
        if "error" in m:
            continue
        desc = phase1_results[label]["description"][:38]
        print(
            f"{desc:<40} {m.get('avg_similarity', 0):>8.4f} "
            f"{m.get('subject_match_pct', 0):>7.1f}% "
            f"{m.get('topic_match_pct', 0):>7.1f}%"
        )

    if phase2_results:
        print(f"\n{'='*60}")
        print("Phase 2 Summary (VLM Accuracy)")
        print(f"{'='*60}")
        for label in sorted(phase2_results.keys()):
            acc = phase2_results[label].get("accuracy", 0)
            print(f"  {label}: {acc:.2f}%")

    print(f"\nResults directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
