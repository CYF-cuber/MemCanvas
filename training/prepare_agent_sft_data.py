#!/usr/bin/env python3
"""
Prepare SFT training data for the Canvas Manager Agent.

Constructs two datasets:
1. Write SFT: Oracle-labeled (canvas_state, knowledge) → write_action
2. Read SFT:  Oracle-labeled (question, candidates) → retrieval_action

Oracle rules use embedding similarity + canvas capacity constraints
to generate ground-truth labels. See agent_memcanvas_design.md §5.2.

Usage:
  conda activate base
  python -u prepare_agent_sft_data.py --max-train 2000 --max-test 1000
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
sys.path.insert(0, "/home/cyf/codex")

from canvas_agent import (
    CanvasMeta,
    WriteAction,
    WRITE_SYSTEM_PROMPT,
    READ_SYSTEM_PROMPT,
    build_write_prompt,
    build_read_prompt,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCIQA_DATA_DIR = Path("/home/cyf/memory/ScienceQA/data/scienceqa")
OUTPUT_DIR = Path("/home/cyf/codex/agent_sft_data")
CACHE_DIR = Path("/home/cyf/codex/agent_experiment_output")

# ---------------------------------------------------------------------------
# Data loading (reuse cached data from agent_experiment)
# ---------------------------------------------------------------------------
def load_scienceqa():
    """Load cached ScienceQA data."""
    cache_file = CACHE_DIR / "sciqa_cached.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["test"]

    # Fallback: load from source
    from agent_experiment import load_scienceqa as _load
    return _load()


def sample_to_knowledge_text(sample: Dict) -> str:
    """Convert a training sample to knowledge text for the agent."""
    parts = []
    parts.append(f"Subject: {sample['subject']} / {sample['topic']}")
    if sample["hint"]:
        parts.append(f"Context: {sample['hint'][:200]}")
    parts.append(f"Q: {sample['question']}")
    correct_idx = sample["answer"]
    choices = sample["choices"]
    correct_answer = choices[correct_idx] if correct_idx < len(choices) else ""
    parts.append(f"A: {correct_answer}")
    if sample["lecture"]:
        parts.append(f"Knowledge: {sample['lecture'][:200]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simulated canvas library (lightweight, no actual rendering)
# ---------------------------------------------------------------------------
@dataclass
class SimCanvas:
    """Lightweight simulated canvas for oracle labeling (no image rendering)."""
    canvas_id: int
    topic: str
    items: List[str] = field(default_factory=list)
    num_patches: int = 1
    max_patches: int = 4
    access_count: int = 0

    @property
    def is_full(self) -> bool:
        return self.num_patches >= self.max_patches

    @property
    def num_items(self) -> int:
        return len(self.items)

    def to_meta(self) -> CanvasMeta:
        return CanvasMeta(
            canvas_id=self.canvas_id,
            topic=self.topic,
            num_items=self.num_items,
            num_patches=self.num_patches,
            max_patches=self.max_patches,
            access_count=self.access_count,
            text_summary=" | ".join(self.items[-3:])[:200],
        )

    def add_item(self, text: str):
        self.items.append(text)
        # Estimate patches: ~2 items per patch for typical ScienceQA content
        self.num_patches = min(1 + len(self.items) // 2, self.max_patches)


class SimLibrary:
    """Simulated canvas library for oracle data construction."""

    def __init__(self):
        self.canvases: Dict[int, SimCanvas] = {}
        self._next_id: int = 0

    def create(self, topic: str, knowledge: str) -> int:
        cid = self._next_id
        self._next_id += 1
        canvas = SimCanvas(canvas_id=cid, topic=topic)
        canvas.add_item(knowledge)
        self.canvases[cid] = canvas
        return cid

    def append(self, canvas_id: int, knowledge: str) -> int:
        if canvas_id not in self.canvases:
            return self.create("General", knowledge)
        canvas = self.canvases[canvas_id]
        if canvas.is_full:
            return self.create(canvas.topic + " (cont.)", knowledge)
        canvas.add_item(knowledge)
        return canvas_id

    def get_metas(self) -> List[CanvasMeta]:
        return [c.to_meta() for c in self.canvases.values()]

    def get_topics(self) -> List[str]:
        return [c.topic for c in self.canvases.values()]

    def get_items(self, canvas_id: int) -> List[str]:
        if canvas_id in self.canvases:
            return self.canvases[canvas_id].items
        return []

    def reset(self):
        self.canvases.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Oracle Write Labeling
# ---------------------------------------------------------------------------
def extract_topic(knowledge: str) -> str:
    """Extract a short topic from knowledge text."""
    # Use Subject / Topic line if present
    for line in knowledge.split("\n"):
        if line.startswith("Subject:"):
            topic = line.replace("Subject:", "").strip()
            # Clean up "subject / topic" format
            parts = topic.split("/")
            if len(parts) >= 2:
                return parts[1].strip()[:50]
            return topic[:50]
    # Fallback: first 50 chars
    return knowledge[:50].split("\n")[0]


_STRUCTURAL_WORDS = frozenset({
    "subject:", "context:", "q:", "a:", "knowledge:", "the", "a", "an", "is",
    "are", "was", "were", "of", "in", "to", "and", "or", "for", "on", "at",
    "by", "it", "this", "that", "with", "from", "as", "be", "has", "have",
    "had", "do", "does", "did", "not", "but", "what", "which", "how", "when",
    "where", "who", "why", "can", "will", "would", "if", "/", "-", "o", "✓",
    "background:", "solution:", "context", "subject", "knowledge",
})


def is_duplicate(knowledge: str, items: List[str]) -> bool:
    """Check if knowledge is a true duplicate (same question + answer) of existing items.

    ScienceQA has many template questions (e.g., "Which country is highlighted?"
    appears 49x with different answers), so we must check both question AND answer
    to detect true duplicates.
    """
    def extract_qa(text: str) -> str:
        q, a = "", ""
        for line in text.split("\n"):
            if line.startswith("Q:"):
                q = line[2:].strip().lower()
            elif line.startswith("A:"):
                a = line[2:].strip().lower()
        return q + " ||| " + a

    new_qa = extract_qa(knowledge)
    if len(new_qa) < 10:
        return False

    for item in items:
        old_qa = extract_qa(item)
        if old_qa and new_qa == old_qa:
            return True
    return False


def oracle_write_decision(
    library: SimLibrary,
    new_knowledge: str,
    knowledge_emb: np.ndarray,
    topic_embeddings: Dict[int, np.ndarray],
    sim_threshold: float = 0.6,
) -> Dict:
    """
    Oracle write decision based on embedding similarity + canvas capacity.

    Returns dict with: action, canvas_id (for APPEND), topic (for CREATE), reason.
    """
    if not library.canvases:
        topic = extract_topic(new_knowledge)
        return {"action": "CREATE", "topic": topic, "reason": "first canvas"}

    # Compute similarities between new knowledge and all canvas topics
    canvas_ids = list(topic_embeddings.keys())
    if not canvas_ids:
        topic = extract_topic(new_knowledge)
        return {"action": "CREATE", "topic": topic, "reason": "no embeddings"}

    emb_matrix = np.array([topic_embeddings[cid] for cid in canvas_ids])
    similarities = emb_matrix @ knowledge_emb
    best_local_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_local_idx])
    best_cid = canvas_ids[best_local_idx]
    best_canvas = library.canvases[best_cid]

    # Rule 1: High similarity + not full → APPEND
    if best_sim > sim_threshold and not best_canvas.is_full:
        # But check for true duplicate first
        if is_duplicate(new_knowledge, best_canvas.items):
            return {
                "action": "NOOP",
                "reason": "duplicate of existing knowledge in canvas #{} '{}'".format(
                    best_cid, best_canvas.topic),
            }
        return {
            "action": "APPEND",
            "canvas_id": best_cid,
            "reason": f"topic match ({best_sim:.2f}) with canvas #{best_cid} '{best_canvas.topic}', canvas has space ({best_canvas.num_patches}/{best_canvas.max_patches})",
        }

    # Rule 2: High similarity + full → CREATE sub-topic
    if best_sim > sim_threshold and best_canvas.is_full:
        if is_duplicate(new_knowledge, best_canvas.items):
            return {
                "action": "NOOP",
                "reason": "duplicate of existing knowledge in canvas #{} '{}'".format(
                    best_cid, best_canvas.topic),
            }
        topic = best_canvas.topic + " (ext)"
        return {
            "action": "CREATE",
            "topic": topic,
            "reason": f"topic match ({best_sim:.2f}) but canvas #{best_cid} is full ({best_canvas.num_patches}/{best_canvas.max_patches})",
        }

    # Rule 3: Low similarity → CREATE new topic
    topic = extract_topic(new_knowledge)
    return {
        "action": "CREATE",
        "topic": topic,
        "reason": f"new topic, best match was canvas #{best_cid} '{best_canvas.topic}' with sim={best_sim:.2f}",
    }


# ---------------------------------------------------------------------------
# Oracle Read Labeling
# ---------------------------------------------------------------------------
def answer_info_in_canvas(
    answer: str, question: str, items: List[str]
) -> bool:
    """Check if canvas items contain information relevant to the answer."""
    answer_lower = answer.lower().strip()
    question_words = set(question.lower().split())

    for item in items:
        item_lower = item.lower()
        # Check 1: Answer text appears in canvas
        if answer_lower in item_lower and len(answer_lower) > 2:
            return True
        # Check 2: Significant keyword overlap between question and item
        item_words = set(item_lower.split())
        overlap = question_words & item_words
        # Need at least 3 meaningful overlapping words (excluding common words)
        common_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                        "to", "and", "or", "for", "on", "at", "by", "it", "this",
                        "that", "with", "from", "as", "be", "has", "have", "had",
                        "do", "does", "did", "not", "but", "what", "which", "how",
                        "when", "where", "who", "why", "can", "will", "would", "if"}
        meaningful_overlap = overlap - common_words
        if len(meaningful_overlap) >= 3:
            return True

    return False


def oracle_read_decision(
    question: str,
    answer_text: str,
    candidates: List[Tuple[int, float]],  # (canvas_id, sim_score)
    library: SimLibrary,
) -> Dict:
    """Oracle read decision: select canvases that contain answer information."""
    relevant_ids = []
    for cid, sim_score in candidates:
        items = library.get_items(cid)
        if answer_info_in_canvas(answer_text, question, items):
            relevant_ids.append(cid)

    if relevant_ids:
        return {
            "action": "RETRIEVE",
            "canvas_ids": relevant_ids[:2],
            "reason": f"canvases {relevant_ids[:2]} contain answer-relevant information",
        }
    elif candidates and candidates[0][1] > 0.5:
        # High similarity but no exact match → still retrieve top-1
        return {
            "action": "RETRIEVE",
            "canvas_ids": [candidates[0][0]],
            "reason": f"top candidate canvas #{candidates[0][0]} has high similarity ({candidates[0][1]:.2f})",
        }
    else:
        return {
            "action": "DIRECT_ANSWER",
            "reason": "no relevant canvases found, answer from general knowledge",
        }


# ---------------------------------------------------------------------------
# Write SFT Data Construction
# ---------------------------------------------------------------------------
def build_write_sft_data(
    train_data: List[Dict],
    max_samples: int = 2000,
    sim_threshold: float = 0.6,
) -> List[Dict]:
    """Construct Write SFT data with oracle labels."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Building Write SFT Data ({min(len(train_data), max_samples)} samples)")
    print(f"{'='*60}")

    # Load sentence embedder
    print("Loading sentence-transformers model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    train_data = train_data[:max_samples]
    library = SimLibrary()
    topic_embeddings: Dict[int, np.ndarray] = {}  # canvas_id → topic_embedding

    sft_data = []
    action_counts = Counter()
    MAX_DISPLAY_CANVASES = 15  # Only show top-K most relevant canvases in prompt

    for i, sample in enumerate(tqdm(train_data, desc="Write SFT oracle")):
        knowledge = sample_to_knowledge_text(sample)

        # Compute embedding for the new knowledge
        knowledge_emb = embedder.encode(knowledge, normalize_embeddings=True)

        # Get current canvas library state (what the agent sees)
        all_metas = library.get_metas()

        # Oracle decision
        decision = oracle_write_decision(
            library, knowledge, knowledge_emb, topic_embeddings, sim_threshold
        )
        action_counts[decision["action"]] += 1

        # For the prompt: only show top-K most relevant canvases (not all 800+)
        if len(all_metas) <= MAX_DISPLAY_CANVASES:
            display_metas = all_metas
        else:
            # Rank by similarity to new knowledge
            canvas_ids = list(topic_embeddings.keys())
            emb_matrix = np.array([topic_embeddings[cid] for cid in canvas_ids])
            sims = emb_matrix @ knowledge_emb
            top_idx = np.argsort(sims)[::-1][:MAX_DISPLAY_CANVASES]
            top_cids = set(canvas_ids[j] for j in top_idx)
            # Always include the target canvas if APPEND
            if decision["action"] == "APPEND":
                top_cids.add(decision["canvas_id"])
            display_metas = [m for m in all_metas if m.canvas_id in top_cids]

        # Build the SFT training sample
        user_prompt = build_write_prompt(display_metas, knowledge)

        # Format oracle output
        if decision["action"] == "CREATE":
            oracle_output = json.dumps({
                "action": "CREATE",
                "canvas_id": None,
                "topic": decision["topic"],
                "reason": decision["reason"],
            }, ensure_ascii=False)
        elif decision["action"] == "APPEND":
            oracle_output = json.dumps({
                "action": "APPEND",
                "canvas_id": decision["canvas_id"],
                "topic": None,
                "reason": decision["reason"],
            }, ensure_ascii=False)
        else:  # NOOP
            oracle_output = json.dumps({
                "action": "NOOP",
                "canvas_id": None,
                "topic": None,
                "reason": decision["reason"],
            }, ensure_ascii=False)

        sft_data.append({
            "instruction": WRITE_SYSTEM_PROMPT,
            "input": user_prompt,
            "output": oracle_output,
        })

        # Execute the oracle decision on the library
        if decision["action"] == "CREATE":
            cid = library.create(decision["topic"], knowledge)
            topic_emb = embedder.encode(
                decision["topic"] + " " + knowledge[:100],
                normalize_embeddings=True,
            )
            topic_embeddings[cid] = topic_emb
        elif decision["action"] == "APPEND":
            cid = library.append(decision["canvas_id"], knowledge)
            if cid != decision["canvas_id"]:
                # Overflowed → new canvas was created
                topic_emb = embedder.encode(
                    library.canvases[cid].topic + " " + knowledge[:100],
                    normalize_embeddings=True,
                )
                topic_embeddings[cid] = topic_emb
            else:
                # Update existing topic embedding with new content
                canvas = library.canvases[cid]
                combined = canvas.topic + " " + " ".join(canvas.items[-3:])[:200]
                topic_embeddings[cid] = embedder.encode(
                    combined, normalize_embeddings=True
                )
        # NOOP: no change to library

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}] Canvases: {len(library.canvases)}, "
                  f"Actions: {dict(action_counts)}")

    print(f"\nWrite SFT complete:")
    print(f"  Total samples: {len(sft_data)}")
    print(f"  Action distribution: {dict(action_counts)}")
    print(f"  Final canvases: {len(library.canvases)}")

    return sft_data, library, topic_embeddings, embedder


# ---------------------------------------------------------------------------
# Read SFT Data Construction
# ---------------------------------------------------------------------------
def build_read_sft_data(
    test_data: List[Dict],
    library: SimLibrary,
    topic_embeddings: Dict[int, np.ndarray],
    embedder,
    max_samples: int = 1000,
    top_k: int = 5,
) -> List[Dict]:
    """Construct Read SFT data with oracle labels."""
    print(f"\n{'='*60}")
    print(f"Building Read SFT Data ({min(len(test_data), max_samples)} samples)")
    print(f"{'='*60}")

    test_data = test_data[:max_samples]
    sft_data = []
    action_counts = Counter()

    # Pre-compute topic embedding matrix
    canvas_ids = list(topic_embeddings.keys())
    if not canvas_ids:
        print("  No canvases available, skipping Read SFT")
        return []

    emb_matrix = np.array([topic_embeddings[cid] for cid in canvas_ids])

    for i, sample in enumerate(tqdm(test_data, desc="Read SFT oracle")):
        question = sample["question"]
        hint = sample.get("hint", "")
        query = question + (" " + hint if hint else "")

        # Get correct answer text
        correct_idx = sample["answer"]
        choices = sample["choices"]
        answer_text = choices[correct_idx] if correct_idx < len(choices) else ""

        # Compute query embedding
        query_emb = embedder.encode(query, normalize_embeddings=True)

        # Get Top-K candidates
        similarities = emb_matrix @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates_for_oracle = []
        candidates_for_prompt = []
        for idx in top_indices:
            cid = canvas_ids[idx]
            sim = float(similarities[idx])
            if sim > 0.1:  # filter very low similarity
                candidates_for_oracle.append((cid, sim))
                meta = library.canvases[cid].to_meta()
                candidates_for_prompt.append((meta, sim))

        if not candidates_for_oracle:
            continue

        # Oracle decision
        decision = oracle_read_decision(
            question, answer_text, candidates_for_oracle, library
        )
        action_counts[decision["action"]] += 1

        # Build SFT sample
        user_prompt = build_read_prompt(question, candidates_for_prompt)

        oracle_output = json.dumps(decision, ensure_ascii=False)

        sft_data.append({
            "instruction": READ_SYSTEM_PROMPT,
            "input": user_prompt,
            "output": oracle_output,
        })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}] Actions: {dict(action_counts)}")

    print(f"\nRead SFT complete:")
    print(f"  Total samples: {len(sft_data)}")
    print(f"  Action distribution: {dict(action_counts)}")

    return sft_data


# ---------------------------------------------------------------------------
# Save in multiple formats
# ---------------------------------------------------------------------------
def save_sft_data(
    write_data: List[Dict],
    read_data: List[Dict],
    output_dir: Path,
):
    """Save SFT data in Alpaca JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Write SFT
    write_file = output_dir / "write_sft.json"
    with open(write_file, "w", encoding="utf-8") as f:
        json.dump(write_data, f, indent=2, ensure_ascii=False)
    print(f"  Write SFT: {len(write_data)} samples → {write_file}")

    # Save Read SFT
    read_file = output_dir / "read_sft.json"
    with open(read_file, "w", encoding="utf-8") as f:
        json.dump(read_data, f, indent=2, ensure_ascii=False)
    print(f"  Read SFT: {len(read_data)} samples → {read_file}")

    # Save combined
    combined = write_data + read_data
    combined_file = output_dir / "combined_sft.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  Combined: {len(combined)} samples → {combined_file}")

    # Also save as ShareGPT format (for ms-swift / LLaMA-Factory compatibility)
    sharegpt_write = []
    for d in write_data:
        sharegpt_write.append({
            "conversations": [
                {"from": "system", "value": d["instruction"]},
                {"from": "human", "value": d["input"]},
                {"from": "gpt", "value": d["output"]},
            ]
        })
    sharegpt_read = []
    for d in read_data:
        sharegpt_read.append({
            "conversations": [
                {"from": "system", "value": d["instruction"]},
                {"from": "human", "value": d["input"]},
                {"from": "gpt", "value": d["output"]},
            ]
        })
    sharegpt_combined = sharegpt_write + sharegpt_read

    with open(output_dir / "write_sft_sharegpt.json", "w", encoding="utf-8") as f:
        json.dump(sharegpt_write, f, indent=2, ensure_ascii=False)
    with open(output_dir / "read_sft_sharegpt.json", "w", encoding="utf-8") as f:
        json.dump(sharegpt_read, f, indent=2, ensure_ascii=False)
    with open(output_dir / "combined_sft_sharegpt.json", "w", encoding="utf-8") as f:
        json.dump(sharegpt_combined, f, indent=2, ensure_ascii=False)
    print(f"  ShareGPT format saved (for ms-swift / LLaMA-Factory)")

    # Statistics
    stats = {
        "write_samples": len(write_data),
        "read_samples": len(read_data),
        "total_samples": len(combined),
        "write_action_dist": dict(Counter(
            json.loads(d["output"])["action"] for d in write_data
        )),
        "read_action_dist": dict(Counter(
            json.loads(d["output"])["action"] for d in read_data
        )),
    }
    with open(output_dir / "sft_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare Agent SFT training data")
    parser.add_argument("--max-train", type=int, default=2000,
                        help="Max training samples for Write SFT")
    parser.add_argument("--max-test", type=int, default=1000,
                        help="Max test samples for Read SFT")
    parser.add_argument("--sim-threshold", type=float, default=0.45,
                        help="Similarity threshold for APPEND decision")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Load ScienceQA
    print("Loading ScienceQA data...")
    train_data, test_data = load_scienceqa()
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Build Write SFT data
    write_data, library, topic_embeddings, embedder = build_write_sft_data(
        train_data,
        max_samples=args.max_train,
        sim_threshold=args.sim_threshold,
    )

    # Build Read SFT data (using oracle library built above)
    read_data = build_read_sft_data(
        test_data,
        library,
        topic_embeddings,
        embedder,
        max_samples=args.max_test,
    )

    # Save all data
    print(f"\n{'='*60}")
    print("Saving SFT data...")
    print(f"{'='*60}")
    save_sft_data(write_data, read_data, output_dir)

    print(f"\nDone! SFT data saved to {output_dir}")


if __name__ == "__main__":
    main()
