#!/usr/bin/env python3
"""
Prepare Read RL training data in verl Parquet format.

Uses oracle write policy to build a canvas library from training samples,
then constructs read prompts for test samples with ground truth labels
for the GRPO reward function.

Output:
  /home/cyf/codex/agent_rl_data/read_train.parquet
  /home/cyf/codex/agent_rl_data/read_val.parquet

Usage:
  python prepare_read_rl_data.py [--max-train 2000] [--max-test 4241]
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
sys.path.insert(0, "/home/cyf/codex")

from canvas_agent import (
    READ_SYSTEM_PROMPT,
    CanvasMeta,
    build_read_prompt,
)
from prepare_agent_sft_data import (
    SimCanvas,
    SimLibrary,
    answer_info_in_canvas,
    extract_topic,
    is_duplicate,
    load_scienceqa,
    oracle_write_decision,
    sample_to_knowledge_text,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/cyf/codex/agent_rl_data")
CACHE_DIR = Path("/home/cyf/codex/agent_experiment_output")


# ---------------------------------------------------------------------------
# Step 1: Build oracle canvas library from training data
# ---------------------------------------------------------------------------
def build_oracle_library(
    train_data: List[Dict],
    max_samples: int = 2000,
    sim_threshold: float = 0.45,
) -> Tuple[SimLibrary, Dict[int, np.ndarray], "SentenceTransformer"]:
    """Build oracle canvas library from training samples."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Building Oracle Canvas Library ({min(len(train_data), max_samples)} samples)")
    print(f"{'='*60}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    train_data = train_data[:max_samples]
    library = SimLibrary()
    topic_embeddings: Dict[int, np.ndarray] = {}
    action_counts = Counter()

    for i, sample in enumerate(tqdm(train_data, desc="Oracle write")):
        knowledge = sample_to_knowledge_text(sample)
        knowledge_emb = embedder.encode(knowledge, normalize_embeddings=True)

        decision = oracle_write_decision(
            library, knowledge, knowledge_emb, topic_embeddings, sim_threshold
        )
        action_counts[decision["action"]] += 1

        # Execute decision
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
                topic_emb = embedder.encode(
                    library.canvases[cid].topic + " " + knowledge[:100],
                    normalize_embeddings=True,
                )
                topic_embeddings[cid] = topic_emb
            else:
                canvas = library.canvases[cid]
                combined = canvas.topic + " " + " ".join(canvas.items[-3:])[:200]
                topic_embeddings[cid] = embedder.encode(
                    combined, normalize_embeddings=True
                )
        # NOOP: no change

        if (i + 1) % 500 == 0:
            print(
                f"  [{i+1}] Canvases: {len(library.canvases)}, "
                f"Actions: {dict(action_counts)}"
            )

    print(f"\nOracle library built:")
    print(f"  Canvases: {len(library.canvases)}")
    print(f"  Action distribution: {dict(action_counts)}")

    return library, topic_embeddings, embedder


# ---------------------------------------------------------------------------
# Step 2: Build read RL data for test samples
# ---------------------------------------------------------------------------
def build_read_rl_data(
    test_data: List[Dict],
    library: SimLibrary,
    topic_embeddings: Dict[int, np.ndarray],
    embedder,
    top_k: int = 5,
) -> List[Dict]:
    """Build read RL data with ground truth for reward computation."""
    print(f"\n{'='*60}")
    print(f"Building Read RL Data ({len(test_data)} samples)")
    print(f"{'='*60}")

    canvas_ids = list(topic_embeddings.keys())
    if not canvas_ids:
        print("  No canvases available!")
        return []

    emb_matrix = np.array([topic_embeddings[cid] for cid in canvas_ids])
    rl_data = []
    stats = Counter()

    for i, sample in enumerate(tqdm(test_data, desc="Read RL data")):
        question = sample["question"]
        hint = sample.get("hint", "")
        query = question + (" " + hint if hint else "")

        # Correct answer text
        correct_idx = sample["answer"]
        choices = sample["choices"]
        answer_text = choices[correct_idx] if correct_idx < len(choices) else ""

        # Compute query embedding and get top-K candidates
        query_emb = embedder.encode(query, normalize_embeddings=True)
        similarities = emb_matrix @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidate_ids = []
        candidates_for_prompt = []
        for idx in top_indices:
            cid = canvas_ids[idx]
            sim = float(similarities[idx])
            if sim > 0.1:
                candidate_ids.append(cid)
                meta = library.canvases[cid].to_meta()
                candidates_for_prompt.append((meta, sim))

        if not candidate_ids:
            stats["skipped_no_candidates"] += 1
            continue

        # Determine oracle_ids: which candidates actually contain answer info
        oracle_ids = []
        for cid in candidate_ids:
            items = library.get_items(cid)
            if answer_info_in_canvas(answer_text, question, items):
                oracle_ids.append(cid)

        has_relevant = len(oracle_ids) > 0
        if has_relevant:
            stats["has_relevant"] += 1
        else:
            stats["no_relevant"] += 1

        # Build the read prompt (same format as SFT/inference)
        user_prompt = build_read_prompt(question, candidates_for_prompt)

        # Construct verl-format data point
        data_point = {
            "data_source": "memcanvas_read",
            "prompt": [
                {"role": "system", "content": READ_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "ability": "retrieval",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "oracle_ids": oracle_ids,
                    "has_relevant": has_relevant,
                    "candidate_ids": candidate_ids,
                }),
            },
            "extra_info": {
                "pid": sample.get("pid", str(i)),
                "index": i,
                "answer_text": answer_text,
                "question": question,
            },
        }
        rl_data.append(data_point)

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}] Samples: {len(rl_data)}, Stats: {dict(stats)}")

    print(f"\nRead RL data built:")
    print(f"  Total samples: {len(rl_data)}")
    print(f"  Stats: {dict(stats)}")

    return rl_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare Read RL training data")
    parser.add_argument(
        "--max-train", type=int, default=2000,
        help="Max training samples for oracle library"
    )
    parser.add_argument(
        "--max-test", type=int, default=0,
        help="Max test samples (0 = all)"
    )
    parser.add_argument(
        "--val-size", type=int, default=541,
        help="Validation split size"
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.45,
        help="Similarity threshold for oracle write APPEND"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-K candidates for read prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for parquet files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ScienceQA
    print("Loading ScienceQA data...")
    train_data, test_data = load_scienceqa()
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    if args.max_test > 0:
        test_data = test_data[: args.max_test]

    # Build oracle canvas library
    library, topic_embeddings, embedder = build_oracle_library(
        train_data,
        max_samples=args.max_train,
        sim_threshold=args.sim_threshold,
    )

    # Build read RL data
    rl_data = build_read_rl_data(
        test_data, library, topic_embeddings, embedder, top_k=args.top_k
    )

    # Train / val split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(rl_data))
    val_size = min(args.val_size, len(rl_data) // 5)
    val_indices = set(indices[:val_size])

    train_rows = []
    val_rows = []
    for idx, row in enumerate(rl_data):
        if idx in val_indices:
            val_rows.append(row)
        else:
            train_rows.append(row)

    print(f"\nSplit: train={len(train_rows)}, val={len(val_rows)}")

    # Save as parquet
    for split_name, rows in [("read_train", train_rows), ("read_val", val_rows)]:
        # Convert to DataFrame-friendly format (lists/dicts stay as-is in parquet)
        df = pd.DataFrame(rows)
        parquet_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved {len(rows)} rows → {parquet_path}")

    # Save metadata
    meta = {
        "num_canvases": len(library.canvases),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "total_test": len(test_data),
        "top_k": args.top_k,
        "sim_threshold": args.sim_threshold,
    }
    with open(output_dir / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Also save the oracle library for later use in evaluation
    with open(output_dir / "oracle_library.pkl", "wb") as f:
        pickle.dump(
            {
                "library": library,
                "topic_embeddings": topic_embeddings,
            },
            f,
        )

    print(f"\nDone! Read RL data saved to {output_dir}")
    print(f"  Metadata: {meta}")


if __name__ == "__main__":
    main()
