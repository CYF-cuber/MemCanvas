#!/usr/bin/env python3
"""
Prepare compressor GRPO training data in verl Parquet format.

Uses ScienceQA training data: for each sample with lecture/solution text,
create a prompt asking the model to compress it, with ground truth for reward.

Output:
  /home/cyf/codex/compress_rl_data/compress_train.parquet
  /home/cyf/codex/compress_rl_data/compress_val.parquet

Usage:
  python prepare_compress_rl_data.py [--max-train 5000] [--val-ratio 0.1]
"""

import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = Path("/home/cyf/codex/compress_rl_data")

COMPRESS_SYSTEM_PROMPT = (
    "You are a text compressor. Given a passage of text, compress it into "
    "the shortest possible form while preserving ALL key factual information "
    "needed to answer questions. Use concise notation: "
    "'entity: key facts' format. Remove filler, background, and redundant info. "
    "Output only the compressed text, nothing else."
)

COMPRESS_USER_TEMPLATE = "Compress the following text:\n\n{text}\n\nCompressed:"


def extract_key_facts(sample):
    """Extract key facts from a ScienceQA sample for reward computation."""
    facts = []
    answer_text = sample["choices"][sample["answer"]]
    facts.append(answer_text)

    # Extract key terms from lecture
    lecture = sample.get("lecture", "") or ""
    if lecture:
        # Simple extraction: sentences containing the answer
        for sent in lecture.split("."):
            sent = sent.strip()
            if sent and answer_text.lower() in sent.lower():
                facts.append(sent)
                break

    # Extract from solution
    solution = sample.get("solution", "") or ""
    if solution:
        for sent in solution.split("."):
            sent = sent.strip()
            if sent and answer_text.lower() in sent.lower():
                facts.append(sent)
                break

    return facts


def build_compressor_data(train_data, max_samples=5000, val_ratio=0.1):
    """Build training/val data for compressor GRPO."""
    samples = []

    for i, item in enumerate(tqdm(train_data[:max_samples], desc="Building data")):
        # Get text to compress (lecture + solution)
        lecture = (item.get("lecture", "") or "").strip()
        solution = (item.get("solution", "") or "").strip()

        if not lecture and not solution:
            continue

        # Combine text to compress
        text_to_compress = ""
        if lecture:
            text_to_compress += f"Background: {lecture}\n"
        if solution:
            text_to_compress += f"Solution: {solution}"
        text_to_compress = text_to_compress.strip()

        if len(text_to_compress) < 50:
            continue

        # Ground truth for reward
        answer_text = item["choices"][item["answer"]]
        key_facts = extract_key_facts(item)
        question = item["question"]

        gt = json.dumps({
            "answer": answer_text,
            "key_facts": key_facts,
            "original_length": len(text_to_compress),
            "question": question,
        })

        # Build prompt in chat format
        prompt = [
            {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
            {"role": "user", "content": COMPRESS_USER_TEMPLATE.format(text=text_to_compress[:800])},
        ]

        samples.append({
            "data_source": "memcanvas_compress",
            "prompt": prompt,
            "reward_model": {
                "style": "rule",
                "ground_truth": gt,
            },
        })

    # Shuffle and split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def save_parquet(samples, path):
    """Save samples in verl Parquet format."""
    rows = []
    for s in samples:
        rows.append({
            "data_source": s["data_source"],
            "prompt": json.dumps(s["prompt"]),
            "reward_model": json.dumps(s["reward_model"]),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} samples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ScienceQA
    print("Loading ScienceQA training data...")
    cache_path = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    train_data = cache["train"] if isinstance(cache, dict) else cache[0]
    print(f"  {len(train_data)} total training samples")

    # Build data
    train_samples, val_samples = build_compressor_data(
        train_data, max_samples=args.max_train, val_ratio=args.val_ratio
    )
    print(f"\n  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Save
    save_parquet(train_samples, OUTPUT_DIR / "compress_train.parquet")
    save_parquet(val_samples, OUTPUT_DIR / "compress_val.parquet")

    # Show sample
    s = train_samples[0]
    print(f"\n--- Sample prompt ---")
    prompt = json.loads(s["prompt"]) if isinstance(s["prompt"], str) else s["prompt"]
    print(f"System: {prompt[0]['content'][:100]}...")
    print(f"User: {prompt[1]['content'][:200]}...")
    gt = json.loads(json.loads(s["reward_model"])["ground_truth"]) if isinstance(s["reward_model"], str) else json.loads(s["reward_model"]["ground_truth"])
    print(f"Answer: {gt['answer']}")
    print(f"Key facts: {gt['key_facts'][:3]}")

    print(f"\nDone! Data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
