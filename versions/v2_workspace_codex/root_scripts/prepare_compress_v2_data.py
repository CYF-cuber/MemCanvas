#!/usr/bin/env python3
"""
Prepare compressor GRPO v2 training data in verl Parquet format.

Same as prepare_compress_rl_data.py but adds canvas rendering params
(font_size, canvas_width, ref_width) to ground truth for VLM readability reward.

Output:
  /home/cyf/codex/compress_v2_data/compress_train.parquet
  /home/cyf/codex/compress_v2_data/compress_val.parquet

Usage:
  python prepare_compress_v2_data.py [--max-train 5000] [--val-ratio 0.1]
"""

import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = Path("/home/cyf/codex/compress_v2_data")

# Canvas0415 parameters
FONT_SIZE = 16
CANVAS_WIDTH = 830
REF_WIDTH = 800

COMPRESS_SYSTEM_PROMPT = (
    "You are a text compressor. Given a passage of text, compress it into "
    "the shortest possible form while preserving ALL key factual information "
    "needed to answer questions. Use concise notation: "
    "'entity: key facts' format. Remove filler, background, and redundant info. "
    "IMPORTANT: The compressed text will be rendered on a canvas image and read by a VLM. "
    "Keep text clear and well-structured so it remains readable when rendered. "
    "Output only the compressed text, nothing else."
)

COMPRESS_USER_TEMPLATE = "Compress the following text:\n\n{text}\n\nCompressed:"


def extract_key_facts(sample):
    """Extract key facts from a ScienceQA sample for reward computation."""
    facts = []
    answer_text = sample["choices"][sample["answer"]]
    facts.append(answer_text)

    lecture = sample.get("lecture", "") or ""
    if lecture:
        for sent in lecture.split("."):
            sent = sent.strip()
            if sent and answer_text.lower() in sent.lower():
                facts.append(sent)
                break

    solution = sample.get("solution", "") or ""
    if solution:
        for sent in solution.split("."):
            sent = sent.strip()
            if sent and answer_text.lower() in sent.lower():
                facts.append(sent)
                break

    return facts


def build_compressor_data(train_data, max_samples=5000, val_ratio=0.1):
    """Build training/val data for compressor GRPO v2."""
    samples = []

    for i, item in enumerate(tqdm(train_data[:max_samples], desc="Building data")):
        lecture = (item.get("lecture", "") or "").strip()
        solution = (item.get("solution", "") or "").strip()

        if not lecture and not solution:
            continue

        text_to_compress = ""
        if lecture:
            text_to_compress += f"Background: {lecture}\n"
        if solution:
            text_to_compress += f"Solution: {solution}"
        text_to_compress = text_to_compress.strip()

        if len(text_to_compress) < 50:
            continue

        answer_text = item["choices"][item["answer"]]
        key_facts = extract_key_facts(item)
        question = item["question"]

        gt = json.dumps({
            "answer": answer_text,
            "key_facts": key_facts,
            "original_length": len(text_to_compress),
            "question": question,
            # Canvas rendering params for VLM readability reward
            "font_size": FONT_SIZE,
            "canvas_width": CANVAS_WIDTH,
            "ref_width": REF_WIDTH,
        })

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

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def save_parquet(samples, path):
    """Save samples in verl Parquet format.

    IMPORTANT: prompt must be a native list of dicts, NOT a JSON string.
    verl expects Arrow list<struct<content, role>> format.
    """
    rows = []
    for s in samples:
        rows.append({
            "data_source": s["data_source"],
            "prompt": s["prompt"],  # keep as list of dicts (native arrow format)
            "reward_model": s["reward_model"],
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} samples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ScienceQA training data...")
    cache_path = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    train_data = cache["train"] if isinstance(cache, dict) else cache[0]
    print(f"  {len(train_data)} total training samples")

    train_samples, val_samples = build_compressor_data(
        train_data, max_samples=args.max_train, val_ratio=args.val_ratio
    )
    print(f"\n  Train: {len(train_samples)}, Val: {len(val_samples)}")

    save_parquet(train_samples, OUTPUT_DIR / "compress_train.parquet")
    save_parquet(val_samples, OUTPUT_DIR / "compress_val.parquet")

    # Show sample
    s = train_samples[0]
    prompt = json.loads(s["prompt"]) if isinstance(s["prompt"], str) else s["prompt"]
    print(f"\n--- Sample ---")
    print(f"System: {prompt[0]['content'][:80]}...")
    print(f"User: {prompt[1]['content'][:150]}...")
    gt = json.loads(json.loads(s["reward_model"])["ground_truth"]) if isinstance(s["reward_model"], str) else json.loads(s["reward_model"]["ground_truth"])
    print(f"Answer: {gt['answer']}")
    print(f"Key facts: {gt['key_facts'][:3]}")
    print(f"Font size: {gt['font_size']}, Canvas width: {gt['canvas_width']}")

    print(f"\nDone! Data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
