#!/usr/bin/env python3
"""
prepare_data.py — Build GRPO training data in verl-compatible parquet format

Converts reward table + ScienceQA data into parquet with columns:
  - data_source: "layout_agent"
  - prompt: chat messages (content block description → agent decides layout)
  - reward_model: {ground_truth: JSON with qa_table + block_info}
  - extra_info: {pid, index, question}

Usage:
    python -m layout_agent.prepare_data
"""

import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .strategies import (
    SemanticRole, adapt_scienceqa, build_agent_prompt,
    STRATEGY_NAMES, LayoutParams,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "/home/cyf/codex/layout_agent/rl_data"
REWARD_TABLE_PATH = "/home/cyf/codex/layout_agent/reward_data/reward_table.json"
SCIQA_CACHE = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"

SYSTEM_PROMPT = (
    "You are a canvas layout optimization agent. "
    "Given a description of content blocks (text, images, tables), "
    "decide the best layout configuration to maximize readability and QA comprehension. "
    "Output ONLY a JSON object with these keys: "
    "strategy, width_class, image_scale, content_priority, truncation, font_profile."
)


# ---------------------------------------------------------------------------
# Build prompt from problem
# ---------------------------------------------------------------------------

def build_prompt_messages(problem: Dict, has_image: bool) -> List[Dict]:
    """Build chat messages for the layout agent prompt.

    Note: The canvas will contain ONLY knowledge/context/image (no question/solution).
    The agent must optimize layout for comprehension when the question is asked separately.
    """
    # Build generic content block description (knowledge-only)
    blocks_desc = []
    idx = 0

    subj = problem.get("subject", "")
    topic = problem.get("topic", "")
    if subj or topic:
        blocks_desc.append(f"  #{idx} [text/header] ~20 chars, font=14")
        idx += 1

    hint = problem.get("hint", "")
    if hint:
        blocks_desc.append(f"  #{idx} [text/context] {len(hint)} chars, font=16")
        idx += 1

    if has_image:
        blocks_desc.append(f"  #{idx} [image/figure] ~300x200, ar=1.50")
        idx += 1

    lecture = problem.get("lecture", "")
    if lecture:
        blocks_desc.append(f"  #{idx} [text/knowledge] {len(lecture)} chars, font=15")
        idx += 1

    # Note: question/choices/solution are NOT on the canvas
    # They will be asked separately after the canvas is shown

    user_content = (
        "Content blocks:\n"
        + "\n".join(blocks_desc)
        + "\n\nAvailable strategies: " + ", ".join(STRATEGY_NAMES)
        + "\n\nChoose the optimal layout. Output JSON:"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def build_block_info(problem: Dict, has_image: bool) -> Dict:
    """Build block_info metadata for reward computation (knowledge-only canvas)."""
    total_chars = 0
    total_chars += len(problem.get("hint", ""))
    total_chars += len(problem.get("lecture", ""))
    # question/solution NOT on canvas

    return {
        "total_chars": total_chars,
        "has_image": has_image,
        "n_blocks": sum([
            1 if problem.get("hint", "") else 0,
            1 if has_image else 0,
            1 if problem.get("lecture", "") else 0,
            1 if problem.get("subject", "") else 0,
        ]),
        "has_knowledge": bool(problem.get("lecture", "")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load reward table
    print("Loading reward table...")
    with open(REWARD_TABLE_PATH) as f:
        rt_data = json.load(f)
    reward_table = rt_data["reward_table"]
    strategy_stats = rt_data.get("strategy_stats", {})

    print(f"  {len(reward_table)} samples in reward table")
    print("  Strategy accuracies:")
    for s in STRATEGY_NAMES:
        st = strategy_stats.get(s, {})
        print(f"    {s:20s}: {st.get('accuracy', 0)*100:5.1f}%")

    # Load ScienceQA data
    print("\nLoading ScienceQA data...")
    with open(SCIQA_CACHE, "rb") as f:
        cache = pickle.load(f)
    if isinstance(cache, dict):
        train_data = cache["train"]
    elif isinstance(cache, (list, tuple)) and len(cache) == 2:
        train_data, _ = cache
    else:
        train_data = cache
    print(f"  {len(train_data)} training samples")

    # Load HF dataset to check which samples have images
    image_pids = set()
    try:
        from datasets import load_dataset
        hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")
        for i in range(len(hf_ds)):
            if hf_ds[i].get("image") is not None:
                image_pids.add(i)
        print(f"  {len(image_pids)} samples have images")
    except Exception as e:
        print(f"  HF dataset not available: {e}")

    # Build training data (only for samples in reward table)
    train_records = []
    for pid_str, qa_table in reward_table.items():
        pid = int(pid_str)
        if pid >= len(train_data):
            continue

        problem = train_data[pid]
        has_image = pid in image_pids

        # Build prompt
        messages = build_prompt_messages(problem, has_image)

        # Build ground truth for reward function
        block_info = build_block_info(problem, has_image)
        ground_truth = json.dumps({
            "qa_table": qa_table,
            "block_info": block_info,
        })

        train_records.append({
            "data_source": "layout_agent",
            "prompt": messages,
            "ability": "layout",
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
            "extra_info": {
                "pid": pid_str,
                "index": pid,
                "question": problem.get("question", ""),
            },
        })

    random.seed(42)
    random.shuffle(train_records)

    # Split: 90% train, 10% val
    split = int(len(train_records) * 0.9)
    train_split = train_records[:split]
    val_split = train_records[split:]

    # Save as parquet
    train_path = os.path.join(OUTPUT_DIR, "layout_train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "layout_val.parquet")

    pd.DataFrame(train_split).to_parquet(train_path, index=False)
    pd.DataFrame(val_split).to_parquet(val_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_split)} samples)")
    print(f"  Val:   {val_path} ({len(val_split)} samples)")

    # Also build SFT warmup data (best layout per sample as label)
    sft_records = []
    for pid_str, qa_table in reward_table.items():
        pid = int(pid_str)
        if pid >= len(train_data):
            continue

        problem = train_data[pid]
        has_image = bool(problem.get("image", ""))
        messages = build_prompt_messages(problem, has_image)

        # Find best layout by QA accuracy (tie-break by quality heuristic)
        if qa_table:
            best_config = max(qa_table.keys(), key=lambda k: qa_table[k])
            # Parse config_hash back to params
            parts = best_config.split("_")
            # Strategy names can have underscores, need to match from STRATEGY_NAMES
            strategy = None
            rest = best_config
            for s in sorted(STRATEGY_NAMES, key=len, reverse=True):
                if rest.startswith(s + "_"):
                    strategy = s
                    rest = rest[len(s) + 1:]
                    break
            if strategy is None:
                continue

            rest_parts = rest.split("_")
            if len(rest_parts) >= 5:
                best_json = json.dumps({
                    "strategy": strategy,
                    "width_class": rest_parts[0],
                    "image_scale": rest_parts[1],
                    "content_priority": rest_parts[2],
                    "truncation": rest_parts[3],
                    "font_profile": rest_parts[4],
                })
            else:
                best_json = json.dumps({"strategy": strategy, "width_class": rest_parts[0] if rest_parts else "medium"})

            sft_records.append({
                "messages": messages + [{"role": "assistant", "content": best_json}],
                "pid": pid_str,
            })

    sft_path = os.path.join(OUTPUT_DIR, "layout_sft.json")
    with open(sft_path, "w") as f:
        json.dump(sft_records, f, indent=1)
    print(f"  SFT:   {sft_path} ({len(sft_records)} samples)")


if __name__ == "__main__":
    main()
