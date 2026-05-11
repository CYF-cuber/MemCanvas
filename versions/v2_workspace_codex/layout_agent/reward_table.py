#!/usr/bin/env python3
"""
reward_table.py — Pre-compute QA accuracy for all candidate layouts

Pipeline:
  1. Stratified sample 2000 training problems
  2. For each: generate 24 candidate canvases (8 strategies × 3 widths)
  3. Reader VLM (Qwen2.5-VL-7B) answers QA for each canvas
  4. Store (pid, config_hash) → qa_accuracy as JSON

Usage:
    python -m layout_agent.reward_table --phase render   # Generate 48K canvas PNGs
    python -m layout_agent.reward_table --phase evaluate  # Batch VLM QA
    python -m layout_agent.reward_table --phase build     # Build reward table
"""

import argparse
import io
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from layout_agent.strategies import (
    STRATEGY_NAMES, LayoutParams, ContentBlock,
    adapt_scienceqa, adapt_scienceqa_knowledge_only,
    generate_layout, render_layout, compute_quality_score,
    generate_all_candidates,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "/home/cyf/codex/layout_agent/reward_data"
SCIQA_CACHE = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"
READER_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
NUM_SAMPLES = 2000
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Phase 0: Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(problems: List[Dict], num_samples: int = NUM_SAMPLES,
                      seed: int = 42) -> List[Tuple[int, Dict]]:
    """Sample from training set, stratified by subject."""
    by_subject = defaultdict(list)
    for i, p in enumerate(problems):
        by_subject[p.get("subject", "other")].append((i, p))

    random.seed(seed)
    per_subject = max(num_samples // len(by_subject), 1)
    sampled = []

    for subj, items in by_subject.items():
        n = min(per_subject, len(items))
        sampled.extend(random.sample(items, n))

    # Fill remainder
    sampled_indices = set(i for i, _ in sampled)
    all_remaining = [(i, p) for i, p in enumerate(problems)
                     if i not in sampled_indices]
    if len(sampled) < num_samples and all_remaining:
        extra = random.sample(all_remaining, min(num_samples - len(sampled),
                                                  len(all_remaining)))
        sampled.extend(extra)

    return sampled[:num_samples]


# ---------------------------------------------------------------------------
# Phase 1: Render all candidate canvases
# ---------------------------------------------------------------------------

def render_candidates(problems: List[Dict], hf_dataset=None,
                      output_dir: str = OUTPUT_DIR, num_samples: int = NUM_SAMPLES):
    """Render 24 candidate canvases per sample, save as PNG."""
    canvas_dir = os.path.join(output_dir, "canvases")
    meta_path = os.path.join(output_dir, "render_meta.json")
    os.makedirs(canvas_dir, exist_ok=True)

    # Load existing progress
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        done_pids = set(meta.get("done_pids", []))
        records = meta.get("records", [])
    else:
        done_pids = set()
        records = []

    sampled = stratified_sample(problems, num_samples=num_samples)
    print(f"Rendering canvases for {len(sampled)} samples "
          f"({len(done_pids)} already done)")

    for si, (idx, problem) in enumerate(sampled):
        pid = str(idx)
        if pid in done_pids:
            continue

        # Get image from HF dataset if available
        img = None
        if hf_dataset is not None:
            try:
                hf_idx = problem.get("hf_index", idx)
                sample = hf_dataset[hf_idx]
                if sample.get("image") is not None:
                    img = sample["image"].convert("RGB")
            except Exception:
                pass

        # Convert to knowledge-only blocks (no question/choices/solution on canvas)
        # QA question is asked separately — layout quality affects comprehension
        blocks = adapt_scienceqa_knowledge_only(problem, img)
        if not blocks:
            continue

        # Generate all 24 candidates
        candidates = generate_all_candidates(blocks)

        for params, layout in candidates:
            config_hash = params.config_hash()
            canvas = render_layout(layout)
            quality = compute_quality_score(layout)

            # Save canvas
            fname = f"{pid}_{config_hash}.png"
            fpath = os.path.join(canvas_dir, fname)
            canvas.save(fpath, "PNG")

            # Build QA prompt info
            answer_idx = problem.get("answer", 0)
            choices = problem.get("choices", [])
            answer_text = choices[answer_idx] if answer_idx < len(choices) else ""
            answer_letter = chr(65 + answer_idx) if answer_idx < len(choices) else "A"

            records.append({
                "pid": pid,
                "config_hash": config_hash,
                "canvas_path": fpath,
                "quality_score": quality,
                "question": problem.get("question", ""),
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": answer_letter,
                "answer_text": answer_text,
                "canvas_w": layout.width,
                "canvas_h": layout.height,
            })

        done_pids.add(pid)

        if (si + 1) % 50 == 0 or si == len(sampled) - 1:
            # Checkpoint
            with open(meta_path, "w") as f:
                json.dump({"done_pids": list(done_pids), "records": records}, f)
            print(f"  [{si+1}/{len(sampled)}] {pid} → {len(records)} canvases total")

    # Final save
    with open(meta_path, "w") as f:
        json.dump({"done_pids": list(done_pids), "records": records}, f)
    print(f"\nDone: {len(records)} canvases for {len(done_pids)} samples")
    return records


# ---------------------------------------------------------------------------
# Phase 2: Batch VLM QA evaluation
# ---------------------------------------------------------------------------

def load_reader_vlm(model_path: str = READER_MODEL, device: str = "cuda"):
    """Load Qwen2.5-VL-7B for QA evaluation."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Reader VLM from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Reader VLM loaded")
    return model, processor


def build_qa_prompt(question: str, choices: List[str]) -> str:
    """Build QA prompt — question asked separately from canvas content."""
    q_text = (
        "The image above shows reference material (background knowledge, context, and/or figures). "
        "Use ONLY the information in the image to answer the following question.\n\n"
        f"Q: {question}\n"
    )
    for i, c in enumerate(choices):
        q_text += f"  {chr(65+i)}. {c}\n"
    q_text += "\nAnswer with ONLY the letter (A, B, C, etc.)."
    return q_text


def evaluate_batch(model, processor, records: List[Dict],
                   batch_size: int = BATCH_SIZE, device: str = "cuda",
                   checkpoint_path: Optional[str] = None,
                   existing_evaluated: Optional[Dict] = None) -> List[Dict]:
    """Evaluate QA accuracy for a batch of canvas-question pairs.

    Args:
        checkpoint_path: If provided, save incremental results every 500 records.
        existing_evaluated: Dict of already-evaluated results to merge with.
    """
    results = []
    evaluated = dict(existing_evaluated) if existing_evaluated else {}

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_results = []

        for rec in batch:
            canvas_path = rec["canvas_path"]
            if not os.path.exists(canvas_path):
                batch_results.append({**rec, "predicted": "", "correct": False})
                continue

            canvas_img = Image.open(canvas_path).convert("RGB")
            qa_prompt = build_qa_prompt(rec["question"], rec["choices"])

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": canvas_img},
                    {"type": "text", "text": qa_prompt},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            inputs = processor(text=[text], images=[canvas_img],
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=16,
                                        do_sample=False, temperature=None, top_p=None)

            prompt_len = inputs["input_ids"].shape[1]
            response = processor.tokenizer.decode(output[0][prompt_len:],
                                                   skip_special_tokens=True).strip()

            # Extract answer letter
            predicted = ""
            for ch in response.upper():
                if ch in "ABCDEFGH":
                    predicted = ch
                    break

            correct = predicted == rec["answer_letter"]
            batch_results.append({
                **rec,
                "predicted": predicted,
                "correct": correct,
            })

        results.extend(batch_results)

        if (i // batch_size + 1) % 10 == 0:
            done = min(i + batch_size, len(records))
            acc = sum(r["correct"] for r in results) / len(results) * 100
            print(f"  [{done}/{len(records)}] running acc={acc:.1f}%")

        # Incremental checkpoint every 500 records
        if checkpoint_path and len(results) % 500 < batch_size:
            for r in results[-(len(batch_results)):]:
                key = f"{r['pid']}_{r['config_hash']}"
                evaluated[key] = r
            all_so_far = list(evaluated.values())
            with open(checkpoint_path, "w") as f:
                json.dump(all_so_far, f)
            print(f"  [checkpoint] saved {len(all_so_far)} results to {checkpoint_path}")

    return results


def run_evaluation(output_dir: str = OUTPUT_DIR, device: str = "cuda",
                   batch_size: int = BATCH_SIZE):
    """Run VLM QA evaluation on all rendered canvases."""
    meta_path = os.path.join(output_dir, "render_meta.json")
    eval_path = os.path.join(output_dir, "eval_results.json")

    with open(meta_path) as f:
        meta = json.load(f)
    records = meta["records"]

    # Load existing eval results for resumption
    evaluated = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            existing = json.load(f)
        for r in existing:
            key = f"{r['pid']}_{r['config_hash']}"
            evaluated[key] = r
        print(f"Resuming: {len(evaluated)} already evaluated")

    # Filter out already-evaluated
    todo = [r for r in records
            if f"{r['pid']}_{r['config_hash']}" not in evaluated]
    print(f"Total: {len(records)}, Todo: {len(todo)}")

    if not todo:
        print("All evaluations complete!")
        return list(evaluated.values())

    model, processor = load_reader_vlm(device=device)

    t0 = time.time()
    new_results = evaluate_batch(model, processor, todo, batch_size=batch_size,
                                 device=device, checkpoint_path=eval_path,
                                 existing_evaluated=evaluated)
    elapsed = time.time() - t0

    # Merge with existing
    for r in new_results:
        key = f"{r['pid']}_{r['config_hash']}"
        evaluated[key] = r

    all_results = list(evaluated.values())
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=1)

    correct = sum(r["correct"] for r in all_results)
    print(f"\nEvaluation done: {correct}/{len(all_results)} correct "
          f"({correct/len(all_results)*100:.1f}%) in {elapsed:.0f}s")

    return all_results


# ---------------------------------------------------------------------------
# Phase 3: Build reward table
# ---------------------------------------------------------------------------

def build_reward_table(output_dir: str = OUTPUT_DIR):
    """Build the (pid, config_hash) → qa_accuracy lookup table."""
    eval_path = os.path.join(output_dir, "eval_results.json")
    table_path = os.path.join(output_dir, "reward_table.json")

    with open(eval_path) as f:
        results = json.load(f)

    # Group by pid
    by_pid = defaultdict(list)
    for r in results:
        by_pid[r["pid"]].append(r)

    # Build table: for each (pid, config_hash) → accuracy
    # Since each (pid, config_hash) is evaluated once, accuracy is 0 or 1
    # We also compute per-strategy averages for smoothing
    reward_table = {}
    strategy_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for pid, pid_results in by_pid.items():
        pid_table = {}
        for r in pid_results:
            acc = 1.0 if r["correct"] else 0.0
            pid_table[r["config_hash"]] = acc

            # Strategy-level stats
            strategy = r["config_hash"].split("_")[0]
            # Handle multi-word strategies like side_by_side
            for s in STRATEGY_NAMES:
                if r["config_hash"].startswith(s):
                    strategy = s
                    break
            strategy_stats[strategy]["total"] += 1
            if r["correct"]:
                strategy_stats[strategy]["correct"] += 1

        reward_table[pid] = pid_table

    # Also compute block_info for each pid
    block_info_table = {}
    for pid, pid_results in by_pid.items():
        r = pid_results[0]  # all share same question
        chars = len(r.get("question", "")) + sum(len(c) for c in r.get("choices", []))
        block_info_table[pid] = {
            "total_chars": chars,
            "has_image": any("figure" in rec.get("config_hash", "") or
                            rec.get("canvas_w", 0) > 0
                            for rec in pid_results),
            "n_blocks": 5,  # approximate for ScienceQA
            "has_knowledge": True,
        }

    # Save
    output = {
        "reward_table": reward_table,
        "block_info": block_info_table,
        "strategy_stats": {s: {"accuracy": d["correct"] / max(d["total"], 1),
                                "total": d["total"]}
                           for s, d in strategy_stats.items()},
    }
    with open(table_path, "w") as f:
        json.dump(output, f, indent=1)

    # Print strategy comparison
    print("\n=== Strategy QA Accuracy ===")
    for s in STRATEGY_NAMES:
        st = strategy_stats.get(s, {"correct": 0, "total": 0})
        acc = st["correct"] / max(st["total"], 1) * 100
        print(f"  {s:20s}: {acc:5.1f}% ({st['correct']}/{st['total']})")

    print(f"\nReward table saved: {table_path}")
    print(f"  {len(reward_table)} samples, "
          f"{sum(len(v) for v in reward_table.values())} entries")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build QA Reward Table")
    parser.add_argument("--phase", choices=["render", "evaluate", "build", "all"],
                        default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    num_samples = args.num_samples
    batch_size = args.batch_size

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.phase in ("render", "all"):
        print("=" * 60)
        print("Phase 1: Rendering candidate canvases")
        print("=" * 60)

        # Load ScienceQA data
        with open(SCIQA_CACHE, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            train_data = cache["train"]
        elif isinstance(cache, (list, tuple)) and len(cache) == 2:
            train_data, _ = cache
        else:
            train_data = cache
        print(f"Loaded {len(train_data)} training samples")

        # Try loading HF dataset for images
        hf_dataset = None
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
            print(f"HF dataset loaded: {len(hf_dataset)} samples")
        except Exception as e:
            print(f"HF dataset not available: {e}")

        render_candidates(train_data, hf_dataset, num_samples=num_samples)

    if args.phase in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("Phase 2: VLM QA Evaluation")
        print("=" * 60)
        run_evaluation(device=args.device, batch_size=batch_size)

    if args.phase in ("build", "all"):
        print("\n" + "=" * 60)
        print("Phase 3: Building Reward Table")
        print("=" * 60)
        build_reward_table()


if __name__ == "__main__":
    main()
