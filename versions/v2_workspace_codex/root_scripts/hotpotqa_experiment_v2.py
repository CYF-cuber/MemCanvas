#!/usr/bin/env python3
"""
HotpotQA Experiment V2 — Current MemCanvas method.

Changes from V1:
  - alpha=0.75 (best from ablation, was 0.0)
  - Saves canvas examples for paper
  - Cleaner prompt (v2 guided + CoT)

Usage:
  # Phase 1: Build retrieval map (CPU only, no GPU needed)
  python -u /home/cyf/codex/hotpotqa_experiment_v2.py --phase prep

  # Phase 2: VLM evaluation (needs GPU)
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/hotpotqa_experiment_v2.py --phase eval

  # Or run both:
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/hotpotqa_experiment_v2.py --phase all
"""

import argparse
import io
import json
import os
import pickle
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — CURRENT METHOD CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/cyf/codex/hotpotqa_data")       # Reuse existing canvases & embeddings
OUTPUT_DIR = Path("/home/cyf/codex/hotpotqa_experiment_v2")
EXAMPLE_DIR = OUTPUT_DIR / "canvas_examples"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

# Current method parameters
ALPHA = 0.75          # Best from ablation (was 0.0 in v1)
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def compute_exact(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Phase 1: Build retrieval map (CPU only)
# ---------------------------------------------------------------------------
def phase_prep():
    """Build retrieval map using existing canvases & embeddings with alpha=0.75."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing embeddings (already computed in v1)
    print("Loading cached embeddings...")
    canvas_img_emb = np.load(DATA_DIR / "canvas_embeddings.npy")
    canvas_txt_emb = np.load(DATA_DIR / "canvas_text_embeddings.npy")
    query_emb = np.load(DATA_DIR / "query_embeddings.npy")
    print(f"  Canvas img: {canvas_img_emb.shape}, txt: {canvas_txt_emb.shape}")
    print(f"  Query: {query_emb.shape}")

    # Load metadata
    with open(DATA_DIR / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train_data = meta["train"]
    dev_data = meta["dev"]
    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Build hybrid key: alpha * img + (1-alpha) * txt
    print(f"\nBuilding retrieval map (alpha={ALPHA}, K={TOP_K})...")
    key_emb = ALPHA * canvas_img_emb + (1 - ALPHA) * canvas_txt_emb
    key_norm = key_emb / np.linalg.norm(key_emb, axis=1, keepdims=True).clip(min=1e-8)
    q_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(min=1e-8)

    # Batch similarity computation
    sims = q_norm @ key_norm.T  # (7405, 50000)

    retrieval_map = {}
    sim_stats = []
    for i in range(len(query_emb)):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:TOP_K + 5]
        results = []
        for idx in top_indices:
            if row[idx] < SIMILARITY_THRESHOLD:
                break
            results.append((int(idx), float(row[idx])))
            if len(results) >= TOP_K:
                break
        retrieval_map[i] = results
        if results:
            sim_stats.append(results[0][1])

    has_mem = sum(1 for v in retrieval_map.values() if len(v) > 0)
    avg_top1_sim = np.mean(sim_stats) if sim_stats else 0
    print(f"  {has_mem}/{len(query_emb)} dev samples have memories")
    print(f"  Avg top-1 similarity: {avg_top1_sim:.4f}")

    # Save retrieval map
    with open(OUTPUT_DIR / "retrieval_map.pkl", "wb") as f:
        pickle.dump(retrieval_map, f)
    print(f"  Saved retrieval map to {OUTPUT_DIR / 'retrieval_map.pkl'}")

    # Save canvas examples for paper
    print("\nSaving canvas examples...")
    example_indices = set()
    for i in [0, 100, 500, 1000, 2000]:
        if i < len(dev_data) and retrieval_map.get(i):
            for canvas_idx, sim in retrieval_map[i][:2]:
                example_indices.add(canvas_idx)
                # Copy canvas
                src = DATA_DIR / "canvases" / f"{canvas_idx:05d}.png"
                dst = EXAMPLE_DIR / f"query{i}_canvas{canvas_idx}_sim{sim:.3f}.png"
                if src.exists():
                    import shutil
                    shutil.copy(src, dst)
                    print(f"  Saved: {dst.name}")

            # Also save query info
            sample = dev_data[i]
            info = {
                "query_idx": i,
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample.get("type", ""),
                "retrieved": [(idx, sim) for idx, sim in retrieval_map[i]],
                "retrieved_questions": [train_data[idx]["question"] for idx, _ in retrieval_map[i]],
                "retrieved_answers": [train_data[idx]["answer"] for idx, _ in retrieval_map[i]],
            }
            with open(EXAMPLE_DIR / f"query{i}_info.json", "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n=== Phase 1 (Prep) Complete ===")
    print(f"  Retrieval map: alpha={ALPHA}, K={TOP_K}")
    print(f"  Canvas examples saved to {EXAMPLE_DIR}")
    return retrieval_map, dev_data, train_data


# ---------------------------------------------------------------------------
# Phase 2: VLM Evaluation (needs GPU)
# ---------------------------------------------------------------------------
def load_canvas(idx: int) -> bytes:
    path = DATA_DIR / "canvases_smart" / f"{idx:05d}.png"
    with open(path, "rb") as f:
        return f.read()


def format_context(sample: dict) -> str:
    parts = []
    for para in sample.get("paragraphs", []):
        title = para["title"]
        text = para["text"][:500]
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def predict_baseline(model, processor, sample):
    """Baseline: context + question → VLM."""
    context = format_context(sample)
    user_text = (
        "Use the following context passages to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {sample['question']}\n"
        "Answer concisely:"
    )

    content = [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def predict_memcanvas(model, processor, sample, retrieved):
    """MemCanvas: canvases + context + question → VLM (v2 guided + CoT)."""
    context = format_context(sample)
    memory_images = []

    prompt_parts = []
    if retrieved:
        prompt_parts.append(
            "Below are memory canvases from previously solved similar questions. "
            "Each canvas shows: relevant context passages, the question, and the "
            "correct answer (marked with ✓). Study these canvases carefully — "
            "they contain knowledge and reasoning patterns that may help."
        )
        prompt_parts.append("")
        for i, (canvas_idx, sim) in enumerate(retrieved):
            canvas_img = Image.open(io.BytesIO(load_canvas(canvas_idx))).convert("RGB")
            memory_images.append(canvas_img)
            prompt_parts.append(f"[Memory Canvas {i+1}]")
        prompt_parts.append("")
        prompt_parts.append("---")
        prompt_parts.append("")

    prompt_parts.append("Now answer the following new question using the context below.")
    prompt_parts.append("")
    prompt_parts.append(context)
    prompt_parts.append("")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    for img in memory_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    import torch
    if memory_images:
        inputs = processor(text=[text], images=memory_images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_condition(condition_name, model, processor, dev_data, retrieval_map):
    """Evaluate a condition with checkpointing."""
    ckpt_file = OUTPUT_DIR / f"checkpoint_{condition_name}.json"

    results = {}
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            results = json.load(f)
        print(f"  Resumed {condition_name}: {len(results)} done")

    done = set(results.keys())
    remaining = [i for i in range(len(dev_data)) if str(i) not in done]

    if not remaining:
        print(f"  {condition_name} already complete ({len(results)} samples)")
    else:
        print(f"  Running {condition_name}: {len(remaining)} remaining")
        for idx in tqdm(remaining, desc=condition_name):
            sample = dev_data[idx]
            try:
                if condition_name == "baseline":
                    raw = predict_baseline(model, processor, sample)
                else:
                    retrieved = retrieval_map.get(idx, [])
                    raw = predict_memcanvas(model, processor, sample, retrieved)
            except Exception as e:
                raw = ""
                print(f"\n  Error on {idx}: {e}")

            gt = sample["answer"]
            results[str(idx)] = {
                "raw": raw,
                "gt": gt,
                "em": compute_exact(raw, gt),
                "f1": compute_f1(raw, gt),
            }

            if len(results) % 200 == 0:
                with open(ckpt_file, "w") as f:
                    json.dump(results, f)

        with open(ckpt_file, "w") as f:
            json.dump(results, f)

    all_em = [v["em"] for v in results.values()]
    all_f1 = [v["f1"] for v in results.values()]
    em_avg = np.mean(all_em) * 100
    f1_avg = np.mean(all_f1) * 100
    print(f"  {condition_name}: EM={em_avg:.2f}%, F1={f1_avg:.2f}% ({len(all_em)} samples)")
    return em_avg, f1_avg, len(all_em)


def phase_eval(skip_baseline=False):
    """Run VLM evaluation."""
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(DATA_DIR / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dev_data = meta["dev"]

    # Load retrieval map
    with open(OUTPUT_DIR / "retrieval_map_smart.pkl", "rb") as f:
        retrieval_map = pickle.load(f)
    print(f"Loaded retrieval map: {len(retrieval_map)} queries")

    # Load VLM
    print("Loading Qwen2.5-VL-7B...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    print("  VLM loaded")

    # Baseline
    if not skip_baseline:
        print("\n=== Evaluating BASELINE ===")
        bl_em, bl_f1, bl_n = evaluate_condition(
            "baseline", model, processor, dev_data, retrieval_map
        )
    else:
        bl_ckpt = OUTPUT_DIR / "checkpoint_baseline.json"
        if bl_ckpt.exists():
            with open(bl_ckpt) as f:
                bl_res = json.load(f)
            bl_em = np.mean([v["em"] for v in bl_res.values()]) * 100
            bl_f1 = np.mean([v["f1"] for v in bl_res.values()]) * 100
            bl_n = len(bl_res)
            print(f"  Baseline (loaded): EM={bl_em:.2f}%, F1={bl_f1:.2f}%")
        else:
            # Try loading from v1
            v1_ckpt = Path("/home/cyf/codex/hotpotqa_experiment/checkpoint_baseline.json")
            if v1_ckpt.exists():
                import shutil
                shutil.copy(v1_ckpt, bl_ckpt)
                with open(bl_ckpt) as f:
                    bl_res = json.load(f)
                bl_em = np.mean([v["em"] for v in bl_res.values()]) * 100
                bl_f1 = np.mean([v["f1"] for v in bl_res.values()]) * 100
                bl_n = len(bl_res)
                print(f"  Baseline (from v1): EM={bl_em:.2f}%, F1={bl_f1:.2f}%")
            else:
                bl_em, bl_f1, bl_n = 0, 0, 0

    # MemCanvas
    print("\n=== Evaluating MEMCANVAS (alpha=0.75) ===")
    mc_em, mc_f1, mc_n = evaluate_condition(
        "memcanvas", model, processor, dev_data, retrieval_map
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"HotpotQA V2 Results (dev set, {mc_n} samples)")
    print(f"Config: alpha={ALPHA}, K={TOP_K}, encoder=CLIP-L/14")
    print(f"Prompt: v2 (guided + CoT)")
    print(f"{'='*60}")
    print(f"  Baseline:  EM={bl_em:.2f}%  F1={bl_f1:.2f}%")
    print(f"  MemCanvas: EM={mc_em:.2f}%  F1={mc_f1:.2f}%")
    print(f"  Delta:     EM=+{mc_em-bl_em:.2f}pp  F1=+{mc_f1-bl_f1:.2f}pp")
    print(f"{'='*60}")

    summary = {
        "dataset": "HotpotQA",
        "split": "dev",
        "n_samples": mc_n,
        "baseline": {"em": bl_em, "f1": bl_f1},
        "memcanvas": {"em": mc_em, "f1": mc_f1},
        "delta": {"em": mc_em - bl_em, "f1": mc_f1 - bl_f1},
        "config": {
            "alpha": ALPHA,
            "top_k": TOP_K,
            "encoder": "CLIP-L/14",
            "vlm": "Qwen2.5-VL-7B",
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt": "v2_guided_cot",
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "eval", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline eval (reuse v1 results)")
    args = parser.parse_args()

    if args.phase in ("prep", "all"):
        phase_prep()
        if args.phase == "prep":
            return

    if args.phase in ("eval", "all"):
        phase_eval(skip_baseline=args.skip_baseline)


if __name__ == "__main__":
    main()
