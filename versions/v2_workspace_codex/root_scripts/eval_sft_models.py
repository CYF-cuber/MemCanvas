#!/usr/bin/env python3
"""
Evaluate SFT (LoRA) fine-tuned Qwen2.5-VL-7B on each benchmark.

Usage:
  # Evaluate a single benchmark:
  python eval_sft_models.py --benchmark okvqa --adapter-path /path/to/lora

  # Evaluate all benchmarks:
  python eval_sft_models.py --all

  # Evaluate with merged model (already exported):
  python eval_sft_models.py --benchmark okvqa --model-path /path/to/merged
"""

import argparse
import json
import os
import re
import string
import sys
import io
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"
LLAMA_FACTORY_DIR = "/home/cyf/LLaMA-Factory-main"

# Default adapter paths (relative to LLAMA_FACTORY_DIR)
DEFAULT_ADAPTERS = {
    "scienceqa": "saves/qwen2_5vl-7b/lora/scienceqa_sft",
    "okvqa":     "saves/qwen2_5vl-7b/lora/okvqa_sft",
    "infovqa":   "saves/qwen2_5vl-7b/lora/infovqa_sft",
    "hotpotqa":  "saves/qwen2_5vl-7b/lora/hotpotqa_sft",
    "mmqa":      "saves/qwen2_5vl-7b/lora/mmqa_sft",
}

# Validation data paths
VAL_DATA = {
    "scienceqa": f"{LLAMA_FACTORY_DIR}/data/scienceqa_sft_val.json",
    "okvqa":     f"{LLAMA_FACTORY_DIR}/data/okvqa_sft_val.json",
    "infovqa":   f"{LLAMA_FACTORY_DIR}/data/infovqa_sft_val.json",
    "hotpotqa":  f"{LLAMA_FACTORY_DIR}/data/hotpotqa_sft_val.json",
    "mmqa":      f"{LLAMA_FACTORY_DIR}/data/mmqa_sft_val.json",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(remove_articles(remove_punc(str(s).lower())).split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def vqa_accuracy(prediction: str, gt_answers: List[str]) -> float:
    """Standard VQA accuracy: min(1, #matching / 3)."""
    pred = prediction.strip().lower().rstrip(".")
    matching = sum(1 for a in gt_answers if a.strip().lower() == pred)
    return min(1.0, matching / 3.0)


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity (1 - distance/max_len)."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len2 + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j-1], prev)
            prev = temp
    dist = dp[len2]
    return 1.0 - dist / max(len1, len2)


def anls_score(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    if not ground_truths:
        return 0.0
    max_nls = max(normalized_levenshtein(prediction, gt) for gt in ground_truths)
    return max_nls if max_nls >= threshold else 0.0


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model(model_path: str = None, adapter_path: str = None):
    """Load base model, optionally with LoRA adapter."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    base_path = model_path or BASE_MODEL_PATH
    print(f"Loading base model from {base_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path and os.path.isdir(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("  LoRA merged")

    model.eval()
    processor = AutoProcessor.from_pretrained(base_path)
    print("  Model ready")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, processor, user_content: str, images: List[Image.Image] = None,
                  max_new_tokens: int = 64) -> str:
    """Run VLM inference with the SFT prompt format."""
    # Build message content
    content = []
    if images:
        for img in images:
            content.append({"type": "image", "image": img})
    # Replace <image> tags from user content (already represented by image objects)
    clean_text = re.sub(r"<image>\s*", "", user_content)
    content.append({"type": "text", "text": clean_text})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    proc_images = images if images else None
    inputs = processor(text=[text], images=proc_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Benchmark Evaluation Functions
# ---------------------------------------------------------------------------
def eval_scienceqa(model, processor, data: List[Dict], max_samples: int = 0) -> Dict:
    """ScienceQA: multiple-choice accuracy."""
    correct = 0
    total = 0
    samples = data[:max_samples] if max_samples > 0 else data

    for i, sample in enumerate(samples):
        user_msg = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"].strip().upper()

        # Load images if present
        images = None
        if "images" in sample and sample["images"]:
            images = [Image.open(p).convert("RGB") for p in sample["images"] if os.path.exists(p)]
            if not images:
                images = None

        pred = run_inference(model, processor, user_msg, images, max_new_tokens=8)
        # Extract answer letter
        pred_letter = ""
        for ch in pred.upper():
            if ch in "ABCDEF":
                pred_letter = ch
                break

        if pred_letter == gold:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}] acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_okvqa(model, processor, data: List[Dict], max_samples: int = 0) -> Dict:
    """OK-VQA: VQA accuracy."""
    scores = []
    samples = data[:max_samples] if max_samples > 0 else data

    for i, sample in enumerate(samples):
        user_msg = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"].strip()

        images = None
        if "images" in sample and sample["images"]:
            images = [Image.open(p).convert("RGB") for p in sample["images"] if os.path.exists(p)]
            if not images:
                images = None

        pred = run_inference(model, processor, user_msg, images, max_new_tokens=32)

        # VQA accuracy uses gt_answers list; for SFT we have single answer
        # Treat gold as if 10 annotators agreed (score = min(1, 10/3) = 1.0 if exact match)
        score = 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0
        scores.append(score)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}] vqa_acc={np.mean(scores):.4f}")

    return {"vqa_accuracy": float(np.mean(scores)), "total": len(scores)}


def eval_infovqa(model, processor, data: List[Dict], max_samples: int = 0) -> Dict:
    """InfographicVQA: ANLS score."""
    scores = []
    samples = data[:max_samples] if max_samples > 0 else data

    for i, sample in enumerate(samples):
        user_msg = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"].strip()

        images = None
        if "images" in sample and sample["images"]:
            images = [Image.open(p).convert("RGB") for p in sample["images"] if os.path.exists(p)]
            if not images:
                images = None

        pred = run_inference(model, processor, user_msg, images, max_new_tokens=64)
        score = anls_score(pred, [gold])
        scores.append(score)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}] ANLS={np.mean(scores):.4f}")

    return {"anls": float(np.mean(scores)), "total": len(scores)}


def eval_hotpotqa(model, processor, data: List[Dict], max_samples: int = 0) -> Dict:
    """HotpotQA: EM and F1."""
    em_scores, f1_scores = [], []
    samples = data[:max_samples] if max_samples > 0 else data

    for i, sample in enumerate(samples):
        user_msg = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"].strip()

        pred = run_inference(model, processor, user_msg, max_new_tokens=64)

        em_scores.append(exact_match(pred, gold))
        f1_scores.append(f1_score(pred, gold))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(samples)}] EM={np.mean(em_scores):.4f} F1={np.mean(f1_scores):.4f}")

    return {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "total": len(em_scores),
    }


def eval_mmqa(model, processor, data: List[Dict], max_samples: int = 0) -> Dict:
    """MMQA: EM and F1."""
    em_scores, f1_scores = [], []
    samples = data[:max_samples] if max_samples > 0 else data

    for i, sample in enumerate(samples):
        user_msg = sample["messages"][0]["content"]
        gold = sample["messages"][1]["content"].strip()

        images = None
        if "images" in sample and sample["images"]:
            valid_imgs = [p for p in sample["images"] if os.path.exists(p)]
            if valid_imgs:
                images = [Image.open(p).convert("RGB") for p in valid_imgs]

        pred = run_inference(model, processor, user_msg, images, max_new_tokens=64)

        em_scores.append(exact_match(pred, gold))
        f1_scores.append(f1_score(pred, gold))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(samples)}] EM={np.mean(em_scores):.4f} F1={np.mean(f1_scores):.4f}")

    return {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "total": len(em_scores),
    }


EVAL_FUNCS = {
    "scienceqa": eval_scienceqa,
    "okvqa":     eval_okvqa,
    "infovqa":   eval_infovqa,
    "hotpotqa":  eval_hotpotqa,
    "mmqa":      eval_mmqa,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT LoRA models")
    parser.add_argument("--benchmark", choices=list(EVAL_FUNCS.keys()),
                        help="Which benchmark to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all benchmarks")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (overrides default)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to merged model (skip LoRA loading)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.error("Specify --benchmark or --all")

    benchmarks = list(EVAL_FUNCS.keys()) if args.all else [args.benchmark]

    results = {}
    for bench in benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {bench}")
        print(f"{'='*60}")

        # Determine adapter path
        adapter_path = args.adapter_path
        if not adapter_path and not args.model_path:
            adapter_path = os.path.join(LLAMA_FACTORY_DIR, DEFAULT_ADAPTERS[bench])

        # Check if adapter exists
        if adapter_path and not os.path.isdir(adapter_path):
            print(f"  WARNING: Adapter not found at {adapter_path}, skipping {bench}")
            continue

        # Load val data
        val_file = VAL_DATA[bench]
        if not os.path.exists(val_file):
            print(f"  WARNING: Val data not found at {val_file}, skipping {bench}")
            continue

        print(f"Loading val data from {val_file}...")
        with open(val_file) as f:
            data = json.load(f)
        print(f"  {len(data)} samples")

        # Load model
        model, processor = load_model(args.model_path, adapter_path)

        # Run evaluation
        start = time.time()
        eval_fn = EVAL_FUNCS[bench]
        result = eval_fn(model, processor, data, args.max_samples)
        elapsed = time.time() - start

        result["elapsed_seconds"] = elapsed
        result["adapter_path"] = adapter_path or "none"
        results[bench] = result

        # Report
        print(f"\n--- {bench} Results ---")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Free GPU memory before next benchmark
        del model, processor
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for bench, res in results.items():
        metrics = {k: f"{v:.4f}" for k, v in res.items() if isinstance(v, float) and k != "elapsed_seconds"}
        print(f"  {bench}: {metrics}")

    # Save results
    out_file = args.output or "/home/cyf/codex/sft_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
