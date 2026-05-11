#!/usr/bin/env python3
"""
Ablation: MemCanvas WITHOUT forgetting (all 12,726 memories at PNG quality).

Compare with the full system (with forgetting, 3,214 surviving = 82.57%).
This tests whether forgetting hurts or helps (noise removal hypothesis).

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/ablation_no_forgetting.py
"""

import copy
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    ExperimentConfig,
    MemoryEntry,
    MemoryIndex,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
)

sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import build_prompt_v2, extract_answer_last

# ---------------------------------------------------------------------------
MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
OUTPUT_ROOT = Path("/home/cyf/codex")
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

ALPHA = 0.50
TOP_K = 2
THRESHOLD = 0.1


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"ablation_no_forgetting_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load memory index
    print("\nLoading memory index...")
    with open(MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    n_memories = len(memories)
    print(f"  {n_memories} memories (ALL kept, no forgetting)")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # Compute hybrid key embeddings
    print(f"\nLoading embeddings (alpha={ALPHA})...")
    img_emb = embeddings
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    key_embeddings = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    query_embeddings = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")
    print(f"  key: {key_embeddings.shape}, query: {query_embeddings.shape}")

    # Compute similarities
    print("\nComputing retrieval (no forgetting, all memories available)...")
    mem_pids = [m.pid for m in memories]

    q_n = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_embeddings / q_n
    k_n = np.linalg.norm(key_embeddings, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_embeddings / k_n

    sims = q_norm @ k_norm.T  # (4241, 12726)

    # Retrieve top-K from ALL memories (no forgetting filter)
    retrieved_map = {}
    for i, pid in enumerate(test_pids):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:TOP_K + 10]
        results = []
        for idx in top_indices:
            if row[idx] < THRESHOLD:
                break
            if mem_pids[idx] == pid:
                continue
            # All memories at PNG quality (quality=0)
            results.append((int(idx), 0, float(row[idx])))
            if len(results) >= TOP_K:
                break
        retrieved_map[pid] = results

    has_mem = sum(1 for v in retrieved_map.values() if len(v) > 0)
    print(f"  {has_mem}/{len(test_pids)} test samples have memories")

    # Load VLM
    print("\nLoading VLM (Qwen2.5-VL-7B)...")
    config = ExperimentConfig()
    vlm = Qwen25VLEvaluator(config)
    print("  VLM loaded")

    def predict(problem, entries):
        if not entries:
            retrieved = None
        else:
            retrieved = []
            for mem_idx, ql, sim in entries:
                mem = memories[mem_idx]
                retrieved.append((mem, sim))

        system_prompt, user_prompt, memory_images = build_prompt_v2(problem, retrieved)
        content = []
        for img in memory_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if memory_images:
            inputs = vlm.processor(
                text=[text], images=memory_images, return_tensors="pt", padding=True
            )
        else:
            inputs = vlm.processor(
                text=[text], return_tensors="pt", padding=True
            )
        inputs = {k: v.to(vlm.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = vlm.model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = vlm.processor.decode(gen_ids, skip_special_tokens=True).strip()
        answer = extract_answer_last(raw)
        return answer, raw

    # Checkpoint
    ckpt_path = out_dir / "checkpoint.json"
    results = {"predictions": {}}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"  Resumed: {len(results['predictions'])} done")

    # Evaluate
    print("\n=== Evaluating MemCanvas WITHOUT forgetting (all 12,726 memories) ===")
    done = set(results["predictions"].keys())
    remaining = [p for p in test_pids if p not in done]
    print(f"  {len(remaining)} remaining")

    if remaining:
        for pid in tqdm(remaining, desc="NoForgetting"):
            problem = test_data[pid]
            answer_idx = problem.get("answer", 0)
            gt = CHOICE_LABELS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
            entries = retrieved_map.get(pid, [])
            pred, raw = predict(problem, entries)
            correct = (pred == gt)
            results["predictions"][pid] = {
                "pred": pred, "gt": gt, "correct": correct
            }
            if len(results["predictions"]) % 200 == 0:
                with open(ckpt_path, "w") as f:
                    json.dump(results, f)

    # Final stats
    correct = sum(1 for v in results["predictions"].values() if v["correct"])
    total = len(results["predictions"])
    acc = correct / total * 100 if total > 0 else 0

    results["config"] = {
        "alpha": ALPHA, "top_k": TOP_K, "encoder": "clip-l14",
        "vlm": "qwen2.5-vl-7b", "forgetting": "NONE (all 12726 memories)",
        "prompt": "v2 (Guided+CoT)", "max_tokens": 512,
    }
    results["summary"] = {
        "no_forgetting_acc": acc,
        "with_forgetting_acc": 82.57,
        "baseline_acc": 81.70,
        "n_memories_no_forgetting": n_memories,
        "n_memories_with_forgetting": 3214,
    }
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ABLATION: Forgetting Effect on ScienceQA")
    print(f"{'='*60}")
    print(f"  Baseline (no memory):           81.70%")
    print(f"  MemCanvas w/ forgetting (3,214): 82.57%")
    print(f"  MemCanvas w/o forgetting (12,726): {acc:.2f}%")
    print(f"{'='*60}")
    if acc > 82.57:
        print(f"  -> No forgetting is BETTER by +{acc - 82.57:.2f}pp")
        print(f"     (more memories = more useful knowledge)")
    elif acc < 82.57:
        print(f"  -> Forgetting HELPS by +{82.57 - acc:.2f}pp")
        print(f"     (removing noisy memories improves quality)")
    else:
        print(f"  -> No difference (forgetting is neutral)")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
