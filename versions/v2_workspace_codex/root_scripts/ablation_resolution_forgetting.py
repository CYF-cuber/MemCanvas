#!/usr/bin/env python3
"""
Ablation: Resolution-based forgetting VLM accuracy evaluation.

Runs the forgetting simulation (I=2000, T=0) with resolution degradation,
then evaluates VLM accuracy with the surviving memories at their assigned resolutions.

Quality levels: PNG@100% → PNG@75% → PNG@50% → PNG@25% → Deleted

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/ablation_resolution_forgetting.py
"""

import copy
import io
import json
import pickle
import sys
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    ExperimentConfig,
    MemoryEntry,
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
REVIEW_INTERVAL = 2000
FREQ_THRESHOLD = 0
N_EPOCHS = 3


class ResQuality(IntEnum):
    RES_100 = 0
    RES_75  = 1
    RES_50  = 2
    RES_25  = 3
    DELETED = 4

RESOLUTION_SCALES = {
    ResQuality.RES_100: 1.0,
    ResQuality.RES_75:  0.75,
    ResQuality.RES_50:  0.50,
    ResQuality.RES_25:  0.25,
    ResQuality.DELETED: 0.0,
}

QUALITY_NAMES = {
    0: "PNG@100%", 1: "PNG@75%", 2: "PNG@50%", 3: "PNG@25%", 4: "Deleted"
}


def compress_to_resolution(canvas_bytes: bytes, quality: int) -> bytes:
    """Resize canvas to target resolution, keep PNG format."""
    if not canvas_bytes or quality == ResQuality.DELETED:
        return b""
    if quality == ResQuality.RES_100:
        return canvas_bytes
    img = Image.open(io.BytesIO(canvas_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = RESOLUTION_SCALES[quality]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def simulate_resolution_forgetting(test_pids, candidates, n_memories):
    """Run forgetting simulation with resolution-based degradation."""
    qualities = [int(ResQuality.RES_100)] * n_memories
    retrieval_counts = [0] * n_memories

    step = 0
    for epoch in range(N_EPOCHS):
        for pid in test_pids:
            cands = candidates.get(pid, [])
            effective = []
            for mem_idx, sim in cands:
                if qualities[mem_idx] == int(ResQuality.DELETED):
                    continue
                effective.append(mem_idx)
                if len(effective) >= TOP_K:
                    break
            for mem_idx in effective:
                retrieval_counts[mem_idx] += 1

            step += 1

            if step % REVIEW_INTERVAL == 0:
                for i in range(n_memories):
                    if qualities[i] == int(ResQuality.DELETED):
                        continue
                    if retrieval_counts[i] <= FREQ_THRESHOLD:
                        qualities[i] = min(qualities[i] + 1, int(ResQuality.DELETED))

    surviving = sum(1 for q in qualities if q != int(ResQuality.DELETED))
    from collections import Counter
    dist = Counter(qualities)
    print(f"\n  Resolution forgetting (I={REVIEW_INTERVAL}, T={FREQ_THRESHOLD}):")
    print(f"  Surviving: {surviving}/{n_memories} ({surviving/n_memories*100:.1f}%)")
    for ql in sorted(dist.keys()):
        print(f"    {QUALITY_NAMES[ql]}: {dist[ql]}")
    return qualities


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"ablation_res_forgetting_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load memory index
    print("\nLoading memory index...")
    with open(MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    n_memories = len(memories)
    print(f"  {n_memories} memories")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # Compute hybrid key embeddings
    print(f"\nLoading embeddings (alpha={ALPHA})...")
    img_emb = embeddings
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    key_embeddings = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    query_embeddings = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")

    # Pre-compute retrieval candidates
    print("\nPre-computing retrieval candidates (top-30)...")
    mem_pids = [m.pid for m in memories]
    q_n = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_embeddings / q_n
    k_n = np.linalg.norm(key_embeddings, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_embeddings / k_n
    sims = q_norm @ k_norm.T

    candidates = {}
    for i, pid in enumerate(test_pids):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:30]
        cands = []
        for idx in top_indices:
            if row[idx] < THRESHOLD:
                break
            if mem_pids[idx] == pid:
                continue
            cands.append((int(idx), float(row[idx])))
        candidates[pid] = cands
    print(f"  {len(candidates)} queries")

    # Run forgetting simulation
    print("\nRunning resolution-based forgetting simulation...")
    final_qualities = simulate_resolution_forgetting(
        test_pids, candidates, n_memories
    )

    # Retrieve with forgetting state
    print("\nRetrieving top-K with forgetting state...")
    retrieved_map = {}
    for i, pid in enumerate(test_pids):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:TOP_K + 20]
        results = []
        for idx in top_indices:
            if row[idx] < THRESHOLD:
                break
            if mem_pids[idx] == pid:
                continue
            ql = final_qualities[idx]
            if ql == int(ResQuality.DELETED):
                continue
            results.append((int(idx), ql, float(row[idx])))
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

    compress_cache = {}

    def predict(problem, entries):
        if not entries:
            retrieved = None
        else:
            retrieved = []
            for mem_idx, ql, sim in entries:
                mem = memories[mem_idx]
                cache_key = (mem_idx, ql)
                if cache_key not in compress_cache:
                    compress_cache[cache_key] = compress_to_resolution(
                        mem.canvas_image_bytes, ql
                    )
                temp_mem = copy.copy(mem)
                temp_mem.canvas_image_bytes = compress_cache[cache_key]
                retrieved.append((temp_mem, sim))

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
    print("\n=== Evaluating MemCanvas with Resolution Forgetting ===")
    done = set(results["predictions"].keys())
    remaining = [p for p in test_pids if p not in done]
    print(f"  {len(remaining)} remaining")

    if remaining:
        for pid in tqdm(remaining, desc="ResForgetting"):
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

    from collections import Counter
    dist = Counter(final_qualities)

    results["config"] = {
        "alpha": ALPHA, "top_k": TOP_K, "encoder": "clip-l14",
        "vlm": "qwen2.5-vl-7b", "prompt": "v2",
        "forgetting": f"resolution_i{REVIEW_INTERVAL}_t{FREQ_THRESHOLD}",
        "quality_distribution": {QUALITY_NAMES[k]: v for k, v in dist.items()},
    }
    results["summary"] = {
        "res_forgetting_acc": acc,
        "no_forgetting_acc": 82.65,
        "format_forgetting_acc": 82.57,
        "baseline_acc": 78.73,
        "n_surviving": sum(1 for q in final_qualities if q != int(ResQuality.DELETED)),
        "n_total": n_memories,
    }
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ABLATION: Resolution-Based Forgetting (ScienceQA)")
    print(f"{'='*60}")
    print(f"  Baseline (no memory):                   78.73%")
    print(f"  MemCanvas w/o forgetting (12,726):      82.65%")
    print(f"  MemCanvas w/ format forgetting (3,214): 82.57%")
    print(f"  MemCanvas w/ resolution forgetting:     {acc:.2f}%")
    print(f"{'='*60}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
