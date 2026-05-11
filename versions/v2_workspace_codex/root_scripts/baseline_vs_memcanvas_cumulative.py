#!/usr/bin/env python3
"""
Baseline vs MemCanvas comparison with cumulative-count forgetting (I=2000, T=0).

Re-runs forgetting simulation with cumulative counts (no count reset),
then evaluates VLM accuracy: baseline (no memory) vs MemCanvas.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/baseline_vs_memcanvas_cumulative.py
"""

import copy
import io
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    ExperimentConfig,
    MemoryEntry,
    MemoryIndex,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
    CLIPLargeMemoryBuilder,
)

sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import build_prompt_v2, extract_answer_last

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
OUTPUT_ROOT = Path("/home/cyf/codex")
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

# Experiment config
ALPHA = 0.50
TOP_K = 2
REVIEW_INTERVAL = 2000
FREQ_THRESHOLD = 0
N_EPOCHS = 3
THRESHOLD = 0.1


class QualityLevel:
    PNG = 0
    WEBP_85 = 1
    AVIF_50 = 2
    AVIF_50_HALF = 3
    DELETED = 4


QUALITY_NAMES = {0: "PNG", 1: "WebP-85", 2: "AVIF-50", 3: "AVIF-50@0.5x", 4: "Deleted"}


# ---------------------------------------------------------------------------
# Image compression
# ---------------------------------------------------------------------------
def compress_to_quality(canvas_bytes: bytes, quality: int) -> bytes:
    if not canvas_bytes or quality == QualityLevel.DELETED:
        return b""
    if quality == QualityLevel.PNG:
        return canvas_bytes
    img = Image.open(io.BytesIO(canvas_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if quality == QualityLevel.AVIF_50_HALF:
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)
    buf = io.BytesIO()
    if quality == QualityLevel.WEBP_85:
        img.save(buf, format="WEBP", quality=85, method=6)
    elif quality in (QualityLevel.AVIF_50, QualityLevel.AVIF_50_HALF):
        img.save(buf, format="AVIF", quality=50)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Forgetting simulation (cumulative counts, NO reset)
# ---------------------------------------------------------------------------
def simulate_cumulative_forgetting(
    test_pids, candidates, n_memories, review_interval, freq_threshold, n_epochs, top_k=2
):
    """Run forgetting simulation with cumulative (never-reset) counts."""
    qualities = [QualityLevel.PNG] * n_memories
    retrieval_counts = [0] * n_memories

    step = 0
    for epoch in range(n_epochs):
        for pid in test_pids:
            cands = candidates.get(pid, [])
            effective = []
            for mem_idx, sim in cands:
                if qualities[mem_idx] == QualityLevel.DELETED:
                    continue
                effective.append(mem_idx)
                if len(effective) >= top_k:
                    break
            for mem_idx in effective:
                retrieval_counts[mem_idx] += 1

            step += 1

            if step % review_interval == 0:
                for i in range(n_memories):
                    if qualities[i] == QualityLevel.DELETED:
                        continue
                    if retrieval_counts[i] <= freq_threshold:
                        qualities[i] = min(qualities[i] + 1, QualityLevel.DELETED)
                # NO reset — counts accumulate

    surviving = sum(1 for q in qualities if q != QualityLevel.DELETED)
    dist = Counter(qualities)
    print(f"\n  Cumulative forgetting (I={review_interval}, T={freq_threshold}):")
    print(f"  Surviving: {surviving}/{n_memories} ({surviving/n_memories*100:.1f}%)")
    for ql in sorted(dist.keys()):
        print(f"    {QUALITY_NAMES[ql]}: {dist[ql]}")
    return qualities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"baseline_vs_memcanvas_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load memory index
    print("\nLoading memory index...")
    with open(MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    n_memories = len(memories)
    print(f"  {n_memories} memories, dim={embedding_dim}")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # Load train data for text embeddings
    train_data = loader.get_split("train")

    # ---------------------------------------------------------------------------
    # Step 1: Compute hybrid key embeddings
    # ---------------------------------------------------------------------------
    print(f"\nLoading/computing embeddings (alpha={ALPHA})...")
    # Image embeddings = original memory index embeddings (CLIP image features)
    img_emb = embeddings  # (12726, 768)

    # Text embeddings from cache
    txt_cache = CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy"
    if txt_cache.exists():
        print("  Loading cached text embeddings...")
        txt_emb = np.load(txt_cache)
    else:
        raise FileNotFoundError(f"Cached text embeddings not found: {txt_cache}")

    # Hybrid key
    key_embeddings = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    print(f"  Hybrid key: {ALPHA}*img + {1-ALPHA}*txt, shape={key_embeddings.shape}")

    # Query embeddings (text-only from CLIP)
    query_cache = CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy"
    if query_cache.exists():
        print("  Loading cached query embeddings...")
        query_embeddings = np.load(query_cache)
    else:
        raise FileNotFoundError(f"Cached query embeddings not found: {query_cache}")

    # ---------------------------------------------------------------------------
    # Step 2: Pre-compute retrieval candidates (top-30)
    # ---------------------------------------------------------------------------
    print("\nPre-computing retrieval candidates (top-30)...")
    mem_pids = [m.pid for m in memories]

    # Normalize
    q_n = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_embeddings / q_n
    k_n = np.linalg.norm(key_embeddings, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_embeddings / k_n

    sims = q_norm @ k_norm.T  # (4241, 12726)

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
    print(f"  Done: {len(candidates)} test queries")

    # ---------------------------------------------------------------------------
    # Step 3: Run cumulative forgetting simulation
    # ---------------------------------------------------------------------------
    print("\nRunning cumulative forgetting simulation...")
    final_qualities = simulate_cumulative_forgetting(
        test_pids, candidates, n_memories,
        REVIEW_INTERVAL, FREQ_THRESHOLD, N_EPOCHS, TOP_K
    )

    # ---------------------------------------------------------------------------
    # Step 4: Retrieve with forgetting state
    # ---------------------------------------------------------------------------
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
            if ql == QualityLevel.DELETED:
                continue
            results.append((int(idx), ql, float(row[idx])))
            if len(results) >= TOP_K:
                break
        retrieved_map[pid] = results

    has_mem = sum(1 for v in retrieved_map.values() if len(v) > 0)
    print(f"  {has_mem}/{len(test_pids)} test samples have memories")

    # ---------------------------------------------------------------------------
    # Step 5: Load VLM and evaluate
    # ---------------------------------------------------------------------------
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
                    compress_cache[cache_key] = compress_to_quality(mem.canvas_image_bytes, ql)
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

        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if memory_images:
            inputs = vlm.processor(text=[text], images=memory_images, return_tensors="pt", padding=True)
        else:
            inputs = vlm.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(vlm.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = vlm.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = vlm.processor.decode(gen_ids, skip_special_tokens=True).strip()
        answer = extract_answer_last(raw)
        return answer, raw

    # Checkpoint support
    ckpt_path = out_dir / "checkpoint.json"
    # Baseline already known: 81.70% (v2 prompt, no memory, from prompt_eval)
    bl_acc = 81.70
    results = {"baseline": {"accuracy": bl_acc, "note": "from prompt_eval_20260223"},
               "memcanvas": {"correct": 0, "total": 0, "predictions": {}}}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"  Resumed: memcanvas={len(results['memcanvas']['predictions'])}")

    print(f"\n  Baseline (known): {bl_acc:.2f}%")

    # --- Evaluate MemCanvas ---
    print("\n=== Evaluating MEMCANVAS (cumulative forgetting) ===")
    mc_done = set(results["memcanvas"]["predictions"].keys())
    remaining = [p for p in test_pids if p not in mc_done]
    if remaining:
        for pid in tqdm(remaining, desc="MemCanvas"):
            problem = test_data[pid]
            answer_idx = problem.get("answer", 0)
            gt = CHOICE_LABELS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
            entries = retrieved_map.get(pid, [])
            pred, raw = predict(problem, entries)
            correct = (pred == gt)
            results["memcanvas"]["predictions"][pid] = {
                "pred": pred, "gt": gt, "correct": correct
            }
            results["memcanvas"]["correct"] += int(correct)
            results["memcanvas"]["total"] += 1
            if results["memcanvas"]["total"] % 200 == 0:
                with open(ckpt_path, "w") as f:
                    json.dump(results, f)
    else:
        print("  Already complete")

    # Recalculate
    mc_correct = sum(1 for v in results["memcanvas"]["predictions"].values() if v["correct"])
    mc_total = len(results["memcanvas"]["predictions"])
    results["memcanvas"]["correct"] = mc_correct
    results["memcanvas"]["total"] = mc_total
    mc_acc = mc_correct / mc_total * 100 if mc_total > 0 else 0
    print(f"  MemCanvas: {mc_correct}/{mc_total} = {mc_acc:.2f}%")

    # Save final
    results["config"] = {
        "alpha": ALPHA, "top_k": TOP_K, "encoder": "clip-l14", "vlm": "qwen2.5-vl-7b",
        "forgetting": f"cumulative_i{REVIEW_INTERVAL}_t{FREQ_THRESHOLD}",
        "prompt": "v2 (Guided+CoT)", "max_tokens": 512,
        "n_epochs": N_EPOCHS,
    }
    results["summary"] = {
        "baseline_acc": bl_acc, "memcanvas_acc": mc_acc,
        "improvement": mc_acc - bl_acc,
    }
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Baseline (no memory): {bl_acc:.2f}%")
    print(f"  MemCanvas (cumul.):   {mc_acc:.2f}%")
    print(f"  Improvement:          +{mc_acc - bl_acc:.2f}pp")
    print(f"{'='*50}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
