#!/usr/bin/env python3
"""
Resolution-based forgetting simulation + storage convergence plot.

Quality levels (all PNG, different resolutions):
  Level 0: 100% resolution (640px)  — SIZE_RATIO = 1.0
  Level 1: 75%  resolution (480px)  — SIZE_RATIO ≈ 0.5625
  Level 2: 50%  resolution (320px)  — SIZE_RATIO ≈ 0.25
  Level 3: 25%  resolution (160px)  — SIZE_RATIO ≈ 0.0625
  Level 4: Deleted                  — SIZE_RATIO = 0.0

Usage:
  python -u /home/cyf/codex/resolution_forgetting_simulation.py
"""

import io
import json
import pickle
import sys
import time
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    MemoryEntry, MemoryIndex, CLIPLargeMemoryBuilder, ScienceQADataLoader,
)
import __main__
__main__.MemoryEntry = MemoryEntry

# ---------------------------------------------------------------------------
# Quality levels: resolution-based
# ---------------------------------------------------------------------------
class ResQuality(IntEnum):
    RES_100 = 0   # 100% (original, 640px)
    RES_75  = 1   # 75%  (480px)
    RES_50  = 2   # 50%  (320px)
    RES_25  = 3   # 25%  (160px)
    DELETED = 4

# Empirical size ratios for PNG at different resolutions
# PNG is lossless so size scales roughly with pixel count, but not exactly
# We measure actual ratios from sample canvases
RESOLUTION_SCALES = {
    ResQuality.RES_100: 1.0,
    ResQuality.RES_75:  0.75,
    ResQuality.RES_50:  0.50,
    ResQuality.RES_25:  0.25,
    ResQuality.DELETED: 0.0,
}

QUALITY_LABELS = {
    ResQuality.RES_100: "PNG@100%",
    ResQuality.RES_75:  "PNG@75%",
    ResQuality.RES_50:  "PNG@50%",
    ResQuality.RES_25:  "PNG@25%",
    ResQuality.DELETED: "Deleted",
}

# ---------------------------------------------------------------------------
MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
REVIEW_INTERVALS = [500, 1000, 2000]
FREQ_THRESHOLDS = [0, 1, 2]
N_EPOCHS = 3
TOP_K = 2
THRESHOLD = 0.1
ALPHA = 0.50
OUTPUT_DIR = Path("/home/cyf/codex/freq_forgetting_resolution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_actual_size_ratios(memories, n_samples=50):
    """Measure actual PNG size ratios at different resolutions from sample canvases."""
    print("\nMeasuring actual size ratios from sample canvases...")
    ratios = {q: [] for q in ResQuality if q != ResQuality.DELETED}

    indices = np.linspace(0, len(memories) - 1, n_samples, dtype=int)
    for idx in indices:
        mem = memories[idx]
        if not mem.canvas_image_bytes:
            continue
        orig_size = len(mem.canvas_image_bytes)
        img = Image.open(io.BytesIO(mem.canvas_image_bytes))
        w, h = img.size
        if img.mode != "RGB":
            img = img.convert("RGB")

        for q in ResQuality:
            if q == ResQuality.DELETED:
                continue
            if q == ResQuality.RES_100:
                ratios[q].append(1.0)
            else:
                scale = RESOLUTION_SCALES[q]
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                resized.save(buf, format="PNG")
                new_size = len(buf.getvalue())
                ratios[q].append(new_size / orig_size)

    actual = {}
    for q in ResQuality:
        if q == ResQuality.DELETED:
            actual[q] = 0.0
        else:
            actual[q] = float(np.mean(ratios[q]))

    print("  Measured size ratios:")
    for q in ResQuality:
        print(f"    {QUALITY_LABELS[q]:12s}: {actual[q]:.4f}")
    return actual


def compute_storage(png_sizes, qualities, size_ratios):
    total = 0
    for i, q in enumerate(qualities):
        total += int(png_sizes[i] * size_ratios[q])
    return total


def compute_quality_dist(qualities):
    dist = {}
    for ql in ResQuality:
        dist[QUALITY_LABELS[ql]] = sum(1 for q in qualities if q == int(ql))
    return dist


def simulate(test_pids, candidates, png_sizes, size_ratios,
             review_interval, freq_threshold, n_epochs, top_k=2):
    n_memories = len(png_sizes)
    qualities = [int(ResQuality.RES_100)] * n_memories
    retrieval_counts = [0] * n_memories

    storage_curve = []
    review_snapshots = []
    init_storage = compute_storage(png_sizes, qualities, size_ratios)
    storage_curve.append((0, init_storage))

    step = 0
    for epoch in range(n_epochs):
        for pid in test_pids:
            cands = candidates.get(pid, [])
            effective = []
            for mem_idx, sim in cands:
                if qualities[mem_idx] == int(ResQuality.DELETED):
                    continue
                effective.append(mem_idx)
                if len(effective) >= top_k:
                    break
            for mem_idx in effective:
                retrieval_counts[mem_idx] += 1

            step += 1

            if step % review_interval == 0:
                for i in range(n_memories):
                    if qualities[i] == int(ResQuality.DELETED):
                        continue
                    if retrieval_counts[i] <= freq_threshold:
                        qualities[i] = min(int(qualities[i]) + 1, int(ResQuality.DELETED))
                # NO reset: counts accumulate

                storage = compute_storage(png_sizes, qualities, size_ratios)
                n_alive = sum(1 for q in qualities if q != int(ResQuality.DELETED))
                review_snapshots.append({
                    "step": step, "epoch": epoch,
                    "storage_mb": storage / (1024 * 1024),
                    "n_alive": n_alive,
                    "quality_distribution": compute_quality_dist(qualities),
                })
                storage_curve.append((step, storage))
            elif step % 100 == 0:
                storage_curve.append((step, compute_storage(png_sizes, qualities, size_ratios)))

    final_storage = compute_storage(png_sizes, qualities, size_ratios)
    storage_curve.append((step, final_storage))
    n_surviving = sum(1 for q in qualities if q != int(ResQuality.DELETED))
    return {
        "review_interval": review_interval,
        "freq_threshold": freq_threshold,
        "storage_curve": storage_curve,
        "review_snapshots": review_snapshots,
        "final_storage_mb": final_storage / (1024 * 1024),
        "n_surviving": n_surviving,
        "quality_distribution": compute_quality_dist(qualities),
    }


def main():
    print("=" * 60)
    print("Resolution-Based Forgetting Simulation")
    print("=" * 60)

    # Load memory index
    print("\nLoading memory index...")
    with open(MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    n = len(memories)
    png_sizes = [len(m.canvas_image_bytes) for m in memories]
    total_png_mb = sum(png_sizes) / (1024 * 1024)
    print(f"  {n} memories, {total_png_mb:.1f} MB total PNG")

    # Measure actual size ratios
    size_ratios = measure_actual_size_ratios(memories)

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except:
        test_pids = sorted(test_data.keys())
    print(f"\n  {len(test_pids)} test samples")

    # Compute hybrid key retrieval candidates
    print("\nComputing hybrid retrieval candidates...")
    img_emb = embeddings
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    query_emb = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")

    key_emb = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    mem_pids = [m.pid for m in memories]

    # Normalize
    q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_emb / q_n
    k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_emb / k_n
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
    print(f"  {len(candidates)} queries with candidates")

    # Run all conditions
    results = {}
    for interval in REVIEW_INTERVALS:
        for threshold in FREQ_THRESHOLDS:
            name = f"res_i{interval}_t{threshold}"
            print(f"\nSimulating {name}...")
            res = simulate(test_pids, candidates, png_sizes, size_ratios,
                           interval, threshold, N_EPOCHS, TOP_K)
            results[name] = res
            print(f"  Surviving: {res['n_surviving']}/{n} ({res['n_surviving']/n*100:.1f}%), "
                  f"Storage: {res['final_storage_mb']:.1f} MB")
            print(f"  Distribution: {res['quality_distribution']}")

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({
            "num_memories": n,
            "total_png_mb": total_png_mb,
            "size_ratios": {QUALITY_LABELS[k]: v for k, v in size_ratios.items()},
            "results": results,
        }, f, indent=2, default=str)

    # =====================================================================
    # Plot: Storage Convergence
    # =====================================================================
    print("\nGenerating storage convergence plot...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = {500: "#e74c3c", 1000: "#3498db", 2000: "#2ecc71"}
    linestyles = {0: "-", 1: "--", 2: ":"}

    for name, res in results.items():
        interval = res["review_interval"]
        threshold = res["freq_threshold"]
        curve = res["storage_curve"]
        steps = [s for s, _ in curve]
        storage_mb = [b / (1024 * 1024) for _, b in curve]
        ax.plot(steps, storage_mb,
                color=colors[interval], linestyle=linestyles[threshold],
                label=f"I={interval}, T={threshold} → {res['final_storage_mb']:.0f}MB ({res['n_surviving']} alive)",
                linewidth=1.8)

    ax.axhline(y=total_png_mb, color="gray", linestyle="--", alpha=0.5,
               label=f"Original ({total_png_mb:.0f} MB, {n} memories)")

    n_test = len(test_pids)
    for e in range(1, N_EPOCHS):
        ax.axvline(x=n_test * e, color="gray", linestyle=":", alpha=0.3)
        ax.text(n_test * e, total_png_mb * 0.97, f"Epoch {e}", ha="center", fontsize=9, alpha=0.5)

    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Total Storage (MB)", fontsize=12)
    ax.set_title("Storage Convergence — Resolution-Based Forgetting (PNG only, cumulative counts)", fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_xlim(0, n_test * N_EPOCHS + 200)
    ax.set_ylim(0, total_png_mb * 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "storage_convergence_resolution.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR / 'storage_convergence_resolution.png'}")

    # =====================================================================
    # Summary table
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"{'Config':20s} | {'Surviving':>10s} | {'Rate':>6s} | {'Storage':>10s} | {'Compress':>8s} | Distribution")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*30}")
    print(f"{'No forgetting':20s} | {n:>10d} | {'100%':>6s} | {total_png_mb:>8.1f}MB | {'0%':>8s} | All PNG@100%")
    for name in sorted(results.keys()):
        res = results[name]
        rate = res['n_surviving'] / n * 100
        compress = (1 - res['final_storage_mb'] / total_png_mb) * 100
        dist_str = ", ".join(f"{k}:{v}" for k, v in res['quality_distribution'].items() if v > 0)
        print(f"{name:20s} | {res['n_surviving']:>10d} | {rate:>5.1f}% | {res['final_storage_mb']:>8.1f}MB | {compress:>6.1f}% | {dist_str}")
    print(f"{'='*80}")
    print(f"\nAll saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
