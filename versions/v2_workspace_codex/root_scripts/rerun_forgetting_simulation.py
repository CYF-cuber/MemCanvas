#!/usr/bin/env python3
"""Re-run forgetting simulation (Phase 1 only) with cumulative counts and generate plot."""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from memory_frequency_forgetting_eval import (
    load_memory_index,
    precompute_retrieval_candidates,
    QualityLevel, SIZE_RATIOS, QUALITY_LABELS,
    compute_storage, compute_quality_dist,
    ExperimentConfig,
)
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    MemoryIndex, MemoryEntry, CLIPLargeMemoryBuilder, ScienceQADataLoader,
)
# Make MemoryEntry available at module level for pickle
import __main__
__main__.MemoryEntry = MemoryEntry

MEMORY_INDEX = "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
REVIEW_INTERVALS = [500, 1000, 2000]
FREQ_THRESHOLDS = [0, 1, 2]
N_EPOCHS = 3
TOP_K = 2
THRESHOLD = 0.1
OUTPUT_DIR = Path("/home/cyf/codex/freq_forgetting_cumulative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate(test_pids, candidates, png_sizes, review_interval, freq_threshold, n_epochs, top_k=2):
    n_memories = len(png_sizes)
    qualities = [int(QualityLevel.PNG)] * n_memories
    retrieval_counts = [0] * n_memories  # cumulative, never reset

    storage_curve = []
    review_snapshots = []
    init_storage = compute_storage(png_sizes, qualities)
    storage_curve.append((0, init_storage))

    step = 0
    for epoch in range(n_epochs):
        for pid in test_pids:
            cands = candidates.get(pid, [])
            effective = []
            for mem_idx, sim in cands:
                if qualities[mem_idx] == int(QualityLevel.DELETED):
                    continue
                effective.append(mem_idx)
                if len(effective) >= top_k:
                    break
            for mem_idx in effective:
                retrieval_counts[mem_idx] += 1

            step += 1

            if step % review_interval == 0:
                n_compressed = 0
                for i in range(n_memories):
                    if qualities[i] == int(QualityLevel.DELETED):
                        continue
                    if retrieval_counts[i] <= freq_threshold:
                        qualities[i] = min(int(qualities[i]) + 1, int(QualityLevel.DELETED))
                        n_compressed += 1
                # NO reset: counts accumulate
                storage = compute_storage(png_sizes, qualities)
                n_alive = sum(1 for q in qualities if q != int(QualityLevel.DELETED))
                review_snapshots.append({
                    "step": step, "epoch": epoch,
                    "storage_mb": storage / (1024 * 1024),
                    "n_alive": n_alive,
                    "n_compressed": n_compressed,
                    "quality_distribution": compute_quality_dist(qualities),
                })
                storage_curve.append((step, storage))
            elif step % 100 == 0:
                storage_curve.append((step, compute_storage(png_sizes, qualities)))

    final_storage = compute_storage(png_sizes, qualities)
    storage_curve.append((step, final_storage))
    n_surviving = sum(1 for q in qualities if q != int(QualityLevel.DELETED))
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
    print("Loading memory index...")
    memories, embeddings, embedding_dim = load_memory_index(MEMORY_INDEX)
    n = len(memories)
    png_sizes = [len(m.canvas_image_bytes) for m in memories]
    total_png_mb = sum(png_sizes) / (1024 * 1024)
    print(f"  {n} memories, {total_png_mb:.1f} MB total PNG")

    # Build MemoryIndex
    mem_index = MemoryIndex(embedding_dim=embedding_dim)
    mem_index.memories = memories
    mem_index.embeddings = embeddings

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # CLIP encoder
    from memory_frequency_forgetting_eval import ExperimentConfig
    config = ExperimentConfig(max_memories_to_retrieve=TOP_K, similarity_threshold=THRESHOLD)
    text_encoder = CLIPLargeMemoryBuilder(config)

    # Pre-compute retrieval candidates
    print("Pre-computing retrieval candidates (top-30)...")
    candidates = precompute_retrieval_candidates(
        test_pids, test_data, text_encoder, mem_index, memories, top_k=30, threshold=THRESHOLD
    )

    # Run all conditions
    results = {}
    for interval in REVIEW_INTERVALS:
        for threshold in FREQ_THRESHOLDS:
            name = f"freq_i{interval}_t{threshold}"
            print(f"\nSimulating {name}...")
            res = simulate(test_pids, candidates, png_sizes, interval, threshold, N_EPOCHS, TOP_K)
            results[name] = res
            print(f"  Surviving: {res['n_surviving']}/{n} ({res['n_surviving']/n*100:.1f}%), "
                  f"Storage: {res['final_storage_mb']:.1f} MB")

    # Save results
    # Convert for JSON
    results_json = {}
    for k, v in results.items():
        results_json[k] = {kk: vv for kk, vv in v.items()}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({"num_memories": n, "total_png_mb": total_png_mb, "results": results_json}, f, indent=2, default=str)

    # Plot storage convergence
    print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = {500: "tab:red", 1000: "tab:blue", 2000: "tab:green"}
    linestyles = {0: "-", 1: "--", 2: ":"}

    for name, res in results.items():
        interval = res["review_interval"]
        threshold = res["freq_threshold"]
        curve = res["storage_curve"]
        steps = [s for s, _ in curve]
        storage_mb = [b / (1024 * 1024) for _, b in curve]
        ax.plot(steps, storage_mb,
                color=colors[interval], linestyle=linestyles[threshold],
                label=f"interval={interval}, threshold={threshold}",
                linewidth=1.5)

    ax.axhline(y=total_png_mb, color="gray", linestyle="--", alpha=0.5, label=f"Original PNG ({total_png_mb:.0f} MB)")

    # Epoch boundaries
    n_test = len(test_pids)
    for e in range(1, N_EPOCHS):
        ax.axvline(x=n_test * e, color="gray", linestyle=":", alpha=0.3)
        ax.text(n_test * e, total_png_mb * 0.97, f"Epoch {e}", ha="center", fontsize=9, alpha=0.5)

    ax.set_xlabel("Simulation Step (test samples processed)", fontsize=12)
    ax.set_ylabel("Total Storage (MB)", fontsize=12)
    ax.set_title("Storage Convergence — Frequency-Based Adaptive Forgetting (Cumulative Counts)", fontsize=13)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_xlim(0, n_test * N_EPOCHS + 200)
    ax.set_ylim(0, total_png_mb * 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "storage_convergence_cumulative.png", dpi=150)
    print(f"Saved to {OUTPUT_DIR / 'storage_convergence_cumulative.png'}")


if __name__ == "__main__":
    main()
