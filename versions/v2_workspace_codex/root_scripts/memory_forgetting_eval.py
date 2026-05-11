#!/usr/bin/env python3
"""
Forgetting Mechanism Experiment (Experiment 1 from MCL Proposal).

Core idea: "Compression as Forgetting" — instead of binary delete, progressively
compress memories through quality levels: PNG → WebP-85 → AVIF-50 → AVIF-50@0.5x → Delete.

Two-phase architecture:
  Phase 1 (Fast, no VLM): Simulate forgetting strategies, compute storage curves, generate plots.
  Phase 2 (Slow, VLM):    Evaluate accuracy for each (strategy, budget) condition with VLM deduplication.

Usage:
  python memory_forgetting_eval.py --phase1-only   # Quick storage curve visualization
  python memory_forgetting_eval.py                  # Full evaluation including VLM accuracy
"""

import argparse
import copy
import io
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing experiment infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (  # noqa: E402
    MemoryIndex,
    MemoryEntry,  # noqa: F401  (needed for pickle)
    CLIPLargeMemoryBuilder,
    Qwen25VLEvaluator,
    ExperimentConfig,
    ScienceQADataLoader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
)
DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"

# Quality levels with size ratios relative to original PNG (validated data)
class QualityLevel(IntEnum):
    PNG = 0           # ratio = 1.0
    WEBP_85 = 1       # ratio = 0.5795
    AVIF_50 = 2       # ratio = 0.3709
    AVIF_50_HALF = 3  # ratio = 0.00518
    DELETED = 4       # ratio = 0.0

SIZE_RATIOS: Dict[int, float] = {
    QualityLevel.PNG: 1.0,
    QualityLevel.WEBP_85: 0.5795,
    QualityLevel.AVIF_50: 0.3709,
    QualityLevel.AVIF_50_HALF: 0.00518,
    QualityLevel.DELETED: 0.0,
}

QUALITY_LABELS: Dict[int, str] = {
    QualityLevel.PNG: "PNG",
    QualityLevel.WEBP_85: "WebP-85",
    QualityLevel.AVIF_50: "AVIF-50",
    QualityLevel.AVIF_50_HALF: "AVIF-50@0.5x",
    QualityLevel.DELETED: "Deleted",
}

STRATEGIES = ["Random", "FIFO", "LRU", "Ebbinghaus", "Progressive"]
BUDGET_FRACTIONS = [0.75, 0.50, 0.25, 0.10]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimMemory:
    """Simulated memory entry for Phase 1."""
    idx: int               # index in the original memory list
    png_size: int          # original PNG byte size
    quality: QualityLevel  # current quality level
    created_step: int      # step when this memory was created
    last_access_step: int  # last retrieval step (simulated)
    access_count: int      # total retrieval count (simulated)
    relevance: float       # pre-computed average cosine similarity to test queries


# ---------------------------------------------------------------------------
# Helper: image compression (reused from compression eval scripts)
# ---------------------------------------------------------------------------

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_image(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    resample = getattr(Image, "Resampling", Image).LANCZOS
    w = max(1, int(round(img.width * scale)))
    h = max(1, int(round(img.height * scale)))
    return img.resize((w, h), resample=resample)


def compress_to_quality(orig_bytes: bytes, quality: QualityLevel) -> bytes:
    """Compress original PNG bytes to the specified quality level."""
    if quality == QualityLevel.PNG:
        return orig_bytes
    if quality == QualityLevel.DELETED:
        return b""

    img = Image.open(io.BytesIO(orig_bytes))
    img = ensure_rgb(img)

    if quality == QualityLevel.WEBP_85:
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=85, method=6)
        return out.getvalue()
    elif quality == QualityLevel.AVIF_50:
        out = io.BytesIO()
        img.save(out, format="AVIF", quality=50)
        return out.getvalue()
    elif quality == QualityLevel.AVIF_50_HALF:
        img = resize_image(img, 0.5)
        out = io.BytesIO()
        img.save(out, format="AVIF", quality=50)
        return out.getvalue()
    else:
        raise ValueError(f"Unknown quality level: {quality}")


# ---------------------------------------------------------------------------
# Memory index loading
# ---------------------------------------------------------------------------

def load_memory_index(path: str) -> Tuple[List, np.ndarray, int]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    return memories, embeddings, embedding_dim


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def build_output_dir(root: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"forgetting_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Phase 1: Relevance pre-computation
# ---------------------------------------------------------------------------

def compute_relevance_scores(
    memories: List,
    embeddings: np.ndarray,
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute average cosine similarity of each memory to all test query embeddings.
    This serves as a proxy for how "frequently accessed" each memory would be.
    Returns array of shape (num_memories,).
    """
    print("Computing test query embeddings...")
    query_embeddings = []
    for pid in tqdm(test_pids, desc="Encode test queries"):
        problem = test_data[pid]
        question_text = (problem.get("question", "") + " " + problem.get("hint", "")).strip()
        qe = text_encoder.encode_text_query(question_text)
        query_embeddings.append(qe)
    query_embeddings = np.array(query_embeddings)  # (num_test, dim)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms  # (num_memories, dim)

    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    norm_queries = query_embeddings / q_norms  # (num_test, dim)

    # Compute average similarity in batches to manage memory
    print("Computing relevance scores (memory × query similarities)...")
    num_memories = norm_embeddings.shape[0]
    avg_sims = np.zeros(num_memories, dtype=np.float64)

    for start in range(0, num_memories, batch_size):
        end = min(start + batch_size, num_memories)
        # (batch, dim) @ (dim, num_test) -> (batch, num_test)
        sims = norm_embeddings[start:end] @ norm_queries.T
        avg_sims[start:end] = sims.mean(axis=1)

    print(f"  Relevance stats: min={avg_sims.min():.4f}, max={avg_sims.max():.4f}, "
          f"mean={avg_sims.mean():.4f}, std={avg_sims.std():.4f}")
    return avg_sims


# ---------------------------------------------------------------------------
# Phase 1: Forgetting strategy simulation
# ---------------------------------------------------------------------------

def current_storage(mem_list: List[SimMemory]) -> int:
    """Compute total current storage in bytes."""
    total = 0
    for m in mem_list:
        if m.quality != QualityLevel.DELETED:
            total += int(m.png_size * SIZE_RATIOS[m.quality])
    return total


def apply_random_deletion(mem_list: List[SimMemory], budget: int) -> None:
    """Randomly delete memories until under budget."""
    while current_storage(mem_list) > budget:
        alive = [m for m in mem_list if m.quality != QualityLevel.DELETED]
        if not alive:
            break
        victim = random.choice(alive)
        victim.quality = QualityLevel.DELETED


def apply_fifo_deletion(mem_list: List[SimMemory], budget: int) -> None:
    """Delete oldest (lowest created_step) memories first."""
    while current_storage(mem_list) > budget:
        alive = [m for m in mem_list if m.quality != QualityLevel.DELETED]
        if not alive:
            break
        # Oldest = smallest created_step
        victim = min(alive, key=lambda m: m.created_step)
        victim.quality = QualityLevel.DELETED


def apply_lru_deletion(mem_list: List[SimMemory], budget: int) -> None:
    """Delete least-recently-used (by relevance proxy) memories first."""
    while current_storage(mem_list) > budget:
        alive = [m for m in mem_list if m.quality != QualityLevel.DELETED]
        if not alive:
            break
        # Least relevant = lowest relevance score
        victim = min(alive, key=lambda m: m.relevance)
        victim.quality = QualityLevel.DELETED


def apply_ebbinghaus_deletion(
    mem_list: List[SimMemory], budget: int, lambda_param: float = 0.0005
) -> None:
    """
    Ebbinghaus-style binary deletion.
    Deletion probability = 1 - exp(-lambda * rank), where rank is by relevance
    (least relevant = highest rank).
    """
    while current_storage(mem_list) > budget:
        alive = [m for m in mem_list if m.quality != QualityLevel.DELETED]
        if not alive:
            break
        # Sort ascending by relevance → least relevant gets highest rank
        alive_sorted = sorted(alive, key=lambda m: m.relevance)
        deleted_any = False
        for rank, m in enumerate(alive_sorted):
            prob = 1.0 - math.exp(-lambda_param * rank)
            if random.random() < prob:
                m.quality = QualityLevel.DELETED
                deleted_any = True
                if current_storage(mem_list) <= budget:
                    break
        # If probabilistic deletion didn't free enough, force-delete least relevant
        if not deleted_any or current_storage(mem_list) > budget:
            alive = [m for m in mem_list if m.quality != QualityLevel.DELETED]
            if alive:
                victim = min(alive, key=lambda m: m.relevance)
                victim.quality = QualityLevel.DELETED


def apply_progressive_compression(mem_list: List[SimMemory], budget: int) -> None:
    """Placeholder — Progressive uses simulate_progressive_optimal() instead."""
    pass


STRATEGY_FNS = {
    "Random": apply_random_deletion,
    "FIFO": apply_fifo_deletion,
    "LRU": apply_lru_deletion,
    "Ebbinghaus": apply_ebbinghaus_deletion,
    "Progressive": apply_progressive_compression,
}


def _progressive_allocate(
    sorted_sizes: List[int],
    budget: int,
) -> Tuple[int, Dict[str, int], List[int]]:
    """
    Given memory sizes sorted by relevance (descending = most relevant first),
    compute the optimal progressive quality allocation under a storage budget.

    Strategy: keep all memories alive at AVIF-50@0.5x (cost is ~0.5% of PNG),
    then upgrade the most relevant ones to higher quality levels greedily.

    Returns (total_storage, quality_dist_dict, quality_per_memory_list).
    """
    k = len(sorted_sizes)
    if k == 0:
        return 0, {QUALITY_LABELS[ql]: 0 for ql in QualityLevel}, []

    r_half = SIZE_RATIOS[QualityLevel.AVIF_50_HALF]

    # Base cost: all alive at AVIF-50@0.5x
    total_half = sum(s * r_half for s in sorted_sizes)

    if total_half > budget:
        # Even AVIF@0.5x doesn't fit — need to delete some (rare for our budgets)
        cumsum = 0.0
        n_alive = 0
        for s in sorted_sizes:
            cost = s * r_half
            if cumsum + cost <= budget:
                cumsum += cost
                n_alive += 1
            else:
                break
        qualities = [int(QualityLevel.AVIF_50_HALF)] * n_alive + [int(QualityLevel.DELETED)] * (k - n_alive)
        dist = {}
        for ql in QualityLevel:
            dist[QUALITY_LABELS[ql]] = sum(1 for q in qualities if q == int(ql))
        return int(cumsum), dist, qualities

    # All alive — upgrade greedily from most relevant
    upgrade_costs_from_half = {
        QualityLevel.PNG: SIZE_RATIOS[QualityLevel.PNG] - r_half,
        QualityLevel.WEBP_85: SIZE_RATIOS[QualityLevel.WEBP_85] - r_half,
        QualityLevel.AVIF_50: SIZE_RATIOS[QualityLevel.AVIF_50] - r_half,
    }

    qualities = [int(QualityLevel.AVIF_50_HALF)] * k
    remaining = budget - total_half

    for i in range(k):
        s = sorted_sizes[i]
        # Try highest quality first
        upgraded = False
        for target_ql in [QualityLevel.PNG, QualityLevel.WEBP_85, QualityLevel.AVIF_50]:
            cost = s * upgrade_costs_from_half[target_ql]
            if cost <= remaining:
                qualities[i] = int(target_ql)
                remaining -= cost
                upgraded = True
                break
        if not upgraded:
            break  # no budget left for further upgrades

    total_storage = int(budget - remaining)
    dist = {}
    for ql in QualityLevel:
        dist[QUALITY_LABELS[ql]] = sum(1 for q in qualities if q == int(ql))
    return total_storage, dist, qualities


def simulate_progressive_optimal(
    png_sizes: List[int],
    relevance_scores: np.ndarray,
    budget_bytes: int,
) -> Dict[str, Any]:
    """
    Simulate progressive compression with optimal reallocation.

    At each step k, compute the optimal quality allocation for memories {0..k-1}
    under the given budget. Most relevant memories get the highest quality;
    least relevant get compressed to lower levels. All memories survive (none
    deleted) because even at AVIF-50@0.5x, the total cost is ~0.5% of PNG.
    """
    import bisect

    n = len(png_sizes)

    # Global relevance ranking: rank[i] = position when sorted by relevance descending
    global_order = sorted(range(n), key=lambda i: -relevance_scores[i])
    rank = [0] * n
    for pos, idx in enumerate(global_order):
        rank[idx] = pos

    # Maintain a sorted list of (rank, size) pairs as memories are added
    active_ranks: List[int] = []
    active_sizes: List[int] = []  # same order as active_ranks (sorted by rank)
    storage_curve: List[int] = []

    total_active_png = 0  # running sum of all active sizes (PNG equivalent)

    for step in range(n):
        r = rank[step]
        s = png_sizes[step]
        total_active_png += s

        # Insert into sorted position
        pos = bisect.bisect_left(active_ranks, r)
        active_ranks.insert(pos, r)
        active_sizes.insert(pos, s)

        if total_active_png <= budget_bytes:
            # All fit at PNG quality
            storage_curve.append(total_active_png)
        else:
            # Compute optimal allocation
            storage, _, _ = _progressive_allocate(active_sizes, budget_bytes)
            storage_curve.append(storage)

    # Final state: compute full allocation
    _, quality_dist, qualities_sorted = _progressive_allocate(active_sizes, budget_bytes)

    # Map qualities back to original memory indices
    final_qualities = [int(QualityLevel.DELETED)] * n
    for i, r in enumerate(active_ranks):
        # Find original memory index for this rank
        original_idx = global_order[r]
        final_qualities[original_idx] = qualities_sorted[i]

    num_surviving = sum(1 for q in final_qualities if q != int(QualityLevel.DELETED))

    return {
        "strategy": "Progressive",
        "budget_bytes": budget_bytes,
        "budget_fraction": budget_bytes / sum(png_sizes) if sum(png_sizes) > 0 else 0,
        "storage_curve": storage_curve,
        "final_qualities": final_qualities,
        "num_surviving": num_surviving,
        "quality_distribution": quality_dist,
        "final_storage_bytes": storage_curve[-1] if storage_curve else 0,
    }


def simulate_forgetting(
    png_sizes: List[int],
    relevance_scores: np.ndarray,
    strategy: str,
    budget_bytes: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate sequential memory construction with a forgetting strategy.

    Returns dict with:
      - storage_curve: list of storage values at each step
      - final_qualities: list of QualityLevel for each memory
      - num_surviving: count of non-DELETED memories
    """
    # Progressive uses optimal reallocation (special path)
    if strategy == "Progressive":
        return simulate_progressive_optimal(png_sizes, relevance_scores, budget_bytes)

    random.seed(seed)
    n = len(png_sizes)
    mem_list: List[SimMemory] = []
    storage_curve = []  # storage in bytes after each addition

    apply_fn = STRATEGY_FNS[strategy]

    for step in range(n):
        sim_mem = SimMemory(
            idx=step,
            png_size=png_sizes[step],
            quality=QualityLevel.PNG,
            created_step=step,
            last_access_step=step,
            access_count=0,
            relevance=float(relevance_scores[step]),
        )
        mem_list.append(sim_mem)

        # Check if over budget
        if current_storage(mem_list) > budget_bytes:
            apply_fn(mem_list, budget_bytes)

        storage_curve.append(current_storage(mem_list))

    # Final state
    final_qualities = [int(m.quality) for m in mem_list]
    num_surviving = sum(1 for m in mem_list if m.quality != QualityLevel.DELETED)

    # Quality distribution
    quality_dist = {}
    for ql in QualityLevel:
        quality_dist[QUALITY_LABELS[ql]] = sum(1 for m in mem_list if m.quality == ql)

    return {
        "strategy": strategy,
        "budget_bytes": budget_bytes,
        "budget_fraction": budget_bytes / sum(png_sizes) if sum(png_sizes) > 0 else 0,
        "storage_curve": storage_curve,
        "final_qualities": final_qualities,
        "num_surviving": num_surviving,
        "quality_distribution": quality_dist,
        "final_storage_bytes": storage_curve[-1] if storage_curve else 0,
    }


# ---------------------------------------------------------------------------
# Phase 1: Run all simulations
# ---------------------------------------------------------------------------

def run_phase1(
    memories: List,
    embeddings: np.ndarray,
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    output_dir: Path,
) -> Dict:
    """Run Phase 1: simulate forgetting strategies and generate plots."""
    n = len(memories)
    png_sizes = [len(m.canvas_image_bytes) for m in memories]
    total_png_bytes = sum(png_sizes)

    print(f"\nPhase 1: Forgetting Strategy Simulation")
    print(f"  Memories: {n}")
    print(f"  Total PNG size: {total_png_bytes:,} bytes ({total_png_bytes / 1024 / 1024:.1f} MB)")

    # Compute relevance scores
    relevance_scores = compute_relevance_scores(
        memories, embeddings, test_pids, test_data, text_encoder
    )

    # Run simulations for each (strategy, budget)
    results = {}
    budgets = {}
    for frac in BUDGET_FRACTIONS:
        budget_bytes = int(total_png_bytes * frac)
        budgets[f"{int(frac*100)}%"] = budget_bytes
        print(f"\n  Budget {int(frac*100)}%: {budget_bytes:,} bytes ({budget_bytes / 1024 / 1024:.1f} MB)")

    for strategy in STRATEGIES:
        for frac in BUDGET_FRACTIONS:
            budget_bytes = int(total_png_bytes * frac)
            label = f"{strategy}_{int(frac*100)}pct"
            print(f"  Simulating: {strategy} @ {int(frac*100)}%...")
            result = simulate_forgetting(
                png_sizes, relevance_scores, strategy, budget_bytes
            )
            results[label] = result

    phase1_data = {
        "num_memories": n,
        "total_png_bytes": total_png_bytes,
        "png_sizes": png_sizes,
        "budgets": budgets,
        "strategies": STRATEGIES,
        "budget_fractions": BUDGET_FRACTIONS,
        "results": results,
        "relevance_scores": relevance_scores.tolist(),
    }

    # Save phase1 results
    results_path = output_dir / "phase1_results.json"
    # Convert storage curves to a serializable format (they can be huge)
    save_data = copy.deepcopy(phase1_data)
    # Keep relevance scores and png_sizes separate for size reasons
    del save_data["relevance_scores"]
    del save_data["png_sizes"]
    # Downsample storage curves for JSON (keep every 10th point + last)
    for key, val in save_data["results"].items():
        curve = val["storage_curve"]
        if len(curve) > 1000:
            step = max(1, len(curve) // 1000)
            sampled = curve[::step]
            if sampled[-1] != curve[-1]:
                sampled.append(curve[-1])
            val["storage_curve_sampled"] = sampled
            val["storage_curve_full_length"] = len(curve)
            del val["storage_curve"]
        else:
            val["storage_curve_sampled"] = curve
            val["storage_curve_full_length"] = len(curve)
            del val["storage_curve"]

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Phase 1 results saved: {results_path}")

    # Generate plots
    generate_plots(phase1_data, output_dir)

    return phase1_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(phase1_data: Dict, output_dir: Path) -> None:
    """Generate 4 figures from Phase 1 simulation data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = phase1_data["results"]
    total_png_bytes = phase1_data["total_png_bytes"]
    n = phase1_data["num_memories"]

    # Color scheme
    strategy_colors = {
        "Random": "#e74c3c",
        "FIFO": "#e67e22",
        "LRU": "#2ecc71",
        "Ebbinghaus": "#3498db",
        "Progressive": "#9b59b6",
    }
    strategy_markers = {
        "Random": "x",
        "FIFO": "^",
        "LRU": "s",
        "Ebbinghaus": "D",
        "Progressive": "o",
    }

    # -----------------------------------------------------------------------
    # Plot 1: Storage Curves at budget=25%
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    target_budget_pct = 25
    budget_bytes = int(total_png_bytes * target_budget_pct / 100)

    for strategy in STRATEGIES:
        key = f"{strategy}_{target_budget_pct}pct"
        if key not in results:
            continue
        curve = results[key]["storage_curve"]
        steps = list(range(len(curve)))
        curve_mb = [v / (1024 * 1024) for v in curve]
        ax.plot(
            steps, curve_mb,
            color=strategy_colors[strategy],
            label=strategy,
            linewidth=1.5,
            alpha=0.85,
        )

    # Budget line
    ax.axhline(
        y=budget_bytes / (1024 * 1024),
        color="red", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Budget ({target_budget_pct}% = {budget_bytes / 1024 / 1024:.0f} MB)",
    )
    # No-forgetting line (cumulative PNG) using actual per-memory sizes
    png_sizes_list = phase1_data["png_sizes"]
    cumulative = []
    running = 0
    for sz in png_sizes_list:
        running += sz
        cumulative.append(running / (1024 * 1024))
    ax.plot(
        range(len(cumulative)), cumulative,
        color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
        label="No Forgetting (cumulative PNG)",
    )

    ax.set_xlabel("Memory Construction Step", fontsize=12)
    ax.set_ylabel("Storage (MB)", fontsize=12)
    ax.set_title(f"Storage Curves — Budget {target_budget_pct}%", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "storage_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: storage_curves.png")

    # -----------------------------------------------------------------------
    # Plot 2: Accuracy vs Storage Budget (placeholder — filled in Phase 2)
    # For Phase 1, show memory survival count instead
    # -----------------------------------------------------------------------
    # We generate a placeholder that will be updated in Phase 2
    fig, ax = plt.subplots(figsize=(10, 6))
    budget_pcts = [int(f * 100) for f in BUDGET_FRACTIONS]
    x_pos = np.arange(len(budget_pcts))
    width = 0.15

    for i, strategy in enumerate(STRATEGIES):
        survival_counts = []
        for pct in budget_pcts:
            key = f"{strategy}_{pct}pct"
            survival_counts.append(results[key]["num_surviving"])
        ax.bar(
            x_pos + i * width, survival_counts, width,
            label=strategy, color=strategy_colors[strategy], alpha=0.8,
        )

    ax.set_xlabel("Storage Budget (% of original)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Storage Budget (Phase 2 required)", fontsize=14)
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels([f"{p}%" for p in budget_pcts])
    ax.text(
        0.5, 0.5, "Run without --phase1-only\nfor accuracy data",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=16, color="gray", alpha=0.5,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: accuracy_vs_budget.png (placeholder — run Phase 2 for accuracy)")

    # -----------------------------------------------------------------------
    # Plot 3: Quality Distribution for Progressive strategy
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    quality_names = ["PNG", "WebP-85", "AVIF-50", "AVIF-50@0.5x", "Deleted"]
    quality_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#95a5a6"]
    x_labels = [f"{p}%" for p in budget_pcts]

    bottom = np.zeros(len(budget_pcts))
    for qi, qname in enumerate(quality_names):
        counts = []
        for pct in budget_pcts:
            key = f"Progressive_{pct}pct"
            dist = results[key]["quality_distribution"]
            counts.append(dist.get(qname, 0))
        counts_arr = np.array(counts, dtype=float)
        # Convert to percentages
        totals = np.array([
            sum(results[f"Progressive_{pct}pct"]["quality_distribution"].values())
            for pct in budget_pcts
        ], dtype=float)
        pcts_arr = (counts_arr / totals) * 100 if totals.any() else counts_arr
        ax.bar(x_labels, pcts_arr, bottom=bottom, label=qname, color=quality_colors[qi], alpha=0.85)
        bottom += pcts_arr

    ax.set_xlabel("Storage Budget", fontsize=12)
    ax.set_ylabel("% of Memories", fontsize=12)
    ax.set_title("Quality Distribution — Progressive Strategy", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: quality_distribution.png")

    # -----------------------------------------------------------------------
    # Plot 4: Memory Survival Count
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(budget_pcts))
    width = 0.15

    for i, strategy in enumerate(STRATEGIES):
        survival_counts = []
        for pct in budget_pcts:
            key = f"{strategy}_{pct}pct"
            survival_counts.append(results[key]["num_surviving"])
        ax.bar(
            x_pos + i * width, survival_counts, width,
            label=strategy, color=strategy_colors[strategy], alpha=0.8,
        )

    ax.set_xlabel("Storage Budget (% of original)", fontsize=12)
    ax.set_ylabel("Surviving Memories", fontsize=12)
    ax.set_title("Memory Survival Count by Strategy and Budget", fontsize=14)
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels([f"{p}%" for p in budget_pcts])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add total count reference line
    ax.axhline(y=n, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Total ({n})")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "memory_survival.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: memory_survival.png")


# ---------------------------------------------------------------------------
# Phase 2: VLM Accuracy Evaluation
# ---------------------------------------------------------------------------

def precompute_retrieval_top10(
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    mem_index: MemoryIndex,
    threshold: float = 0.1,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    For each test sample, pre-compute top-10 retrieval (instead of top-2)
    to handle deletions gracefully.
    Returns mapping pid -> list of (memory_index, similarity).
    """
    retrieved_map: Dict[str, List[Tuple[int, float]]] = {}
    mem_id_to_index = {id(m): i for i, m in enumerate(mem_index.memories)}

    for pid in tqdm(test_pids, desc="Precompute retrieval (top-10)"):
        problem = test_data[pid]
        question_text = (problem.get("question", "") + " " + problem.get("hint", "")).strip()
        query_embedding = text_encoder.encode_text_query(question_text)
        retrieved = mem_index.search(query_embedding, top_k=10, threshold=threshold)

        filtered = []
        for m, s in retrieved:
            if m.pid == pid:
                continue
            filtered.append((mem_id_to_index[id(m)], s))
        retrieved_map[pid] = filtered

    return retrieved_map


def compute_effective_retrieval(
    test_pids: List[str],
    retrieved_map: Dict[str, List[Tuple[int, float]]],
    final_qualities: List[int],
    top_k: int = 2,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    For each test sample, walk top-10 list, skip DELETED, take first top_k
    surviving memories with their quality levels.

    Returns mapping pid -> list of (memory_index, quality_level, similarity).
    """
    effective_map: Dict[str, List[Tuple[int, int, float]]] = {}
    for pid in test_pids:
        candidates = retrieved_map.get(pid, [])
        effective = []
        for mem_idx, sim in candidates:
            ql = final_qualities[mem_idx]
            if ql == QualityLevel.DELETED:
                continue
            effective.append((mem_idx, ql, sim))
            if len(effective) >= top_k:
                break
        effective_map[pid] = effective
    return effective_map


def build_dedup_groups(
    test_pids: List[str],
    conditions: Dict[str, Dict[str, List[Tuple[int, int, float]]]],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[Tuple[int, int, float]]]]]:
    """
    Group conditions that produce identical (retrieval_indices, quality_levels)
    tuples for all test samples → run VLM only once per unique group.

    Returns:
      - group_to_conditions: mapping group_key -> list of condition labels
      - group_effective_maps: mapping group_key -> effective_map
    """
    # Build a fingerprint for each condition
    condition_fingerprints: Dict[str, str] = {}
    for cond_label, eff_map in conditions.items():
        parts = []
        for pid in test_pids:
            entries = eff_map.get(pid, [])
            # Fingerprint: (mem_idx, quality) pairs
            fp = tuple((idx, ql) for idx, ql, _ in entries)
            parts.append(str(fp))
        fingerprint = "|".join(parts)
        condition_fingerprints[cond_label] = fingerprint

    # Group by fingerprint
    fp_to_conditions: Dict[str, List[str]] = {}
    for cond_label, fp in condition_fingerprints.items():
        if fp not in fp_to_conditions:
            fp_to_conditions[fp] = []
        fp_to_conditions[fp].append(cond_label)

    # Build group maps
    group_to_conditions: Dict[str, List[str]] = {}
    group_effective_maps: Dict[str, Dict[str, List[Tuple[int, int, float]]]] = {}
    for i, (fp, cond_labels) in enumerate(fp_to_conditions.items()):
        group_key = f"group_{i}"
        group_to_conditions[group_key] = cond_labels
        # Use the effective_map from the first condition in the group
        group_effective_maps[group_key] = conditions[cond_labels[0]]

    return group_to_conditions, group_effective_maps


def eval_with_vlm(
    group_key: str,
    effective_map: Dict[str, List[Tuple[int, int, float]]],
    memories: List,
    vlm: Qwen25VLEvaluator,
    test_pids: List[str],
    test_data: Dict,
) -> Dict:
    """
    Evaluate accuracy for a single dedup group using VLM.
    Compresses memory bytes to appropriate quality before feeding to VLM.
    """
    choice_labels = ["A", "B", "C", "D", "E", "F"]
    correct = 0
    total = 0

    # Cache compressed bytes to avoid re-compression
    compressed_cache: Dict[Tuple[int, int], bytes] = {}

    for pid in tqdm(test_pids, desc=f"Eval {group_key}"):
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"

        entries = effective_map.get(pid, [])
        if not entries:
            # No memory available — predict without memory
            pred = vlm.predict(problem, retrieved_memories=None)
        else:
            retrieved = []
            for mem_idx, ql, sim in entries:
                mem = memories[mem_idx]
                cache_key = (mem_idx, ql)
                if cache_key not in compressed_cache:
                    compressed_cache[cache_key] = compress_to_quality(
                        mem.canvas_image_bytes, QualityLevel(ql)
                    )
                # Create a temporary MemoryEntry with compressed bytes
                temp_mem = copy.copy(mem)
                temp_mem.canvas_image_bytes = compressed_cache[cache_key]
                retrieved.append((temp_mem, sim))

            pred = vlm.predict(problem, retrieved_memories=retrieved)

        if pred == correct_answer:
            correct += 1
        total += 1

    acc = (correct / total) * 100 if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def eval_baseline_no_memory(
    vlm: Qwen25VLEvaluator,
    test_pids: List[str],
    test_data: Dict,
) -> Dict:
    """Evaluate baseline accuracy without any memory."""
    choice_labels = ["A", "B", "C", "D", "E", "F"]
    correct = 0
    total = 0
    for pid in tqdm(test_pids, desc="Eval baseline (no memory)"):
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"
        pred = vlm.predict(problem, retrieved_memories=None)
        if pred == correct_answer:
            correct += 1
        total += 1
    acc = (correct / total) * 100 if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def run_phase2(
    memories: List,
    embeddings: np.ndarray,
    embedding_dim: int,
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    phase1_data: Dict,
    output_dir: Path,
    top_k: int = 2,
    threshold: float = 0.1,
) -> Dict:
    """Run Phase 2: VLM accuracy evaluation with deduplication."""
    print(f"\nPhase 2: VLM Accuracy Evaluation")

    # Build MemoryIndex for retrieval
    mem_index = MemoryIndex(embedding_dim=embedding_dim)
    mem_index.memories = memories
    mem_index.embeddings = embeddings

    # Pre-compute top-10 retrieval
    retrieved_map = precompute_retrieval_top10(
        test_pids, test_data, text_encoder, mem_index, threshold
    )

    # Initialize VLM
    config = ExperimentConfig(
        max_memories_to_retrieve=top_k,
        similarity_threshold=threshold,
    )
    print("Loading VLM evaluator...")
    vlm = Qwen25VLEvaluator(config)

    # Load checkpoint if available
    checkpoint_path = output_dir / "phase2_checkpoint.json"
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        print(f"\n  Resumed from checkpoint")

    # Baseline evaluation
    if "baseline" in checkpoint:
        baseline_result = checkpoint["baseline"]
        print(f"\n  Baseline (cached): {baseline_result['accuracy']:.2f}%")
    else:
        print("\nEvaluating baseline (no memory)...")
        baseline_result = eval_baseline_no_memory(vlm, test_pids, test_data)
        print(f"  Baseline accuracy: {baseline_result['accuracy']:.2f}%")

    # Oracle evaluation (full memory, PNG quality)
    if "oracle" in checkpoint:
        oracle_result = checkpoint["oracle"]
        print(f"  Oracle (cached): {oracle_result['accuracy']:.2f}%")
    else:
        print("\nEvaluating oracle (full memory, PNG)...")
        oracle_qualities = [int(QualityLevel.PNG)] * len(memories)
        oracle_eff = compute_effective_retrieval(
            test_pids, retrieved_map, oracle_qualities, top_k
        )
        oracle_result = eval_with_vlm(
            "oracle", oracle_eff, memories, vlm, test_pids, test_data
        )
        print(f"  Oracle accuracy: {oracle_result['accuracy']:.2f}%")

    # Build effective retrieval for each (strategy, budget) condition
    conditions: Dict[str, Dict[str, List[Tuple[int, int, float]]]] = {}
    phase1_results = phase1_data["results"]
    for key, res in phase1_results.items():
        final_quals = res["final_qualities"]
        eff_map = compute_effective_retrieval(
            test_pids, retrieved_map, final_quals, top_k
        )
        conditions[key] = eff_map

    # Deduplication
    group_to_conditions, group_effective_maps = build_dedup_groups(
        test_pids, conditions
    )
    num_unique = len(group_to_conditions)
    num_total = len(conditions)
    print(f"\n  Deduplication: {num_total} conditions → {num_unique} unique groups")

    # Load group results from checkpoint
    group_results: Dict[str, Dict] = checkpoint.get("group_results", {})
    if group_results:
        print(f"\n  Checkpoint: {len(group_results)} groups already evaluated")

    # Evaluate each unique group (skip already completed)
    for group_key, eff_map in group_effective_maps.items():
        if group_key in group_results:
            print(f"\n  Skipping {group_key} (already evaluated: {group_results[group_key]['accuracy']:.2f}%)")
            continue
        cond_labels = group_to_conditions[group_key]
        print(f"\n  Evaluating {group_key} (covers: {', '.join(cond_labels)})...")
        result = eval_with_vlm(
            group_key, eff_map, memories, vlm, test_pids, test_data
        )
        group_results[group_key] = result
        print(f"  Accuracy: {result['accuracy']:.2f}%")

        # Save checkpoint after each group
        checkpoint_data = {
            "baseline": baseline_result,
            "oracle": oracle_result,
            "group_results": group_results,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  Checkpoint saved ({len(group_results)}/{num_unique} groups done)")

    # Map results back to conditions
    condition_accuracies: Dict[str, Dict] = {}
    for group_key, cond_labels in group_to_conditions.items():
        for cond_label in cond_labels:
            condition_accuracies[cond_label] = group_results[group_key]

    phase2_data = {
        "baseline": baseline_result,
        "oracle": oracle_result,
        "condition_accuracies": condition_accuracies,
        "dedup_stats": {
            "total_conditions": num_total,
            "unique_groups": num_unique,
            "group_to_conditions": group_to_conditions,
        },
    }

    # Save
    results_path = output_dir / "phase2_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(phase2_data, f, indent=2)
    print(f"\n  Phase 2 results saved: {results_path}")

    # Update accuracy plot
    generate_accuracy_plot(phase1_data, phase2_data, output_dir)

    return phase2_data


def generate_accuracy_plot(
    phase1_data: Dict, phase2_data: Dict, output_dir: Path
) -> None:
    """Generate the accuracy vs budget plot with real VLM data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategy_colors = {
        "Random": "#e74c3c",
        "FIFO": "#e67e22",
        "LRU": "#2ecc71",
        "Ebbinghaus": "#3498db",
        "Progressive": "#9b59b6",
    }
    strategy_markers = {
        "Random": "x",
        "FIFO": "^",
        "LRU": "s",
        "Ebbinghaus": "D",
        "Progressive": "o",
    }

    budget_pcts = [int(f * 100) for f in BUDGET_FRACTIONS]
    condition_acc = phase2_data["condition_accuracies"]
    baseline_acc = phase2_data["baseline"]["accuracy"]
    oracle_acc = phase2_data["oracle"]["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in STRATEGIES:
        accs = []
        for pct in budget_pcts:
            key = f"{strategy}_{pct}pct"
            if key in condition_acc:
                accs.append(condition_acc[key]["accuracy"])
            else:
                accs.append(0)
        ax.plot(
            budget_pcts, accs,
            color=strategy_colors[strategy],
            marker=strategy_markers[strategy],
            label=strategy,
            linewidth=2,
            markersize=8,
        )

    # Reference lines
    ax.axhline(y=oracle_acc, color="green", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Oracle (no forgetting): {oracle_acc:.2f}%")
    ax.axhline(y=baseline_acc, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Baseline (no memory): {baseline_acc:.2f}%")

    ax.set_xlabel("Storage Budget (% of original)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Storage Budget", fontsize=14)
    ax.set_xticks(budget_pcts)
    ax.set_xticklabels([f"{p}%" for p in budget_pcts])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Plot updated: accuracy_vs_budget.png")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    phase1_data: Dict,
    phase2_data: Optional[Dict],
    output_dir: Path,
) -> None:
    """Generate a markdown summary report."""
    lines = []
    lines.append("# Forgetting Mechanism Experiment Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    n = phase1_data["num_memories"]
    total_mb = phase1_data["total_png_bytes"] / (1024 * 1024)
    lines.append("## Dataset")
    lines.append(f"- Memories: {n}")
    lines.append(f"- Total PNG size: {total_mb:.1f} MB")
    lines.append("")

    lines.append("## Storage Budgets")
    for label, bval in phase1_data["budgets"].items():
        lines.append(f"- {label}: {bval / (1024 * 1024):.1f} MB")
    lines.append("")

    # Phase 1 summary
    lines.append("## Phase 1: Simulation Results")
    lines.append("")
    lines.append("### Memory Survival Counts")
    lines.append("| Strategy | 75% | 50% | 25% | 10% |")
    lines.append("|----------|----:|----:|----:|----:|")
    for strategy in STRATEGIES:
        row = [strategy]
        for frac in BUDGET_FRACTIONS:
            key = f"{strategy}_{int(frac*100)}pct"
            res = phase1_data["results"][key]
            row.append(str(res["num_surviving"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### Final Storage (MB)")
    lines.append("| Strategy | 75% | 50% | 25% | 10% |")
    lines.append("|----------|----:|----:|----:|----:|")
    for strategy in STRATEGIES:
        row = [strategy]
        for frac in BUDGET_FRACTIONS:
            key = f"{strategy}_{int(frac*100)}pct"
            res = phase1_data["results"][key]
            row.append(f"{res['final_storage_bytes'] / (1024 * 1024):.1f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Progressive quality distribution
    lines.append("### Progressive Strategy — Quality Distribution")
    lines.append("| Budget | PNG | WebP-85 | AVIF-50 | AVIF-50@0.5x | Deleted |")
    lines.append("|--------|----:|--------:|--------:|-------------:|--------:|")
    for frac in BUDGET_FRACTIONS:
        key = f"Progressive_{int(frac*100)}pct"
        dist = phase1_data["results"][key]["quality_distribution"]
        total = sum(dist.values())
        row = [f"{int(frac*100)}%"]
        for qname in ["PNG", "WebP-85", "AVIF-50", "AVIF-50@0.5x", "Deleted"]:
            cnt = dist.get(qname, 0)
            pct = (cnt / total * 100) if total > 0 else 0
            row.append(f"{cnt} ({pct:.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Phase 2 summary
    if phase2_data:
        lines.append("## Phase 2: VLM Accuracy Results")
        lines.append("")
        lines.append(f"- Baseline (no memory): {phase2_data['baseline']['accuracy']:.2f}%")
        lines.append(f"- Oracle (no forgetting): {phase2_data['oracle']['accuracy']:.2f}%")
        lines.append("")

        lines.append("### Accuracy by Strategy and Budget")
        lines.append("| Strategy | 75% | 50% | 25% | 10% |")
        lines.append("|----------|----:|----:|----:|----:|")
        cond_acc = phase2_data["condition_accuracies"]
        for strategy in STRATEGIES:
            row = [strategy]
            for frac in BUDGET_FRACTIONS:
                key = f"{strategy}_{int(frac*100)}pct"
                if key in cond_acc:
                    row.append(f"{cond_acc[key]['accuracy']:.2f}%")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        dedup = phase2_data["dedup_stats"]
        lines.append(f"### Deduplication")
        lines.append(f"- Total conditions: {dedup['total_conditions']}")
        lines.append(f"- Unique VLM evaluations: {dedup['unique_groups']}")
        lines.append(f"- Savings: {dedup['total_conditions'] - dedup['unique_groups']} redundant evaluations avoided")
        lines.append("")
    else:
        lines.append("## Phase 2: Not run (use without --phase1-only for accuracy data)")
        lines.append("")

    lines.append("## Plots")
    lines.append("- `storage_curves.png` — Storage usage over memory construction steps")
    lines.append("- `accuracy_vs_budget.png` — Accuracy vs storage budget")
    lines.append("- `quality_distribution.png` — Quality level distribution for Progressive strategy")
    lines.append("- `memory_survival.png` — Number of surviving memories per strategy/budget")
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    report_path = output_dir / "forgetting_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forgetting Mechanism Experiment (MCL Proposal Experiment 1)"
    )
    parser.add_argument("--memory-index", default=DEFAULT_MEMORY_INDEX,
                        help="Path to memory index pickle")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Root directory for output")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 (simulation + plots, no VLM)")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="Limit test samples (0 = full set)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of memories to retrieve per test sample")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Similarity threshold for retrieval")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume Phase 2 from existing output directory")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists(), f"Resume dir not found: {output_dir}"
        print(f"Resuming from: {output_dir}")
    else:
        output_dir = build_output_dir(args.output_root)
    print(f"Output directory: {output_dir}")

    # Save config (skip if resuming)
    if not args.resume:
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    # Load memory index
    print(f"\nLoading memory index: {args.memory_index}")
    memories, embeddings, embedding_dim = load_memory_index(args.memory_index)
    print(f"  Loaded {len(memories)} memories, embedding dim={embedding_dim}")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())

    if args.max_test_samples and args.max_test_samples > 0:
        test_pids = test_pids[:args.max_test_samples]
    print(f"  Test samples: {len(test_pids)}")

    # Initialize CLIP text encoder (needed for relevance scoring and retrieval)
    config = ExperimentConfig(
        max_memories_to_retrieve=args.top_k,
        similarity_threshold=args.threshold,
    )
    print("\nLoading CLIP text encoder...")
    text_encoder = CLIPLargeMemoryBuilder(config)

    # Phase 1 (skip if resuming — load from saved results)
    if args.resume:
        phase1_path = output_dir / "phase1_results.json"
        assert phase1_path.exists(), f"Phase 1 results not found: {phase1_path}"
        with open(phase1_path, "r") as f:
            phase1_data = json.load(f)
        print(f"\n  Loaded Phase 1 results from {phase1_path}")
    else:
        phase1_data = run_phase1(
            memories, embeddings, test_pids, test_data, text_encoder, output_dir
        )

    # Phase 2 (optional)
    phase2_data = None
    if not args.phase1_only:
        phase2_data = run_phase2(
            memories, embeddings, embedding_dim,
            test_pids, test_data, text_encoder,
            phase1_data, output_dir,
            top_k=args.top_k,
            threshold=args.threshold,
        )

    # Generate report
    generate_report(phase1_data, phase2_data, output_dir)

    print(f"\nDone. Results in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
