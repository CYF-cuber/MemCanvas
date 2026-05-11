#!/usr/bin/env python3
"""
Generate 4 figures for forgetting mechanism comparison experiments.

Fig A: Strategy comparison bar chart (accuracy at ~26.5% retention)
Fig B: Streaming memory growth curves
Fig C: Streaming retrieval hit rate curves
Fig D: Frequency distribution of kept vs discarded memories

Usage:
  python plot_forgetting_experiments.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUTPUT_DIR = Path("/home/cyf/memcanvas0402/forgetting_experiments")
CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")

# Style
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11
rcParams["axes.linewidth"] = 1.2
rcParams["figure.dpi"] = 150

# Colors
COLORS = {
    "no_forgetting": "#888888",
    "random": "#e74c3c",
    "fifo": "#f39c12",
    "lru": "#3498db",
    "ebbinghaus": "#9b59b6",
    "freq_adaptive": "#2ecc71",
}
LABELS = {
    "no_forgetting": "No Forgetting",
    "random": "Random",
    "fifo": "FIFO",
    "lru": "LRU",
    "ebbinghaus": "Ebbinghaus",
    "freq_adaptive": "Freq-Adaptive\n(Ours)",
}


def plot_fig_a():
    """Strategy comparison: dual bar chart showing accuracy and retention side by side."""
    results = {}

    # Load VLM eval results for all strategies
    for strat in ["no_forgetting", "random", "fifo", "lru", "ebbinghaus", "freq_adaptive"]:
        ckpt_path = OUTPUT_DIR / f"checkpoint_{strat}.json"
        if ckpt_path.exists():
            data = json.load(open(ckpt_path))
            if data.get("complete"):
                surv = data.get("surviving", 12726 if strat == "no_forgetting" else 3374)
                results[strat] = {
                    "acc": data["accuracy"],
                    "surviving": surv,
                    "retention": surv / 12726 * 100,
                }

    order = ["no_forgetting", "random", "fifo", "lru", "ebbinghaus", "freq_adaptive"]
    order = [s for s in order if s in results]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(order))
    width = 0.55

    # Accuracy bars
    bars = ax1.bar(x, [results[s]["acc"] for s in order],
                   color=[COLORS[s] for s in order],
                   edgecolor="black", linewidth=0.8, width=width)

    # Accuracy labels
    for bar, s in zip(bars, order):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Baseline
    ax1.axhline(y=results["no_forgetting"]["acc"], color="#888888",
                linestyle="--", linewidth=1, alpha=0.5)

    # Retention rate as secondary axis (line)
    ax2 = ax1.twinx()
    retention_vals = [results[s]["retention"] for s in order]
    ax2.plot(x, retention_vals, "ko--", markersize=8, linewidth=1.5, alpha=0.7, label="Retention Rate")
    for i, (xi, rv) in enumerate(zip(x, retention_vals)):
        surv = results[order[i]]["surviving"]
        ax2.annotate(f"{rv:.0f}%\n({surv})", (xi, rv),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color="#333333")

    labels = [LABELS[s].replace("\n", " ") for s in order]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(bottom=max(0, min(results[s]["acc"] for s in order) - 4))
    ax2.set_ylabel("Memory Retention (%)", fontsize=12)
    ax2.set_ylim(0, 115)
    ax1.set_title("Forgetting Strategy Comparison on ScienceQA", fontsize=13)
    ax1.grid(axis="y", alpha=0.2)

    # Combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#888888", linestyle="--", label="No Forgetting baseline"),
        Line2D([0], [0], color="black", marker="o", linestyle="--", markersize=6,
               label="Retention Rate", alpha=0.7),
    ]
    ax1.legend(handles=legend_elements, loc="lower left", fontsize=9)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_a_strategy_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig A saved ({len(order)} strategies)")


def plot_fig_b():
    """Streaming memory growth curves."""
    stream_path = OUTPUT_DIR / "streaming_results.json"
    if not stream_path.exists():
        print("  Fig B: streaming_results.json not found, skipping")
        return

    data = json.load(open(stream_path))
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_order = ["no_forgetting", "fifo", "lru", "ebbinghaus", "freq_adaptive"]
    linestyles = {"no_forgetting": "-", "fifo": "--", "lru": "-.", "ebbinghaus": ":", "freq_adaptive": "-"}
    markers = {"no_forgetting": "s", "fifo": "^", "lru": "D", "ebbinghaus": "v", "freq_adaptive": "o"}

    for strat in plot_order:
        if strat not in data:
            continue
        d = data[strat]
        x = list(range(len(d["memory_sizes"])))
        ax.plot(x, d["memory_sizes"], color=COLORS[strat], linestyle=linestyles[strat],
                marker=markers[strat], markersize=6, linewidth=2,
                label=LABELS[strat].replace("\n", " "))

    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Memory Bank Size", fontsize=12)
    ax.set_title("Streaming Memory Growth Under Different Strategies", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(len(data[plot_order[0]]["memory_sizes"])))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_b_streaming_growth.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Fig B saved")


def plot_fig_c():
    """Streaming retrieval hit rate curves."""
    stream_path = OUTPUT_DIR / "streaming_results.json"
    if not stream_path.exists():
        print("  Fig C: streaming_results.json not found, skipping")
        return

    data = json.load(open(stream_path))
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_order = ["no_forgetting", "fifo", "lru", "ebbinghaus", "freq_adaptive"]
    linestyles = {"no_forgetting": "-", "fifo": "--", "lru": "-.", "ebbinghaus": ":", "freq_adaptive": "-"}
    markers = {"no_forgetting": "s", "fifo": "^", "lru": "D", "ebbinghaus": "v", "freq_adaptive": "o"}

    for strat in plot_order:
        if strat not in data:
            continue
        d = data[strat]
        x = list(range(len(d["hit_rates"])))
        ax.plot(x, d["hit_rates"], color=COLORS[strat], linestyle=linestyles[strat],
                marker=markers[strat], markersize=6, linewidth=2,
                label=LABELS[strat].replace("\n", " "))

    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Retrieval Hit Rate (%)", fontsize=12)
    ax.set_title("Retrieval Hit Rate Under Streaming Memory Arrival", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(len(data[plot_order[0]]["hit_rates"])))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_c_streaming_hitrate.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Fig C saved")


def plot_fig_d():
    """Frequency distribution of kept vs discarded memories for each strategy."""
    sim_path = OUTPUT_DIR / "simulation_results.json"
    if not sim_path.exists():
        print("  Fig D: simulation_results.json not found, skipping")
        return

    # We need full freq data — recompute from embeddings
    print("  Fig D: computing frequency distributions...")
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    n_memories = len(img_emb)
    n_test = len(query_emb)

    # Build retrieval map
    keys = 0.00 * img_emb + 1.00 * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T

    # Compute access counts
    access_counts = np.zeros(n_memories, dtype=int)
    for i in range(n_test):
        top = np.argsort(sims[i])[::-1][:7]
        for j in top[:2]:
            if sims[i][j] >= 0.1:
                access_counts[j] += 1

    # Load strategies and compute freq-adaptive quality
    import math
    from collections import Counter

    TARGET_KEEP = 3374
    rmap = {}
    for i in range(n_test):
        top = np.argsort(sims[i])[::-1][:7]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:2]
        rmap[i] = res

    strategies = {}

    # Random
    rng = np.random.RandomState(42)
    q = [4] * n_memories
    for i in rng.choice(n_memories, TARGET_KEEP, replace=False):
        q[i] = 0
    strategies["Random"] = q

    # FIFO
    q = [4] * n_memories
    for i in range(n_memories - TARGET_KEEP, n_memories):
        q[i] = 0
    strategies["FIFO"] = q

    # LRU
    last_acc = [-1] * n_memories
    for qi in range(n_test):
        for mi, s in rmap.get(qi, []):
            last_acc[mi] = qi
    ranked = sorted(range(n_memories), key=lambda i: last_acc[i], reverse=True)
    q = [4] * n_memories
    for i in ranked[:TARGET_KEEP]:
        q[i] = 0
    strategies["LRU"] = q

    # Ebbinghaus
    strength = [1.0] * n_memories
    last_acc_e = [0] * n_memories
    for qi in range(n_test):
        for mi, s in rmap.get(qi, []):
            strength[mi] += 1.0
            last_acc_e[mi] = qi
    retention = [math.exp(-(n_test - last_acc_e[mi]) / max(strength[mi], 1e-8)) for mi in range(n_memories)]
    sorted_r = sorted(retention, reverse=True)
    threshold = sorted_r[min(TARGET_KEEP - 1, len(sorted_r) - 1)]
    q = [0 if retention[i] >= threshold else 4 for i in range(n_memories)]
    actual = sum(1 for x in q if x == 0)
    if actual > TARGET_KEEP:
        kept = [i for i in range(n_memories) if q[i] == 0]
        for i in np.random.RandomState(42).choice(kept, actual - TARGET_KEEP, replace=False):
            q[i] = 4
    strategies["Ebbinghaus"] = q

    # Freq-Adaptive (simulate)
    quality_fa = [0] * n_memories
    ret_count = [0] * n_memories
    for qi in range(n_test):
        for mi, s in rmap.get(qi, []):
            if quality_fa[mi] < 4:
                ret_count[mi] += 1
        if (qi + 1) % 1000 == 0:
            for mi in range(n_memories):
                if quality_fa[mi] >= 4:
                    continue
                if ret_count[mi] <= 1:
                    quality_fa[mi] += 1
    strategies["Freq-Adaptive\n(Ours)"] = quality_fa

    # Plot
    strat_names = ["Random", "FIFO", "LRU", "Ebbinghaus", "Freq-Adaptive\n(Ours)"]
    strat_colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71"]

    fig, axes = plt.subplots(1, len(strat_names), figsize=(16, 4), sharey=True)

    max_freq = int(access_counts.max())
    bins = np.arange(0, max_freq + 2) - 0.5

    for ax, name, color in zip(axes, strat_names, strat_colors):
        q = strategies[name]
        kept_freqs = access_counts[[i for i in range(n_memories) if q[i] < 4]]
        disc_freqs = access_counts[[i for i in range(n_memories) if q[i] >= 4]]

        ax.hist(disc_freqs, bins=bins, alpha=0.5, color="#cccccc", label="Discarded", density=True)
        ax.hist(kept_freqs, bins=bins, alpha=0.7, color=color, label="Kept", density=True)
        ax.set_title(name.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.set_xlabel("Access Frequency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Density")
    fig.suptitle("Frequency Distribution of Kept vs Discarded Memories", fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_d_frequency_distribution.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Fig D saved")


def main():
    print("Generating forgetting experiment figures...\n")
    plot_fig_a()
    plot_fig_b()
    plot_fig_c()
    plot_fig_d()
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
