#!/usr/bin/env python3
"""Plot memory bank size changes for T&S forgetting ablation."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_FILE = "/home/cyf/memcanvas0402/scienceqa_ablation/abl3_forgetting/forgetting_curves.json"
OUTPUT_DIR = "/home/cyf/memcanvas0402/scienceqa_ablation/abl3_forgetting"

with open(DATA_FILE) as f:
    data = json.load(f)

TOTAL = 12726
N_TEST = 4241

# ============================================================
# Figure 1: Surviving memories over queries (S=1 only)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: surviving count
ax = axes[0]
colors = {"500": "#e74c3c", "1000": "#f39c12", "2000": "#2ecc71"}
markers = {"500": "o", "1000": "s", "2000": "D"}

for key in sorted(data.keys()):
    v = data[key]
    if v["S"] != 1:
        continue
    T = str(v["T"])
    snaps = v["snapshots"]

    # Build timeline: start at full, then each snapshot
    xs = [0] + [s["query_idx"] for s in snaps]
    ys = [TOTAL] + [s["surviving"] for s in snaps]

    acc = v.get("accuracy", 0)
    label = f"T={T} (acc={acc:.1f}%, final={ys[-1]})"
    ax.plot(xs, ys, color=colors[T], marker=markers[T], markersize=6,
            linewidth=2, label=label)

ax.axhline(y=TOTAL, color="gray", linestyle="--", alpha=0.5, label=f"No forgetting ({TOTAL})")
ax.set_xlabel("Number of Queries Processed", fontsize=12)
ax.set_ylabel("Surviving Memories (quality < deleted)", fontsize=12)
ax.set_title("Memory Bank Size During Forgetting (S=1)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="center right")
ax.set_ylim(-200, TOTAL + 500)
ax.set_xlim(-100, N_TEST + 100)
ax.grid(True, alpha=0.3)

# Right: accuracy vs survival rate
ax = axes[1]
configs_s1 = []
for key in sorted(data.keys()):
    v = data[key]
    if v["S"] != 1:
        continue
    T = str(v["T"])
    surv_rate = v["surviving"] / TOTAL * 100
    acc = v.get("accuracy", 0)
    configs_s1.append((T, surv_rate, acc))

# Add no-forgetting baseline
configs_s1.append(("none", 100.0, 88.82))

for T, sr, acc in configs_s1:
    if T == "none":
        ax.scatter(sr, acc, color="gray", s=120, zorder=5, marker="*")
        ax.annotate("No forgetting", (sr, acc), textcoords="offset points",
                    xytext=(-60, 10), fontsize=9)
    else:
        ax.scatter(sr, acc, color=colors[T], s=100, zorder=5, marker=markers[T])
        ax.annotate(f"T={T}", (sr, acc), textcoords="offset points",
                    xytext=(8, -5), fontsize=10)

ax.set_xlabel("Memory Survival Rate (%)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy vs Memory Compression (S=1)", fontsize=13, fontweight="bold")
ax.set_xlim(-5, 110)
ax.set_ylim(85, 90)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/forgetting_curves.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/forgetting_curves.pdf", bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/forgetting_curves.png")

# ============================================================
# Figure 2: Quality distribution stacked bar
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

quality_names = ["Original (1.0x)", "0.75x", "0.5x", "0.25x", "Deleted"]
quality_colors = ["#2ecc71", "#82e0aa", "#f9e79f", "#f5b041", "#e74c3c"]

configs = []
labels = []
for key in ["abl3_T2000_S1", "abl3_T1000_S1", "abl3_T500_S1"]:
    v = data[key]
    dist = v["quality_distribution"]
    counts = [dist.get(str(i), 0) for i in range(5)]
    configs.append(counts)
    acc = v.get("accuracy", 0)
    surv = v["surviving"]
    labels.append(f"T={v['T']}, S=1\nacc={acc:.1f}%\n{surv} alive")

# Add no-forgetting
configs.insert(0, [TOTAL, 0, 0, 0, 0])
labels.insert(0, f"No forgetting\nacc=88.82%\n{TOTAL} alive")

x = np.arange(len(configs))
bottoms = np.zeros(len(configs))

for qi in range(5):
    vals = [c[qi] for c in configs]
    bars = ax.bar(x, vals, bottom=bottoms, color=quality_colors[qi],
                  label=quality_names[qi], edgecolor="white", linewidth=0.5)
    # Add count labels for significant segments
    for i, v in enumerate(vals):
        if v > 300:
            ax.text(x[i], bottoms[i] + v / 2, str(v), ha="center", va="center",
                    fontsize=8, fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Number of Memories", fontsize=12)
ax.set_title("Memory Quality Distribution After Forgetting", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, TOTAL + 500)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/quality_distribution.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/quality_distribution.pdf", bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/quality_distribution.png")

# ============================================================
# Figure 3: Detailed timeline for T=500,S=1 (most dramatic)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

v = data["abl3_T500_S1"]
snaps = v["snapshots"]

xs = [0] + [s["query_idx"] for s in snaps]

# Track quality counts over time from snapshots
# We need to reconstruct quality distribution at each snapshot
# Use the degraded_this_round to show degradation intensity
degraded_per_round = [0] + [s["degraded_this_round"] for s in snaps]
surviving = [TOTAL] + [s["surviving"] for s in snaps]

ax2 = ax.twinx()

line1, = ax.plot(xs, surviving, color="#2ecc71", marker="o", linewidth=2.5,
                 markersize=7, label="Surviving memories", zorder=3)
bars = ax2.bar(xs[1:], degraded_per_round[1:], width=80, color="#e74c3c",
               alpha=0.4, label="Degraded this round")

ax.set_xlabel("Number of Queries Processed", fontsize=12)
ax.set_ylabel("Surviving Memories", fontsize=12, color="#2ecc71")
ax2.set_ylabel("Memories Degraded Per Round", fontsize=12, color="#e74c3c")
ax.set_title("Memory Bank Evolution: T=500, S=1 (Aggressive Forgetting)", fontsize=13, fontweight="bold")

# Annotate key points
ax.annotate(f"{surviving[-1]} alive\n({surviving[-1]/TOTAL*100:.1f}%)",
            xy=(xs[-1], surviving[-1]),
            xytext=(xs[-1] - 800, surviving[-1] + 2000),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=10, fontweight="bold")

lines = [line1, bars]
labels = [l.get_label() for l in [line1]] + ["Degraded this round"]
ax.legend([line1, bars], labels, loc="center right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-200, TOTAL + 1000)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/timeline_T500_S1.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/timeline_T500_S1.pdf", bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/timeline_T500_S1.png")

print("\nAll plots generated.")
