#!/usr/bin/env python3
"""Plot T&S forgetting ablation results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/cyf/memcanvas0402/scienceqa_ablation/abl3_forgetting")
OUT.mkdir(parents=True, exist_ok=True)

# Data from corrected experiments
TOTAL = 12726
BASELINE_ACC = 88.82

data = {
    (250, 0): {"surviving": 1301, "acc": 87.10, "qdist": {0: 412, 1: 337, 2: 285, 3: 267, 4: 11425}},
    (250, 1): {"surviving": 287,  "acc": 86.02, "qdist": {0: 52, 1: 68, 2: 79, 3: 88, 4: 12439}},
    (250, 2): {"surviving": 103,  "acc": 86.04, "qdist": {0: 19, 1: 16, 2: 35, 3: 33, 4: 12623}},
    (500, 0): {"surviving": 2172, "acc": 87.50, "qdist": {0: 749, 1: 552, 2: 432, 3: 439, 4: 10554}},
    (500, 1): {"surviving": 673,  "acc": 86.80, "qdist": {0: 120, 1: 167, 2: 180, 3: 206, 4: 12053}},
    (500, 2): {"surviving": 279,  "acc": 86.25, "qdist": {0: 35, 1: 68, 2: 91, 3: 85, 4: 12447}},
    (750, 0): {"surviving": 2823, "acc": 88.00, "qdist": {0: 1034, 1: 699, 2: 619, 3: 471, 4: 9903}},
    (750, 1): {"surviving": 1056, "acc": 86.63, "qdist": {0: 199, 1: 268, 2: 292, 3: 297, 4: 11670}},
    (750, 2): {"surviving": 478,  "acc": 86.30, "qdist": {0: 70, 1: 124, 2: 140, 3: 144, 4: 12248}},
    (1000, 0): {"surviving": 3374, "acc": 88.12, "qdist": {0: 1301, 1: 871, 2: 651, 3: 551, 4: 9352}},
    (1000, 1): {"surviving": 1447, "acc": 87.17, "qdist": {0: 287, 1: 386, 2: 383, 3: 391, 4: 11279}},
    (1000, 2): {"surviving": 668,  "acc": 86.28, "qdist": {0: 103, 1: 176, 2: 199, 3: 190, 4: 12058}},
}

T_vals = [250, 500, 750, 1000]
S_vals = [0, 1, 2]
colors_S = {0: '#2196F3', 1: '#FF9800', 2: '#F44336'}
markers_S = {0: 'o', 1: 's', 2: '^'}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

# ============================================================
# Plot 1: Surviving memories vs T for different S
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for S in S_vals:
    surv = [data[(T, S)]["surviving"] for T in T_vals]
    rate = [s / TOTAL * 100 for s in surv]
    ax.plot(T_vals, surv, marker=markers_S[S], color=colors_S[S],
            linewidth=2.5, markersize=9, label=f'S={S}')
    for i, (t, s, r) in enumerate(zip(T_vals, surv, rate)):
        ax.annotate(f'{s}\n({r:.1f}%)', (t, s), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8.5, color=colors_S[S])

ax.axhline(y=TOTAL, color='gray', linestyle='--', alpha=0.5, label=f'Total ({TOTAL})')
ax.set_xlabel('Review Interval T (queries)')
ax.set_ylabel('Surviving Memories')
ax.set_title('Memory Bank Size after Forgetting')
ax.set_xticks(T_vals)
ax.legend(loc='upper left')
ax.set_ylim(-200, TOTAL * 1.08)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'surviving_vs_T.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT / 'surviving_vs_T.png'}")
plt.close()

# ============================================================
# Plot 2: Accuracy vs T for different S
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for S in S_vals:
    acc = [data[(T, S)]["acc"] for T in T_vals]
    ax.plot(T_vals, acc, marker=markers_S[S], color=colors_S[S],
            linewidth=2.5, markersize=9, label=f'S={S}')
    for t, a in zip(T_vals, acc):
        ax.annotate(f'{a:.2f}%', (t, a), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, color=colors_S[S])

ax.axhline(y=BASELINE_ACC, color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline ({BASELINE_ACC}%)')
ax.set_xlabel('Review Interval T (queries)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy vs Review Interval T')
ax.set_xticks(T_vals)
ax.legend(loc='lower right')
ax.set_ylim(85, 89.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'accuracy_vs_T.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT / 'accuracy_vs_T.png'}")
plt.close()

# ============================================================
# Plot 3: Accuracy vs Surviving memories (scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for S in S_vals:
    surv = [data[(T, S)]["surviving"] for T in T_vals]
    acc = [data[(T, S)]["acc"] for T in T_vals]
    ax.scatter(surv, acc, marker=markers_S[S], color=colors_S[S],
               s=100, zorder=5, label=f'S={S}')
    for t, s, a in zip(T_vals, surv, acc):
        ax.annotate(f'T={t}', (s, a), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5, color=colors_S[S])

ax.axhline(y=BASELINE_ACC, color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline ({BASELINE_ACC}%)')
ax.axvline(x=TOTAL, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Surviving Memories')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy vs Memory Bank Size')
ax.legend(loc='lower right')
ax.set_ylim(85, 89.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'accuracy_vs_surviving.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT / 'accuracy_vs_surviving.png'}")
plt.close()

# ============================================================
# Plot 4: Quality distribution stacked bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
labels = []
q0_vals, q1_vals, q2_vals, q3_vals, q4_vals = [], [], [], [], []
for T in T_vals:
    for S in S_vals:
        labels.append(f'T={T}\nS={S}')
        d = data[(T, S)]["qdist"]
        q0_vals.append(d[0])
        q1_vals.append(d[1])
        q2_vals.append(d[2])
        q3_vals.append(d[3])
        q4_vals.append(d[4])

x = np.arange(len(labels))
width = 0.65
colors_q = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
q_labels = ['q0 (original)', 'q1 (0.75x)', 'q2 (0.5x)', 'q3 (0.25x)', 'q4 (deleted)']

# Stacked bars (only surviving, exclude q4 for clarity)
bottom = np.zeros(len(labels))
for vals, color, label in zip(
    [q0_vals, q1_vals, q2_vals, q3_vals],
    colors_q[:4], q_labels[:4]
):
    ax.bar(x, vals, width, bottom=bottom, color=color, label=label, edgecolor='white', linewidth=0.5)
    bottom += np.array(vals)

# Add total surviving count on top
for i, (b, total_surv) in enumerate(zip(bottom, [data[(T,S)]["surviving"] for T in T_vals for S in S_vals])):
    if total_surv > 0:
        ax.text(i, b + 50, f'{total_surv}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Configuration')
ax.set_ylabel('Number of Memories')
ax.set_title('Quality Distribution of Surviving Memories')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(OUT / 'quality_distribution.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT / 'quality_distribution.png'}")
plt.close()

print("\nAll plots generated!")
