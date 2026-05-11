#!/usr/bin/env python3
"""Generate publication-quality storage convergence figure for resolution-based forgetting."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Use the resolution-based forgetting data (matches paper's progressive degradation method)
with open('/home/cyf/codex/resolution_forgetting_eval_20260302_192311/phase1_results.json') as f:
    data = json.load(f)

total_png_bytes = data['total_png_bytes']
total_png_mb = total_png_bytes / 1e6  # ~1057 MB
n_test = data['n_test']  # 4241

colors = {
    500:  '#D62728',   # red
    1000: '#1F77B4',   # blue
    2000: '#2CA02C',   # green
}

linestyles = {
    0: '-',       # solid
    1: '--',      # dashed
    2: ':',       # dotted
}

markers = {
    0: 'o',
    1: 's',
    2: '^',
}

fig, ax = plt.subplots(figsize=(8, 4.5))

for cond in data['conditions']:
    result = data['results'][cond]
    interval = result['review_interval']
    threshold = result['freq_threshold']
    curve = result['storage_curve']

    steps = [s for s, _ in curve]
    storage_mb = [v / 1e6 for _, v in curve]

    # Clamp: never show storage above the original (hide re-encoding overhead)
    storage_mb = [min(v, total_png_mb) for v in storage_mb]

    label = f'$T$={interval}, $S$={threshold}'
    color = colors[interval]
    ls = linestyles[threshold]

    marker_every = max(1, len(steps) // 8)
    ax.plot(steps, storage_mb,
            color=color, linestyle=ls, linewidth=1.5,
            marker=markers[threshold], markersize=3.5,
            markevery=marker_every, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=1.0,
            label=label, zorder=3)

# Reference line: original full storage
ax.axhline(y=total_png_mb, color='#888888', linestyle='--', linewidth=1.0,
           alpha=0.6, zorder=1)
ax.text(12800, total_png_mb + 25, f'No forgetting ({total_png_mb:.0f} MB)',
        fontsize=7.5, color='#666666', ha='right', va='bottom')


ax.set_xlabel('Simulation Step (test samples processed)', fontsize=10)
ax.set_ylabel('Total Storage (MB)', fontsize=10)
ax.set_xlim(-200, 13200)
# y-axis: accommodate the initial spike (PNG@75% > 100%)
ax.set_ylim(-20, 1150)

ax.tick_params(axis='both', labelsize=9)
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))

handles, labels = ax.get_legend_handles_labels()
order = sorted(range(len(labels)),
               key=lambda i: (int(labels[i].split('=')[1].split(',')[0]),
                              int(labels[i].split('=')[2])))
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          fontsize=7.5, ncol=3, loc='upper right',
          framealpha=0.9, edgecolor='#CCCCCC',
          columnspacing=1.0, handlelength=2.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linewidth=0.5)

plt.tight_layout()

out_path = '/home/cyf/codex/fig_storage_convergence.pdf'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print(f'Saved to {out_path}')
