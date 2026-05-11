#!/usr/bin/env python3
"""Generate all figures for the MemCanvas paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from PIL import Image
import os, io, requests

OUT = Path("/home/cyf/memcanvas0402/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'no_ret': '#9E9E9E',
    'text_rag': '#42A5F5',
    'memcanvas': '#EF5350',
    'gpt4o': '#FF9800',
}


# ============================================================
# Fig 1: System Overview (DALL-E 3)
# ============================================================
def generate_fig1():
    print("Generating Fig 1 (DALL-E 3)...")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  OPENAI_API_KEY not set, skipping Fig 1")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = """Create a clean, professional academic paper system diagram for "MemCanvas" — a visual memory framework for Vision-Language Models.

The diagram should have TWO main phases, arranged left-to-right:

LEFT SIDE — "Training Phase (Offline)":
- A stack of training samples (Q&A pairs with images/text) flows into
- A "Canvas Renderer" box that produces structured canvas images
- These canvas images flow into a "CLIP Encoder" producing embeddings
- Embeddings are stored in a "Memory Bank" (depicted as a database/grid)

RIGHT SIDE — "Inference Phase (Online)":
- A new query enters a "CLIP Encoder"
- The query embedding performs "Hybrid Retrieval" (α·img + (1-α)·text) against the Memory Bank
- Top-K retrieved canvas images are selected
- These canvases plus the query go into a "VLM (Qwen2.5-VL-7B)" box
- Output: Answer

BOTTOM — "Memory Consolidation":
- A small feedback loop showing frequency-based forgetting: low-frequency canvases get progressively degraded (resolution reduction) or deleted

Style: White background, clean lines, blue/gray color scheme, rounded rectangles for components, arrows showing data flow. Academic paper quality. No text smaller than readable size. Horizontal layout, landscape orientation."""

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        n=1,
    )

    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    with open(OUT / "fig1_overview.png", "wb") as f:
        f.write(img_data)
    print(f"  Saved: {OUT / 'fig1_overview.png'} ({len(img_data)//1024}KB)")


# ============================================================
# Fig 2: Canvas Examples (4 datasets)
# ============================================================
def generate_fig2():
    print("Generating Fig 2 (Canvas examples)...")
    canvas_paths = {
        '(a) ScienceQA': '/home/cyf/codex/scienceqa_smart_canvases/00000.png',
        '(b) OK-VQA': '/home/cyf/codex/okvqa_data/canvases_smart/00000.png',
        '(c) MMQA': '/home/cyf/codex/mmqa_data/canvases_smart/00000.png',
        '(d) HotpotQA': '/home/cyf/codex/hotpotqa_data/canvases_smart/00000.png',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 14))
    for ax, (label, path) in zip(axes.flat, canvas_paths.items()):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(label, fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')

    fig.suptitle('Canvas Examples from Four Benchmarks', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'fig2_canvas_examples.png')
    fig.savefig(OUT / 'fig2_canvas_examples.pdf')
    print(f"  Saved: fig2_canvas_examples.png/pdf")
    plt.close()


# ============================================================
# Fig 3: Main Results Bar Chart
# ============================================================
def generate_fig3():
    print("Generating Fig 3 (Main results)...")
    benchmarks = ['ScienceQA\n(Acc)', 'OK-VQA\n(Acc)', 'HotpotQA\n(EM)', 'MMQA\n(EM)']
    no_ret =     [87.90, 18.77,  3.24, 42.65]
    text_rag =   [89.70, 23.54, 21.84, 42.61]
    memcanvas =  [89.74, 56.92, 50.00, 44.08]
    gpt4o_mc =   [89.40, 49.13, 60.80, 45.20]  # GPT-4o-mini MemCanvas

    x = np.arange(len(benchmarks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, no_ret, width, label='No Retrieval', color=COLORS['no_ret'], edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, text_rag, width, label='Text-RAG', color=COLORS['text_rag'], edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + width, memcanvas, width, label='MemCanvas (Qwen2.5-VL-7B)', color=COLORS['memcanvas'], edgecolor='white', linewidth=0.8)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Add GPT-4o-mini markers
    for i, val in enumerate(gpt4o_mc):
        ax.plot(x[i] + width, val, marker='*', color=COLORS['gpt4o'], markersize=14, zorder=5)
    ax.plot([], [], marker='*', color=COLORS['gpt4o'], linestyle='None', markersize=10, label='GPT-4o-mini + MemCanvas')

    ax.set_xlabel('Benchmark', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Cross-Benchmark Comparison: No Retrieval vs Text-RAG vs MemCanvas', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))

    fig.tight_layout()
    fig.savefig(OUT / 'fig3_main_results.png')
    fig.savefig(OUT / 'fig3_main_results.pdf')
    print(f"  Saved: fig3_main_results.png/pdf")
    plt.close()


# ============================================================
# Fig 4: Alpha Ablation
# ============================================================
def generate_fig4():
    print("Generating Fig 4 (Alpha ablation)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) ScienceQA
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    overall = [89.74, 88.96, 88.94, 88.82, 88.47]
    natural = [90.28, 89.30, 89.52, 89.30, 89.65]
    social =  [90.55, 89.99, 90.10, 90.21, 89.99]
    language = [88.00, 87.45, 86.82, 86.73, 84.82]

    ax1.plot(alphas, overall, 'o-', color='#333333', linewidth=2.5, markersize=8, label='Overall', zorder=5)
    ax1.plot(alphas, natural, 's--', color='#4CAF50', linewidth=1.5, markersize=6, label='Natural Sci.')
    ax1.plot(alphas, social, '^--', color='#2196F3', linewidth=1.5, markersize=6, label='Social Sci.')
    ax1.plot(alphas, language, 'D--', color='#FF9800', linewidth=1.5, markersize=6, label='Language Sci.')

    ax1.set_xlabel('α (Image Weight)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) ScienceQA: α Ablation', fontsize=13, fontweight='bold')
    ax1.set_xticks(alphas)
    ax1.set_ylim(84, 91.5)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.axvline(x=0.0, color='red', linestyle=':', alpha=0.3)
    ax1.annotate('Best: α=0.00', xy=(0.0, 89.74), xytext=(0.15, 90.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                fontsize=9, color='red', fontweight='bold')

    # (b) MMQA
    alphas_mmqa = [0.0, 0.5, 0.75]
    em_mmqa = [43.67, 43.83, 44.08]
    f1_mmqa = [48.96, 49.21, 49.32]

    ax2.plot(alphas_mmqa, em_mmqa, 'o-', color='#EF5350', linewidth=2.5, markersize=8, label='EM')
    ax2.plot(alphas_mmqa, f1_mmqa, 's-', color='#42A5F5', linewidth=2.5, markersize=8, label='F1')

    for a, e, f in zip(alphas_mmqa, em_mmqa, f1_mmqa):
        ax2.annotate(f'{e:.2f}', (a, e), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='#EF5350')
        ax2.annotate(f'{f:.2f}', (a, f), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#42A5F5')

    ax2.set_xlabel('α (Image Weight)', fontsize=12)
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('(b) MMQA: α Ablation', fontsize=13, fontweight='bold')
    ax2.set_xticks(alphas_mmqa)
    ax2.set_ylim(42, 51)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axvline(x=0.75, color='red', linestyle=':', alpha=0.3)
    ax2.annotate('Best: α=0.75', xy=(0.75, 44.08), xytext=(0.45, 42.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                fontsize=9, color='red', fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_alpha_ablation.png')
    fig.savefig(OUT / 'fig4_alpha_ablation.pdf')
    print(f"  Saved: fig4_alpha_ablation.png/pdf")
    plt.close()


# ============================================================
# Fig 5: Forgetting T&S Ablation (3 subplots)
# ============================================================
def generate_fig5():
    print("Generating Fig 5 (Forgetting ablation)...")
    TOTAL = 12726
    BASELINE_ACC = 88.82

    data = {
        (250, 0): {"surviving": 1301, "acc": 87.10, "qdist": {0: 412, 1: 337, 2: 285, 3: 267}},
        (250, 1): {"surviving": 287,  "acc": 86.02, "qdist": {0: 52, 1: 68, 2: 79, 3: 88}},
        (250, 2): {"surviving": 103,  "acc": 86.04, "qdist": {0: 19, 1: 16, 2: 35, 3: 33}},
        (500, 0): {"surviving": 2172, "acc": 87.50, "qdist": {0: 749, 1: 552, 2: 432, 3: 439}},
        (500, 1): {"surviving": 673,  "acc": 86.80, "qdist": {0: 120, 1: 167, 2: 180, 3: 206}},
        (500, 2): {"surviving": 279,  "acc": 86.25, "qdist": {0: 35, 1: 68, 2: 91, 3: 85}},
        (750, 0): {"surviving": 2823, "acc": 88.00, "qdist": {0: 1034, 1: 699, 2: 619, 3: 471}},
        (750, 1): {"surviving": 1056, "acc": 86.63, "qdist": {0: 199, 1: 268, 2: 292, 3: 297}},
        (750, 2): {"surviving": 478,  "acc": 86.30, "qdist": {0: 70, 1: 124, 2: 140, 3: 144}},
        (1000, 0): {"surviving": 3374, "acc": 88.12, "qdist": {0: 1301, 1: 871, 2: 651, 3: 551}},
        (1000, 1): {"surviving": 1447, "acc": 87.17, "qdist": {0: 287, 1: 386, 2: 383, 3: 391}},
        (1000, 2): {"surviving": 668,  "acc": 86.28, "qdist": {0: 103, 1: 176, 2: 199, 3: 190}},
    }

    T_vals = [250, 500, 750, 1000]
    S_vals = [0, 1, 2]
    colors_S = {0: '#2196F3', 1: '#FF9800', 2: '#F44336'}
    markers_S = {0: 'o', 1: 's', 2: '^'}

    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.3)

    # (a) Surviving memories vs T
    ax1 = fig.add_subplot(gs[0])
    for S in S_vals:
        surv = [data[(T, S)]["surviving"] for T in T_vals]
        rate = [s / TOTAL * 100 for s in surv]
        ax1.plot(T_vals, surv, marker=markers_S[S], color=colors_S[S],
                linewidth=2.5, markersize=8, label=f'S={S}')
        for t, s, r in zip(T_vals, surv, rate):
            ax1.annotate(f'{s}\n({r:.0f}%)', (t, s), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=7.5, color=colors_S[S])

    ax1.axhline(y=TOTAL, color='gray', linestyle='--', alpha=0.4, label=f'Full ({TOTAL})')
    ax1.set_xlabel('Review Interval T')
    ax1.set_ylabel('Surviving Memories')
    ax1.set_title('(a) Memory Bank Size', fontweight='bold')
    ax1.set_xticks(T_vals)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(-200, TOTAL * 1.08)

    # (b) Accuracy vs Surviving
    ax2 = fig.add_subplot(gs[1])
    for S in S_vals:
        surv = [data[(T, S)]["surviving"] for T in T_vals]
        acc = [data[(T, S)]["acc"] for T in T_vals]
        ax2.scatter(surv, acc, marker=markers_S[S], color=colors_S[S],
                   s=80, zorder=5, label=f'S={S}')
        for t, s, a in zip(T_vals, surv, acc):
            ax2.annotate(f'T={t}', (s, a), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=colors_S[S])

    ax2.axhline(y=BASELINE_ACC, color='gray', linestyle='--', alpha=0.4,
               label=f'Baseline ({BASELINE_ACC}%)')
    ax2.set_xlabel('Surviving Memories')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Accuracy vs Memory Size', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(85.5, 89.5)

    # (c) Quality distribution stacked bar
    ax3 = fig.add_subplot(gs[2])
    labels = []
    q_vals = {0: [], 1: [], 2: [], 3: []}
    for T in T_vals:
        for S in S_vals:
            labels.append(f'T={T}\nS={S}')
            d = data[(T, S)]["qdist"]
            for q in range(4):
                q_vals[q].append(d[q])

    x = np.arange(len(labels))
    width = 0.65
    colors_q = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800']
    q_labels = ['q0 (1.0×)', 'q1 (0.75×)', 'q2 (0.5×)', 'q3 (0.25×)']

    bottom = np.zeros(len(labels))
    for q in range(4):
        ax3.bar(x, q_vals[q], width, bottom=bottom, color=colors_q[q],
               label=q_labels[q], edgecolor='white', linewidth=0.3)
        bottom += np.array(q_vals[q])

    # Total count on top
    for i, b in enumerate(bottom):
        if b > 0:
            ax3.text(i, b + 30, f'{int(b)}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Memories')
    ax3.set_title('(c) Quality Distribution', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.legend(loc='upper right', fontsize=8)

    fig.savefig(OUT / 'fig5_forgetting.png')
    fig.savefig(OUT / 'fig5_forgetting.pdf')
    print(f"  Saved: fig5_forgetting.png/pdf")
    plt.close()


# ============================================================
# Fig 6: Text Compression Cross-Benchmark
# ============================================================
def generate_fig6():
    print("Generating Fig 6 (Compression)...")
    benchmarks = ['HotpotQA\n(EM)', 'LoCoMo\n(F1)', 'ScienceQA\n(Acc)']
    original = [44.1, 32.59, 89.4]
    light =    [49.6, 30.68, 77.6]
    heavy =    [50.0, 23.75, 75.8]

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x - width, original, width, label='Original', color='#78909C', edgecolor='white')
    bars2 = ax.bar(x, light, width, label='Light Compression', color='#42A5F5', edgecolor='white')
    bars3 = ax.bar(x + width, heavy, width, label='Heavy Compression', color='#EF5350', edgecolor='white')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate effect direction
    ax.annotate('Compression\nHELPS', xy=(0, 52), fontsize=11, color='#4CAF50',
               ha='center', fontweight='bold')
    ax.annotate('Compression\nHURTS', xy=(1.5, 52), fontsize=11, color='#F44336',
               ha='center', fontweight='bold')

    # Divider line
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Text Compression: Task-Dependent Effectiveness', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(OUT / 'fig6_compression.png')
    fig.savefig(OUT / 'fig6_compression.pdf')
    print(f"  Saved: fig6_compression.png/pdf")
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT}")

    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()

    print("\n=== All figures generated! ===")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}: {f.stat().st_size // 1024}KB")
