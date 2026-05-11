#!/usr/bin/env python3
"""Generate all main figures for the MemCanvas paper.

Figure 1 (Teaser): AI-generated via Gemini - user-agent-canvas interaction
Figure 2 (System Overview): AI-generated via Gemini - 3-module pipeline
Figure 3 (Qualitative Examples): Programmatic from real eval data
"""

import os
import json
import base64
import pickle
from pathlib import Path
from openai import OpenAI

OUT = Path("/home/cyf/codex/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# OpenRouter client for Gemini image generation
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
MODEL = "google/gemini-3.1-flash-image-preview"


def generate_image(prompt: str, save_path: str):
    """Call Gemini to generate an image and save it."""
    print(f"  Generating: {save_path}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"modalities": ["image", "text"]},
    )
    msg = response.choices[0].message
    if hasattr(msg, 'images') and msg.images:
        for i, image in enumerate(msg.images):
            url = image['image_url']['url']
            if url.startswith('data:'):
                # Base64 data URL
                header, b64data = url.split(',', 1)
                img_bytes = base64.b64decode(b64data)
                suffix = f"_{i}" if i > 0 else ""
                out_path = save_path.replace('.png', f'{suffix}.png')
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"    Saved: {out_path} ({len(img_bytes)//1024}KB)")
                return out_path
    # Check content parts for image
    if hasattr(msg, 'content') and msg.content:
        if isinstance(msg.content, list):
            for part in msg.content:
                if hasattr(part, 'type') and part.type == 'image_url':
                    url = part.image_url.url
                    if url.startswith('data:'):
                        header, b64data = url.split(',', 1)
                        img_bytes = base64.b64decode(b64data)
                        with open(save_path, 'wb') as f:
                            f.write(img_bytes)
                        print(f"    Saved: {save_path} ({len(img_bytes)//1024}KB)")
                        return save_path
        elif isinstance(msg.content, str) and 'base64' in msg.content[:100].lower():
            print(f"    Response contains base64 text, attempting extraction...")
    print(f"    Warning: No image in response. Text: {str(msg.content)[:200]}")
    # Save the text response for debugging
    with open(save_path.replace('.png', '_debug.txt'), 'w') as f:
        f.write(str(msg.content) if msg.content else "No content")
    return None


# ============================================================
# Figure 1: Teaser — User-Agent Canvas Interaction
# ============================================================
def gen_fig1_teaser():
    print("\n=== Figure 1: Teaser ===")

    prompts = [
        # V1: Clean conceptual diagram
        """Generate a clean, professional academic paper figure illustrating the concept of "MemCanvas" — a visual memory system for AI agents.

The image should be a DIAGRAM (not a photograph) in flat illustration / vector style with these elements arranged LEFT to RIGHT:

LEFT SECTION — "User Query":
- A simple human icon (flat design, blue) with a speech bubble saying "When did Arthur's Magazine start?"
- Below: a small document icon representing the conversation history

CENTER SECTION — "MLLM Agent" (a rounded rectangle box):
- Inside: a brain/AI icon with text "Multimodal LLM"
- Below: text "CLIP Retrieval" with a magnifying glass icon
- An arrow from the user query enters this box

RIGHT SECTION — "Visual Memory Bank":
- A 3×3 grid of small thumbnail images representing stored "canvases" (memory pages)
- Each thumbnail should look like a small white card with tiny text lines and occasional small image placeholders
- One canvas in the grid is HIGHLIGHTED with a blue glow/border (the retrieved memory)
- An arrow from the agent box points to this grid with label "retrieve"

BOTTOM RIGHT — "Retrieved Canvas" (enlarged):
- A larger version of the highlighted canvas showing structured content:
  - A header "[HotpotQA] comparison"
  - Two text paragraphs (representing multi-document knowledge)
  - A question-answer section at bottom
- This canvas should have a white background with clean black text, looking like a rendered document page

BOTTOM — The agent produces an answer speech bubble: "Arthur's Magazine (1844)"

STYLE: White background, clean flat design, blue (#4285F4) and gray (#5F6368) color scheme, thin lines, rounded rectangles, sans-serif font labels, suitable for an academic paper. NO photorealism, NO 3D effects. The overall layout is LANDSCAPE (wider than tall). Resolution should be high.""",

        # V2: More emphasis on the memory retrieval flow
        """Create a professional academic diagram showing how "MemCanvas" visual memory works for an AI assistant.

LAYOUT: Horizontal flow diagram, left to right, on white background.

STEP 1 (far left): A user icon asks "What hairstyle does the blond woman have?" — shown as a speech bubble.

STEP 2 (left-center): A box labeled "CLIP Text Encoder" processes the query into a vector (shown as a small colored bar/embedding icon).

STEP 3 (center): A "Memory Bank" shown as a bookshelf or grid containing 12-16 small rectangular thumbnails. Each thumbnail is a miniature "canvas" — a white card with tiny text lines and a small image placeholder. The thumbnails should look like small structured document pages.

STEP 4 (right-center): One canvas is pulled out from the bank (highlighted with a colored border). An arrow labeled "Top-K Retrieval" connects the query embedding to this canvas. The canvas shows: a small photo placeholder on top, text "Q: What is the hairstyle of the blond called?" and "A: pony tail" and "Caption: two women on a tennis court".

STEP 5 (far right): The selected canvas feeds into a box labeled "Qwen2.5-VL-7B (MLLM)" which outputs "Answer: pony tail".

STYLE: Flat vector illustration, academic paper quality. Colors: primarily blue (#4285F4), gray (#757575), with orange (#FF9800) accent for the highlighted canvas. Clean thin arrows, rounded rectangle boxes, sans-serif labels. Landscape orientation. NO photorealistic elements.""",

        # V3: Comparison-focused (text memory vs visual memory)
        """Create an academic paper teaser figure comparing traditional text-only memory with MemCanvas visual memory for AI agents.

The figure has TWO ROWS on a white background:

TOP ROW — "Traditional Text Memory" (grayed out, less prominent):
- User asks a question → Agent searches text database (shown as lines of text in a cylinder/database icon) → Retrieved text snippets (shown as small text blocks) → Agent gives WRONG answer (marked with red X)
- Caption: "Text-only memory loses visual structure"

BOTTOM ROW — "MemCanvas (Ours)" (prominent, highlighted):
- Same user question → Agent searches Visual Memory Bank (shown as a grid of small canvas thumbnail images — each canvas is a white rectangular card with tiny text + small image/table placeholders) → One canvas is retrieved and enlarged, showing structured content: a header, an embedded image, organized text sections, a Q&A section at bottom → Agent gives CORRECT answer (marked with green checkmark)
- Caption: "Visual canvas preserves multimodal structure"

Between the two rows, there's a clear dividing line.

The visual memory bank canvases should look like miniature rendered document pages (white background, black text, occasional small image placeholders, like real MemCanvas outputs).

STYLE: Clean academic flat illustration. White background. Blue (#4285F4) for MemCanvas elements, gray (#9E9E9E) for traditional approach. Thin lines, rounded boxes, sans-serif labels. Landscape format, suitable for a conference paper figure.""",
    ]

    for i, prompt in enumerate(prompts, 1):
        path = str(OUT / f"fig1_teaser_v{i}.png")
        generate_image(prompt, path)


# ============================================================
# Figure 2: System Overview — 3-Module Pipeline
# ============================================================
def gen_fig2_overview():
    print("\n=== Figure 2: System Overview ===")

    prompts = [
        # V1: Detailed 3-panel pipeline
        """Create a detailed academic paper system overview diagram for "MemCanvas" — a visual memory framework. The diagram should have THREE clearly labeled panels arranged LEFT to RIGHT, with arrows connecting them.

PANEL 1 — "Memory Construction" (left, blue header):
Box 1: "Raw Input" — icon showing mixed content (text paragraph + photo + data table)
Arrow down to Box 2: "Text Compressor" — labeled "(RL-trained)" with a compress/shrink icon
Arrow down to Box 3: "Flexible Matrix" — showing 3 block types:
  - "TEXT" block (flexible width, blue)
  - "IMAGE" block (fixed ratio, green)
  - "TABLE" block (flexible columns, orange)
Arrow down to Box 4: "SmartCanvas Layout" — showing a small canvas being assembled from blocks
Arrow down to Output: A rendered canvas image (white rectangle with text lines and a small image, looking like a real document page)

PANEL 2 — "Storage & Retrieval" (center, green header):
Top: The canvas from Panel 1 enters a "CLIP ViT-L/14" encoder
Two arrows emerge: "Visual Embedding (768-d)" and "Text Embedding (768-d)"
Both flow into a "Vector Index (FAISS)" cylinder/database
Below: A "Query" box enters a "CLIP Text Encoder"
Arrow to "Hybrid Similarity": formula "α·cos(img) + (1-α)·cos(txt)"
Arrow to "Top-K Canvases" (2-3 small canvas thumbnails retrieved)
Arrow to "MLLM" box → "Answer"

PANEL 3 — "Update & Forgetting" (right, orange header):
A vertical progression showing 4 stages of a canvas at decreasing sizes:
  Stage 1: Full size canvas (labeled "1.0×, Full Quality")
  Stage 2: Slightly smaller (labeled "0.75×")
  Stage 3: Smaller (labeled "0.5×")
  Stage 4: Tiny (labeled "0.25×")
  Stage 5: X mark (labeled "Delete")
Side annotation: "Frequency Counter" with a bar chart icon showing access frequencies
Arrow: "Low frequency → Progressive degradation"
Arrow: "High frequency → Retain at full quality"
Bottom: "Compact Memory Bank" showing a smaller grid of canvases

STYLE: White background, clean academic diagram. Each panel has a colored header bar (Panel 1: blue #4285F4, Panel 2: green #34A853, Panel 3: orange #FBBC04). Rounded rectangles for components, thin gray arrows for data flow, sans-serif labels. Landscape orientation (width ≈ 1.5× height). Professional quality suitable for ACM SIGGRAPH paper.""",

        # V2: Simpler, more elegant
        """Design a clean system overview figure for an academic paper. The system is called "MemCanvas" — it converts multimodal data into visual canvas images for AI memory.

THREE SECTIONS left to right, connected by arrows:

SECTION 1 — MEMORY CONSTRUCTION:
Shows a pipeline: [Multimodal Input: text + images + tables] → [Text Compression] → [Block Layout (SmartCanvas)] → [Rendered Canvas Image]
The canvas image at the end should look like a white card with organized text, a small image, and structured sections.

SECTION 2 — STORAGE & RETRIEVAL:
Shows: [Canvas] → [CLIP Dual Encoder] → [Vector Database]
Then: [New Query] → [CLIP Text] → [Cosine Similarity Search] → [Top-K Retrieved Canvases] → [Vision-Language Model] → [Answer]

SECTION 3 — MEMORY FORGETTING:
Shows a memory bank with canvases at different resolution levels (full → 3/4 → 1/2 → 1/4 → deleted), with a frequency indicator showing that rarely-used memories get progressively downscaled.

OVERALL STYLE: Minimalist academic diagram on white background. Use blue, green, and orange to distinguish the three sections. Thin arrows, clean boxes with rounded corners, small clear labels. Landscape layout. No decoration — pure information design for a top-tier venue paper.""",

        # V3: Flow-chart style
        """Generate an academic paper figure showing the complete pipeline of "MemCanvas", a visual memory system for multimodal AI.

The figure is a HORIZONTAL FLOW CHART with three color-coded stages:

BLUE STAGE — "① Memory Construction":
Input node: "Historical Conversations (text, images, tables)"
→ Process node: "QA-Reward Text Compressor" (with subtitle "RL-trained")
→ Process node: "Flexible Matrix Representation" (with small icons showing TEXT/IMAGE/TABLE block types)
→ Process node: "SmartCanvas Adaptive Layout" (with subtitle "8 strategies")
→ Output node: A small rendered canvas preview (white rectangle with text + image content)

GREEN STAGE — "② Storage & Retrieval":
→ Process node: "CLIP ViT-L/14 Dual Encoding" (subtitle "visual + textual embeddings")
→ Database node: "FAISS Vector Index"
← Input from bottom: "User Query" → "CLIP Text Encoder" → "Hybrid Similarity Search"
→ Output: "Top-K Canvases" (2-3 small canvas thumbnails)
→ Process node: "Multimodal LLM (Qwen2.5-VL)"
→ Final output: "Answer"

ORANGE STAGE — "③ Update & Forgetting" (shown as a side branch from the database):
→ "Access Frequency Monitor"
→ "Progressive Resolution Degradation" showing 4 resolution levels (1.0× → 0.75× → 0.5× → 0.25×)
→ "Memory Eviction" (lowest frequency entries deleted)

STYLE: Professional academic diagram. White background. Color-coded stages (blue #4285F4, green #0F9D58, orange #F4B400). Rounded rectangle nodes, thin directional arrows, small clear sans-serif labels. Landscape orientation, width about 1.8× height. Suitable for ACM Transactions on Graphics.""",
    ]

    for i, prompt in enumerate(prompts, 1):
        path = str(OUT / f"fig2_overview_v{i}.png")
        generate_image(prompt, path)


# ============================================================
# Figure 3: Qualitative Examples (Programmatic)
# ============================================================
def gen_fig3_qualitative():
    """Create Figure 3 from real benchmark evaluation data."""
    print("\n=== Figure 3: Qualitative Examples ===")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import textwrap

    # --- Load all question data ---
    print("  Loading question data...")
    # OK-VQA
    with open('/home/cyf/codex/okvqa_data/okvqa_cached.pkl', 'rb') as f:
        okvqa_test = pickle.load(f)['test']

    # MMQA
    with open('/home/cyf/codex/mmqa_data/mmqa_parsed.pkl', 'rb') as f:
        mmqa_all = pickle.load(f)
    mmqa_dev = mmqa_all['dev']

    # HotpotQA
    with open('/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl', 'rb') as f:
        hotpotqa_dev = pickle.load(f)['dev']

    # ScienceQA
    with open('/home/cyf/LLaMA-Factory-main/data/scienceqa_sft_val.json') as f:
        scienceqa_val = json.load(f)

    def get_scienceqa_question(idx):
        if idx >= len(scienceqa_val):
            return ""
        msg = scienceqa_val[idx]['messages']
        for m in msg:
            if m['role'] == 'user':
                content = m['content'] if isinstance(m['content'], str) else str(m['content'])
                for line in content.split('\n'):
                    if line.strip().startswith('Question:'):
                        return line.strip().replace('Question: ', '')
                # Fallback: extract between "Question:" and "Choices:"
                import re
                match = re.search(r'Question:\s*(.+?)(?:\n|Choices:)', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
        return ""

    # --- Benchmark configs ---
    benchmarks = [
        ('ScienceQA', {
            'checkpoint': '/home/cyf/codex/memcanvas0413_eval/scienceqa_alpha0.00/checkpoint.json',
            'canvas_dir': '/home/cyf/codex/scienceqa_smart_canvases',
            'correct_key': 'correct',
            'get_question': lambda sid: get_scienceqa_question(int(sid)),
            'get_test_img': lambda sid: None,  # Image embedded in canvas
        }),
        ('OK-VQA', {
            'checkpoint': '/home/cyf/codex/memcanvas0413_eval/okvqa_alpha0.75/checkpoint.json',
            'canvas_dir': '/home/cyf/codex/okvqa_data/canvases_smart',
            'correct_key': 'correct',
            'get_question': lambda sid: okvqa_test[int(sid)]['question'] if int(sid) < len(okvqa_test) else "",
            'get_test_img': lambda sid: f"/home/cyf/codex/okvqa_data/images/test/{int(sid):05d}.jpg",
        }),
        ('MMQA', {
            'checkpoint': '/home/cyf/codex/memcanvas0413_eval/mmqa_alpha0.75/checkpoint.json',
            'canvas_dir': '/home/cyf/codex/mmqa_data/canvases_smart',
            'correct_key': 'em',
            'get_question': lambda sid: mmqa_dev[int(sid)]['question'] if int(sid) < len(mmqa_dev) else "",
            'get_test_img': lambda sid: None,  # Tables/text, no single test image
        }),
        ('HotpotQA', {
            'checkpoint': '/home/cyf/codex/memcanvas0413_eval/hotpotqa_alpha0.75/checkpoint.json',
            'canvas_dir': '/home/cyf/codex/hotpotqa_data/canvases_smart',
            'correct_key': 'em',
            'get_question': lambda sid: hotpotqa_dev[int(sid)]['question'] if int(sid) < len(hotpotqa_dev) else "",
            'get_test_img': lambda sid: None,  # Text-only
        }),
    ]

    # --- Find good examples ---
    selected = []
    for bm_name, bm_info in benchmarks:
        cp_path = bm_info['checkpoint']
        if not os.path.exists(cp_path):
            print(f"  Skipping {bm_name}: checkpoint not found")
            continue

        with open(cp_path) as f:
            cp = json.load(f)

        canvas_dir = bm_info['canvas_dir']
        correct_key = bm_info['correct_key']
        found = False

        # Try to find a visually interesting correct example
        for sample_id, result in cp.items():
            if result.get(correct_key, 0) < 1.0:
                continue
            canvas_path = os.path.join(canvas_dir, f"{int(sample_id):05d}.png")
            if not os.path.exists(canvas_path):
                continue
            canvas_size = os.path.getsize(canvas_path)
            if canvas_size < 8000:  # Prefer larger (image-containing) canvases
                continue

            test_img_path = bm_info['get_test_img'](sample_id)
            if test_img_path and not os.path.exists(test_img_path):
                continue

            question = bm_info['get_question'](sample_id)
            gt = result['gt']
            pred = result['pred']
            gt_str = gt[0] if isinstance(gt, list) else gt

            selected.append({
                'bm_name': bm_name,
                'sample_id': sample_id,
                'canvas_path': canvas_path,
                'test_img_path': test_img_path,
                'question': question,
                'gt': gt_str,
                'pred': pred,
            })
            print(f"  {bm_name}: id={sample_id}, Q={question[:60]}..., GT={gt_str[:30]}, Pred={pred[:30]}")
            found = True
            break

        if not found:
            print(f"  {bm_name}: no suitable example found, using first correct")
            for sample_id, result in cp.items():
                if result.get(correct_key, 0) >= 1.0:
                    canvas_path = os.path.join(canvas_dir, f"{int(sample_id):05d}.png")
                    if os.path.exists(canvas_path):
                        question = bm_info['get_question'](sample_id)
                        gt = result['gt']
                        gt_str = gt[0] if isinstance(gt, list) else gt
                        selected.append({
                            'bm_name': bm_name,
                            'sample_id': sample_id,
                            'canvas_path': canvas_path,
                            'test_img_path': bm_info['get_test_img'](sample_id),
                            'question': question,
                            'gt': gt_str,
                            'pred': result['pred'],
                        })
                        break

    if not selected:
        print("  ERROR: No examples found!")
        return

    # --- Render the figure (2-column: Canvas | Q&A) ---
    n_rows = len(selected)
    fig_width = 14
    row_height = 3.8
    fig_height = n_rows * row_height + 0.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1.0, 1.2],
                          hspace=0.30, wspace=0.05,
                          left=0.10, right=0.98, top=0.96, bottom=0.01)

    bm_colors = {
        'ScienceQA': '#4285F4',
        'OK-VQA': '#34A853',
        'MMQA': '#E8710A',
        'HotpotQA': '#EA4335',
    }

    for row_idx, info in enumerate(selected):
        bm_name = info['bm_name']
        color = bm_colors.get(bm_name, '#666')

        # --- Col 1: Retrieved canvas ---
        ax1 = fig.add_subplot(gs[row_idx, 0])
        canvas_img = Image.open(info['canvas_path']).convert('RGB')
        ax1.imshow(np.array(canvas_img))
        ax1.axis('off')
        # Add thin colored border
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)
        if row_idx == 0:
            ax1.set_title('Retrieved Memory Canvas', fontsize=12, fontweight='bold', pad=8)

        # Row label on left
        ax1.text(-0.15, 0.5, f"({chr(97+row_idx)}) {bm_name}", transform=ax1.transAxes,
                 fontsize=13, fontweight='bold', color=color,
                 ha='center', va='center', rotation=90)

        # --- Col 2: Question & Answer panel ---
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.axis('off')

        q_wrapped = textwrap.fill(info['question'], width=48) if info['question'] else "(question from benchmark)"
        gt_str = str(info['gt'])
        pred_str = str(info['pred'])

        # Build formatted text
        lines = []
        lines.append(f"Question:")
        lines.append(f"  {q_wrapped}")
        lines.append(f"")
        lines.append(f"Ground Truth:  {gt_str}")
        lines.append(f"Prediction:    {pred_str}")
        lines.append(f"")
        match = "CORRECT" if gt_str.lower().strip() == pred_str.lower().strip() else "CORRECT (semantic match)"
        lines.append(f"Result:  {match}")

        qa_text = "\n".join(lines)

        ax2.text(0.03, 0.95, qa_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 linespacing=1.5,
                 bbox=dict(boxstyle='round,pad=0.7', facecolor='#FAFAFA',
                          edgecolor=color, linewidth=2.0, alpha=0.95))
        if row_idx == 0:
            ax2.set_title('Question & Answer', fontsize=12, fontweight='bold', pad=8)

    plt.savefig(str(OUT / "fig3_qualitative.png"), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(str(OUT / "fig3_qualitative.pdf"), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {OUT / 'fig3_qualitative.png'}")
    print(f"  Saved: {OUT / 'fig3_qualitative.pdf'}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        which = sys.argv[1]
        if which == "fig1":
            gen_fig1_teaser()
        elif which == "fig2":
            gen_fig2_overview()
        elif which == "fig3":
            gen_fig3_qualitative()
        else:
            print(f"Unknown figure: {which}. Use fig1, fig2, or fig3.")
    else:
        gen_fig1_teaser()
        gen_fig2_overview()
        gen_fig3_qualitative()

    print("\nDone! All figures saved to:", OUT)
