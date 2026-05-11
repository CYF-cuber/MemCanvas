#!/usr/bin/env python3
"""
Generate layout candidates for MMQA samples with image+table+text.
Produces multiple candidate layouts per sample for visual comparison.
"""
import io, pickle, sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table,
    layout_single_column, layout_two_column, layout_side_by_side,
    render_layout, ContentBlock, BlockType, PADDING, GAP,
    reflow_text_height,
)

# Canvas0415 params (larger fonts)
FONT_HEADER = 14
FONT_BODY = 16
FONT_QA = 18
FONT_TABLE = 14
REF_WIDTH = 800
IMG_MAX_DIM = 800
IMG_MIN_DIM = 200   # minimum image dimension (user request)
TABLE_MIN_WIDTH = 300  # minimum table width

OUT_DIR = Path("/home/cyf/codex/layout_candidates")
OUT_DIR.mkdir(exist_ok=True)


def find_image_table_samples(data, n=5):
    """Find MMQA samples with both image and table in supporting_context."""
    results = []
    for i, s in enumerate(data["train"]):
        ctx_types = set()
        for ctx in s.get("supporting_context", []):
            ctx_types.add(ctx["doc_part"])
        if "image" in ctx_types and "table" in ctx_types:
            results.append(i)
            if len(results) >= n:
                break
    return results


def build_blocks(s, data):
    """Build content blocks for an MMQA sample."""
    tables = data["tables"]
    texts = data["texts"]
    images_meta = data["images"]
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    blocks = []

    # Header
    modalities = s.get("metadata", {}).get("modalities", [])
    qtype = s.get("metadata", {}).get("type", "")
    if qtype or modalities:
        header = f"[{qtype}] Modalities: {', '.join(modalities)}"
        blocks.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))

    # Supporting context
    for ctx in s.get("supporting_context", [])[:3]:
        doc_id = ctx["doc_id"]
        doc_part = ctx["doc_part"]

        if doc_part == "text" and doc_id in texts:
            text_doc = texts[doc_id]
            title = text_doc.get("title", "")
            passage = text_doc.get("text", "")[:300]
            blocks.append(measure_text(f"[Text] {title}\n{passage}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        elif doc_part == "table" and doc_id in tables:
            table_doc = tables[doc_id]
            title = table_doc.get("title", "")
            if title:
                blocks.append(measure_text(f"[Table] {title}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
            headers = [h["column_name"] for h in table_doc["table"]["header"]]
            rows = [[cell["text"][:40] for cell in row] for row in table_doc["table"]["table_rows"][:8]]
            table_data = [headers] + rows
            if len(table_doc["table"]["table_rows"]) > 8:
                table_data.append([f"...({len(table_doc['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
            blocks.append(measure_table(table_data, font_size=FONT_TABLE))

        elif doc_part == "image" and doc_id in images_meta:
            img_info = images_meta[doc_id]
            img_path = IMAGES_DIR / img_info["path"]
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    title = img_info.get("title", "")
                    blocks.append(measure_text(f"[Image] {title}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                    blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))
                except Exception:
                    pass

    # Question + answer
    q = s.get("question", "")
    answers = s.get("answers", [])
    ans = str(answers[0]["answer"]) if answers else ""
    blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

    return blocks


def generate_candidates(blocks):
    """Generate multiple layout candidates with different strategies and widths."""
    # Estimate total area
    total_area = sum(b.area for b in blocks) * 1.3
    target = max(350, min(1200, int(np.sqrt(total_area))))

    candidates = []

    # Strategy 1: Single column at various widths
    for w in [600, 700, 830, 1000, 1200]:
        try:
            layout = layout_single_column(blocks, target_width=w)
            candidates.append((f"Single w={w}", layout))
        except Exception:
            pass

    # Strategy 2: Two column at various widths
    for w in [800, 1000, 1200]:
        try:
            layout = layout_two_column(blocks, target_width=w)
            candidates.append((f"TwoCol w={w}", layout))
        except Exception:
            pass

    # Strategy 3: Side-by-side at various widths
    for w in [800, 1000, 1200]:
        try:
            layout = layout_side_by_side(blocks, target_width=w)
            candidates.append((f"SideBySide w={w}", layout))
        except Exception:
            pass

    # Strategy 4: Auto (choose_best_layout)
    from smart_canvas_layout import choose_best_layout
    try:
        layout = choose_best_layout(blocks)
        candidates.append(("AutoBest", layout))
    except Exception:
        pass

    return candidates


def annotate_layout(img, name, layout):
    """Add layout info text at top of image."""
    from PIL import ImageDraw
    from smart_canvas_layout import get_font

    sq = layout.squareness
    util = layout.utilization
    score = 0.6 * sq + 0.3 * util + 0.1 * (1.0 if 300 <= layout.width <= 1200 and 300 <= layout.height <= 1200 else 0.7)

    # Create new image with info bar
    bar_h = 30
    new_img = Image.new("RGB", (img.width, img.height + bar_h), (240, 240, 240))
    new_img.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(new_img)
    try:
        font = get_font(12)
    except Exception:
        font = None
    info = f"{name} | {layout.width}x{layout.height} | sq={sq:.2f} util={util:.2f} score={score:.2f}"
    draw.text((5, 5), info, fill=(0, 0, 0), font=font)
    return new_img


def main():
    print("Loading MMQA data...")
    cache_file = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    print("Finding image+table samples...")
    indices = find_image_table_samples(data, n=3)
    print(f"  Found {len(indices)} samples: {indices}")

    for idx in indices:
        s = data["train"][idx]
        q = s.get("question", "")[:80]
        print(f"\n--- Sample {idx}: {q} ---")

        blocks = build_blocks(s, data)
        print(f"  Blocks: {len(blocks)} ({', '.join(b.type.name for b in blocks)})")

        candidates = generate_candidates(blocks)
        print(f"  Generated {len(candidates)} candidates")

        sample_dir = OUT_DIR / f"sample_{idx}"
        sample_dir.mkdir(exist_ok=True)

        for j, (name, layout) in enumerate(candidates):
            img = render_layout(layout)
            img = annotate_layout(img, name, layout)
            path = sample_dir / f"{j:02d}_{name.replace(' ', '_').replace('=', '')}.png"
            img.save(path)
            print(f"  [{j}] {name}: {layout.width}x{layout.height} sq={layout.squareness:.2f} util={layout.utilization:.2f}")

    print(f"\nAll candidates saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
