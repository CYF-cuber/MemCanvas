#!/usr/bin/env python3
"""
Generate before/after layout comparison for diverse ScienceQA samples.
Saves to /home/cyf/codex/paper_canvas_examples/layout_comparison/
"""

import json
import os
import sys

sys.path.insert(0, "/home/cyf/codex")
sys.path.insert(0, "/home/cyf/memory")

from PIL import Image

from smart_canvas_layout import (
    choose_best_layout,
    measure_image,
    measure_table,
    measure_text,
    render_baseline,
    render_layout,
    get_font,
    Layout,
    ContentBlock,
    BlockType,
)

SCIQA_DIR = "/home/cyf/memory/ScienceQA/data/scienceqa"
OUTPUT_DIR = "/home/cyf/codex/paper_canvas_examples/layout_comparison"


def load_problems():
    with open(os.path.join(SCIQA_DIR, "problems.json")) as f:
        return json.load(f)


def get_image(pid: str, split: str):
    img_path = os.path.join(SCIQA_DIR, "images", split, pid, "image.png")
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


def problem_to_blocks(p: dict, img=None):
    """Convert a ScienceQA problem dict into content blocks."""
    blocks = []

    # Header: subject / topic
    subj = p.get("subject", "")
    topic = p.get("topic", "")
    if subj or topic:
        blocks.append(measure_text(f"[{subj} / {topic}]", font_size=14))

    # Hint / context
    hint = p.get("hint", "")
    if hint:
        blocks.append(measure_text(f"Context: {hint}", font_size=16))

    # Image
    if img is not None:
        blocks.append(measure_image(img, max_dim=500))

    # Question + choices
    q_text = f"Q: {p['question']}\n"
    choices = p.get("choices", [])
    answer = p.get("answer", -1)
    for i, c in enumerate(choices):
        letter = chr(65 + i)
        marker = "✓" if i == answer else "○"
        q_text += f"  {marker} {letter}. {c}\n"
    blocks.append(measure_text(q_text.strip(), font_size=18))

    # Lecture (background knowledge) — may contain tables
    lecture = p.get("lecture", "")
    if lecture:
        if "|" in lecture and lecture.count("|") >= 4:
            table_lines = [l.strip() for l in lecture.split("\n") if "|" in l]
            table_data = []
            for line in table_lines:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells and not all(set(c) <= {"-", " "} for c in cells):
                    table_data.append(cells)
            if table_data:
                non_table = "\n".join(
                    l for l in lecture.split("\n") if "|" not in l
                ).strip()
                if non_table:
                    blocks.append(
                        measure_text(f"Knowledge: {non_table}", font_size=15)
                    )
                blocks.append(measure_table(table_data, font_size=15))
            else:
                blocks.append(
                    measure_text(f"Knowledge: {lecture[:500]}", font_size=15)
                )
        else:
            blocks.append(measure_text(f"Knowledge: {lecture[:500]}", font_size=15))

    # Solution
    solution = p.get("solution", "")
    if solution:
        blocks.append(measure_text(f"Solution: {solution[:300]}", font_size=15))

    return blocks


def create_comparison(pid, p, img, output_dir):
    blocks = problem_to_blocks(p, img)

    # Baseline (old DynamicCanvas style)
    baseline = render_baseline(blocks)

    # Smart layout
    best = choose_best_layout(blocks)
    smart = render_layout(best)

    # Print stats
    bl_pixels = baseline.width * baseline.height
    sm_pixels = smart.width * smart.height
    ratio = sm_pixels / bl_pixels if bl_pixels > 0 else 0

    subj = p.get("subject", "")
    topic = p.get("topic", "")
    has_img = img is not None
    has_tbl = any(b.type == BlockType.TABLE for b in blocks)

    print(f"\n  pid={pid}: {subj}/{topic}")
    print(f"  Content: {len(blocks)} blocks, img={has_img}, table={has_tbl}")
    print(
        f"  Baseline: {baseline.width}x{baseline.height} = {bl_pixels:,}px"
    )
    print(
        f"  Smart:    {smart.width}x{smart.height} = {sm_pixels:,}px "
        f"({best.name}, sq={best.squareness:.2f}, util={best.utilization:.2f})"
    )
    print(f"  Pixel ratio: {ratio:.2f}x (smart/baseline)")

    # Save individual images
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{pid}_{subj.replace(' ', '_')}_{topic.replace(' ', '_')}"

    baseline.save(os.path.join(output_dir, f"baseline_{tag}.png"))
    smart.save(os.path.join(output_dir, f"smart_{tag}.png"))

    # Side-by-side comparison
    margin = 20
    label_h = 35
    total_w = baseline.width + smart.width + 3 * margin
    total_h = max(baseline.height, smart.height) + 2 * margin + label_h

    comp = Image.new("RGB", (total_w, total_h), (235, 235, 235))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(comp)
    font = get_font(16)
    font_sm = get_font(12)

    # Left: baseline
    lx = margin
    draw.text(
        (lx, 8),
        f"BEFORE: DynamicCanvas ({baseline.width}x{baseline.height})",
        font=font,
        fill=(200, 0, 0),
    )
    comp.paste(baseline, (lx, label_h))
    draw.rectangle(
        [lx - 1, label_h - 1, lx + baseline.width, label_h + baseline.height],
        outline=(200, 0, 0),
        width=2,
    )

    # Right: smart
    rx = margin + baseline.width + margin
    sq_pct = best.squareness * 100
    util_pct = best.utilization * 100
    draw.text(
        (rx, 8),
        f"AFTER: {best.name} ({best.width}x{best.height}, sq={sq_pct:.0f}%, util={util_pct:.0f}%)",
        font=font,
        fill=(0, 130, 0),
    )
    comp.paste(smart, (rx, label_h))
    draw.rectangle(
        [rx - 1, label_h - 1, rx + smart.width, label_h + smart.height],
        outline=(0, 130, 0),
        width=2,
    )

    # Bottom info
    info = f"pid={pid} | {subj}/{topic} | {len(blocks)} blocks | img={has_img} | table={has_tbl}"
    draw.text((margin, total_h - 18), info, font=font_sm, fill=(100, 100, 100))

    comp.save(os.path.join(output_dir, f"comparison_{tag}.png"))

    return baseline, smart, best


def main():
    print("=" * 60)
    print("SmartCanvasLayout — Before/After Demo")
    print("=" * 60)

    problems = load_problems()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Categorize train problems
    train_with_img = []
    train_with_table = []
    train_long = []
    train_short = []
    train_no_img = []

    for pid, p in problems.items():
        if p["split"] != "train":
            continue
        lec = p.get("lecture", "")
        sol = p.get("solution", "")
        hint = p.get("hint", "")
        total_text = len(lec) + len(sol) + len(hint)
        has_image = bool(p.get("image"))
        has_table = "|" in lec and lec.count("|") >= 4

        if has_image:
            train_with_img.append((pid, p, total_text, has_table))
        else:
            train_no_img.append((pid, p, total_text, has_table))

        if has_table:
            train_with_table.append((pid, p, total_text, has_image))
        if total_text > 600:
            train_long.append((pid, p, total_text, has_image))
        elif total_text < 80:
            train_short.append((pid, p, total_text, has_image))

    print(f"\nTrain stats:")
    print(f"  With image: {len(train_with_img)}")
    print(f"  With table: {len(train_with_table)}")
    print(f"  Long (>600 chars): {len(train_long)}")
    print(f"  Short (<80 chars): {len(train_short)}")
    print(f"  No image: {len(train_no_img)}")

    # Select 8 diverse samples
    selected = []

    # 1. Image + medium content
    for pid, p, tlen, has_tbl in train_with_img:
        if 200 < tlen < 600 and not has_tbl:
            selected.append(("img_medium", pid, p))
            break

    # 2. Image + long content
    for pid, p, tlen, has_tbl in train_with_img:
        if tlen > 600 and not has_tbl:
            selected.append(("img_long", pid, p))
            break

    # 3. Image + table
    img_table = [(pid, p, tlen) for pid, p, tlen, has_img in train_with_table if has_img]
    if img_table:
        selected.append(("img_table", img_table[0][0], img_table[0][1]))

    # 4. Table only (no image)
    tbl_noimg = [(pid, p, tlen) for pid, p, tlen, has_img in train_with_table if not has_img]
    if tbl_noimg:
        selected.append(("table_only", tbl_noimg[2][0], tbl_noimg[2][1]))

    # 5. Long text only (no image)
    long_noimg = [(pid, p, tlen) for pid, p, tlen, has_img in train_long if not has_img]
    if long_noimg:
        selected.append(("long_text", long_noimg[5][0], long_noimg[5][1]))

    # 6. Short content + image
    short_img = [(pid, p, tlen, ht) for pid, p, tlen, ht in train_with_img if tlen < 120]
    if short_img:
        selected.append(("short_img", short_img[3][0], short_img[3][1]))

    # 7. Short content, no image
    if train_short:
        for pid, p, tlen, has_img in train_short:
            if not has_img:
                selected.append(("short_text", pid, p))
                break

    # 8. Image + very long content
    very_long_img = [(pid, p, tlen, ht) for pid, p, tlen, ht in train_with_img if tlen > 800]
    if very_long_img:
        selected.append(("img_verylong", very_long_img[0][0], very_long_img[0][1]))

    print(f"\nSelected {len(selected)} samples:")

    stats = []
    for label, pid, p in selected:
        print(f"\n{'='*50}")
        print(f"[{label}] pid={pid}")

        img = get_image(pid, p["split"])
        if img:
            print(f"  Image loaded: {img.size}")

        baseline, smart, layout = create_comparison(pid, p, img, OUTPUT_DIR)
        stats.append(
            {
                "label": label,
                "pid": pid,
                "subject": p.get("subject", ""),
                "topic": p.get("topic", ""),
                "baseline_size": f"{baseline.width}x{baseline.height}",
                "smart_size": f"{smart.width}x{smart.height}",
                "layout": layout.name,
                "squareness": round(layout.squareness, 3),
                "utilization": round(layout.utilization, 3),
                "baseline_pixels": baseline.width * baseline.height,
                "smart_pixels": smart.width * smart.height,
            }
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<16} {'Baseline':>12} {'Smart':>12} {'Layout':<14} {'Sq':>5} {'Util':>5} {'Px Ratio':>9}")
    print("-" * 80)
    for s in stats:
        px_ratio = s["smart_pixels"] / s["baseline_pixels"]
        print(
            f"{s['label']:<16} {s['baseline_size']:>12} {s['smart_size']:>12} "
            f"{s['layout']:<14} {s['squareness']:>5.2f} {s['utilization']:>5.2f} {px_ratio:>8.2f}x"
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
