#!/usr/bin/env python3
"""
SmartCanvasLayout — 先量后造的画布布局优化器

核心思想：先预估每个内容块的矩形尺寸，再通过排列优化组成接近方形的画布。

对比 DynamicCanvas（固定 640×640 patch，单列填充）的改进：
1. 动态画布尺寸，按内容量自适应
2. 支持多列/图文并排布局
3. 目标：最小化空白率，纵横比接近 1:1

Usage:
    python smart_canvas_layout.py
"""

import io
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BlockType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class ContentBlock:
    """A measured content block with flexible dimensions."""
    type: BlockType
    data: Any  # str for text, PIL.Image for image, List[List[str]] for table

    # Measured dimensions at reference width
    ref_width: int = 0
    ref_height: int = 0
    area: float = 0.0

    # For images: fixed aspect ratio
    aspect_ratio: float = 1.0  # w/h

    # For text: can reflow
    char_count: int = 0
    font_size: int = 18


@dataclass
class PlacedBlock:
    """A block with its final position and size on the canvas."""
    block: ContentBlock
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class Layout:
    """A complete layout solution."""
    name: str
    width: int
    height: int
    placements: List[PlacedBlock] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def squareness(self) -> float:
        """1.0 = perfect square, lower = more elongated."""
        r = self.aspect_ratio
        return min(r, 1 / r) if r > 0 else 0

    @property
    def total_area(self) -> int:
        return self.width * self.height

    @property
    def content_area(self) -> int:
        return sum(p.width * p.height for p in self.placements)

    @property
    def utilization(self) -> float:
        return self.content_area / max(self.total_area, 1)


# ---------------------------------------------------------------------------
# Font utilities
# ---------------------------------------------------------------------------

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
_font_path: Optional[str] = None

def _find_font() -> Optional[str]:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None

def get_font(size: int) -> ImageFont.FreeTypeFont:
    global _font_path
    if _font_path is None:
        _font_path = _find_font() or ""
    if size not in _font_cache:
        if _font_path:
            _font_cache[size] = ImageFont.truetype(_font_path, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


# ---------------------------------------------------------------------------
# Measure phase: estimate content block sizes
# ---------------------------------------------------------------------------

def measure_text(text: str, font_size: int = 18, ref_width: int = 600) -> ContentBlock:
    """Measure a text block. Text is flexible: given width → compute height."""
    font = get_font(font_size)
    line_height = int(font_size * 1.3)

    # Count how many lines at ref_width
    lines = wrap_text(text, font, ref_width)
    ref_height = len(lines) * line_height

    # Approximate area (stays roughly constant across widths)
    area = ref_width * ref_height

    return ContentBlock(
        type=BlockType.TEXT,
        data=text,
        ref_width=ref_width,
        ref_height=ref_height,
        area=area,
        char_count=len(text),
        font_size=font_size,
    )


def measure_image(img: Image.Image, max_dim: int = 500) -> ContentBlock:
    """Measure an image block. Image is rigid: fixed aspect ratio."""
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return ContentBlock(
        type=BlockType.IMAGE,
        data=img,
        ref_width=new_w,
        ref_height=new_h,
        area=new_w * new_h,
        aspect_ratio=new_w / max(new_h, 1),
    )


def measure_table(data: List[List[str]], font_size: int = 16) -> ContentBlock:
    """Measure a table block."""
    font = get_font(font_size)
    cell_pad = 8
    row_height = int(font_size * 1.3) + 2 * cell_pad

    num_cols = max(len(row) for row in data) if data else 1
    num_rows = len(data)

    # Estimate column widths
    col_widths = [60] * num_cols
    for row in data:
        for i, cell in enumerate(row):
            if i < num_cols:
                bbox = font.getbbox(str(cell))
                cw = (bbox[2] - bbox[0]) + 2 * cell_pad
                col_widths[i] = max(col_widths[i], cw)

    total_w = sum(col_widths) + 4  # border
    total_h = num_rows * row_height + 4

    return ContentBlock(
        type=BlockType.TABLE,
        data=data,
        ref_width=total_w,
        ref_height=total_h,
        area=total_w * total_h,
        font_size=font_size,
    )


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width pixels."""
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        current_line = ''
        current_width = 0
        for char in para:
            bbox = font.getbbox(char)
            char_width = (bbox[2] - bbox[0]) if bbox else 10
            if current_width + char_width <= max_width:
                current_line += char
                current_width += char_width
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
                current_width = char_width
        if current_line:
            lines.append(current_line)
    return lines


def reflow_text_height(block: ContentBlock, target_width: int) -> int:
    """Compute text height at a different width."""
    font = get_font(block.font_size)
    line_height = int(block.font_size * 1.3)
    lines = wrap_text(block.data, font, target_width)
    return len(lines) * line_height


# ---------------------------------------------------------------------------
# Layout strategies
# ---------------------------------------------------------------------------

PADDING = 15
GAP = 10


def layout_single_column(blocks: List[ContentBlock], target_width: int = 0) -> Layout:
    """Single column layout — similar to current DynamicCanvas."""
    if target_width == 0:
        target_width = max(b.ref_width for b in blocks) + 2 * PADDING

    content_w = target_width - 2 * PADDING
    placements = []
    y = PADDING

    for b in blocks:
        if b.type == BlockType.TEXT:
            h = reflow_text_height(b, content_w)
            placements.append(PlacedBlock(b, PADDING, y, content_w, h))
            y += h + GAP
        elif b.type == BlockType.IMAGE:
            # Scale image to fit width
            scale = min(content_w / b.ref_width, 1.0)
            w = int(b.ref_width * scale)
            h = int(b.ref_height * scale)
            x = PADDING + (content_w - w) // 2
            placements.append(PlacedBlock(b, x, y, w, h))
            y += h + GAP
        elif b.type == BlockType.TABLE:
            scale = min(content_w / b.ref_width, 1.0)
            w = int(b.ref_width * scale)
            h = int(b.ref_height * scale)
            placements.append(PlacedBlock(b, PADDING, y, w, h))
            y += h + GAP

    y += PADDING - GAP
    return Layout("single_column", target_width, y, placements)


def layout_two_column(blocks: List[ContentBlock], target_width: int = 0) -> Layout:
    """Two-column layout. Images span full width, text split into columns."""
    if target_width == 0:
        total_area = sum(b.area for b in blocks) * 1.3
        target_width = max(int(math.sqrt(total_area)), 400)

    content_w = target_width - 2 * PADDING
    col_w = (content_w - GAP) // 2

    # Separate blocks: images go full-width, text/tables go into columns
    full_width_blocks = [b for b in blocks if b.type == BlockType.IMAGE]
    column_blocks = [b for b in blocks if b.type != BlockType.IMAGE]

    placements = []
    y = PADDING

    # Full-width images first
    for b in full_width_blocks:
        scale = min(content_w / b.ref_width, 1.0)
        w = int(b.ref_width * scale)
        h = int(b.ref_height * scale)
        x = PADDING + (content_w - w) // 2
        placements.append(PlacedBlock(b, x, y, w, h))
        y += h + GAP

    # Two-column for remaining blocks
    if column_blocks:
        left_blocks = column_blocks[:len(column_blocks) // 2 + 1]
        right_blocks = column_blocks[len(column_blocks) // 2 + 1:]

        left_y = y
        right_y = y

        for b in left_blocks:
            if b.type == BlockType.TEXT:
                h = reflow_text_height(b, col_w)
                placements.append(PlacedBlock(b, PADDING, left_y, col_w, h))
                left_y += h + GAP
            elif b.type == BlockType.TABLE:
                scale = min(col_w / b.ref_width, 1.0)
                w = int(b.ref_width * scale)
                h = int(b.ref_height * scale)
                placements.append(PlacedBlock(b, PADDING, left_y, w, h))
                left_y += h + GAP

        right_x = PADDING + col_w + GAP
        for b in right_blocks:
            if b.type == BlockType.TEXT:
                h = reflow_text_height(b, col_w)
                placements.append(PlacedBlock(b, right_x, right_y, col_w, h))
                right_y += h + GAP
            elif b.type == BlockType.TABLE:
                scale = min(col_w / b.ref_width, 1.0)
                w = int(b.ref_width * scale)
                h = int(b.ref_height * scale)
                placements.append(PlacedBlock(b, right_x, right_y, w, h))
                right_y += h + GAP

        y = max(left_y, right_y)

    y += PADDING - GAP
    return Layout("two_column", target_width, y, placements)


def layout_side_by_side(blocks: List[ContentBlock], target_width: int = 0) -> Layout:
    """Side-by-side: image on one side, text on the other."""
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]
    tables = [b for b in blocks if b.type == BlockType.TABLE]

    if not images:
        # No images → fall back to single column
        return layout_single_column(blocks, target_width)

    img = images[0]
    other_images = images[1:]

    # Calculate proportions
    if target_width == 0:
        total_area = sum(b.area for b in blocks) * 1.2
        target_width = max(int(math.sqrt(total_area)), 500)

    content_w = target_width - 2 * PADDING

    # Image gets up to 50% of width
    img_frac = min(0.5, img.ref_width / content_w)
    img_frac = max(img_frac, 0.3)  # at least 30%
    img_w = int(content_w * img_frac)
    text_w = content_w - img_w - GAP

    # Scale image
    scale = min(img_w / img.ref_width, 1.0)
    img_actual_w = int(img.ref_width * scale)
    img_actual_h = int(img.ref_height * scale)

    placements = []
    y = PADDING

    # Image on left
    img_x = PADDING + (img_w - img_actual_w) // 2
    placements.append(PlacedBlock(img, img_x, y, img_actual_w, img_actual_h))

    # Text on right
    text_x = PADDING + img_w + GAP
    text_y = y

    for b in texts:
        h = reflow_text_height(b, text_w)
        placements.append(PlacedBlock(b, text_x, text_y, text_w, h))
        text_y += h + GAP

    for b in tables:
        scale_t = min(text_w / b.ref_width, 1.0)
        w = int(b.ref_width * scale_t)
        h = int(b.ref_height * scale_t)
        placements.append(PlacedBlock(b, text_x, text_y, w, h))
        text_y += h + GAP

    # Additional images below
    below_y = max(y + img_actual_h + GAP, text_y)
    for b in other_images:
        scale_b = min(content_w / b.ref_width, 1.0)
        w = int(b.ref_width * scale_b)
        h = int(b.ref_height * scale_b)
        x = PADDING + (content_w - w) // 2
        placements.append(PlacedBlock(b, x, below_y, w, h))
        below_y += h + GAP

    total_h = max(y + img_actual_h, text_y, below_y) + PADDING
    return Layout("side_by_side", target_width, total_h, placements)


def choose_best_layout(blocks: List[ContentBlock]) -> Layout:
    """Try all layout strategies, pick the best one (most square + highest utilization)."""
    # Estimate target width from total area
    total_area = sum(b.area for b in blocks) * 1.3
    target_side = int(math.sqrt(total_area))
    target_side = max(target_side, 350)
    target_side = min(target_side, 1200)

    candidates = [
        layout_single_column(blocks, target_side),
        layout_two_column(blocks, target_side),
        layout_side_by_side(blocks, target_side),
        # Also try wider variants
        layout_single_column(blocks, int(target_side * 1.3)),
        layout_two_column(blocks, int(target_side * 1.2)),
        layout_side_by_side(blocks, int(target_side * 1.2)),
    ]

    def score(layout: Layout) -> float:
        sq = layout.squareness  # 0-1, higher = more square
        util = layout.utilization  # 0-1, higher = more used
        # Penalize very small or very large canvases
        size_ok = 1.0 if 300 <= layout.width <= 1200 and 300 <= layout.height <= 1200 else 0.7
        return sq * 0.6 + util * 0.3 + size_ok * 0.1

    best = max(candidates, key=score)
    return best


# ---------------------------------------------------------------------------
# Render phase
# ---------------------------------------------------------------------------

def render_layout(layout: Layout) -> Image.Image:
    """Render a layout to a PIL Image."""
    canvas = Image.new("RGB", (layout.width, layout.height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for p in layout.placements:
        if p.block.type == BlockType.TEXT:
            render_text_block(canvas, p)
        elif p.block.type == BlockType.IMAGE:
            render_image_block(canvas, p)
        elif p.block.type == BlockType.TABLE:
            render_table_block(canvas, p)

    return canvas


def render_text_block(canvas: Image.Image, p: PlacedBlock):
    """Render a text block at the given position."""
    font = get_font(p.block.font_size)
    line_height = int(p.block.font_size * 1.3)
    draw = ImageDraw.Draw(canvas)

    lines = wrap_text(p.block.data, font, p.width)
    y = p.y
    for line in lines:
        if y + line_height > p.y + p.height + 5:
            # Truncate with ellipsis
            draw.text((p.x, y), "...", font=font, fill=(150, 150, 150))
            break
        draw.text((p.x, y), line, font=font, fill=(0, 0, 0))
        y += line_height


def render_image_block(canvas: Image.Image, p: PlacedBlock):
    """Render an image block at the given position."""
    img = p.block.data
    resized = img.resize((p.width, p.height), Image.Resampling.LANCZOS)
    canvas.paste(resized, (p.x, p.y))


def render_table_block(canvas: Image.Image, p: PlacedBlock):
    """Render a table block at the given position."""
    draw = ImageDraw.Draw(canvas)
    data = p.block.data
    font = get_font(p.block.font_size)

    if not data:
        return

    num_rows = len(data)
    num_cols = max(len(row) for row in data)
    cell_pad = 6

    row_h = p.height // max(num_rows, 1)
    col_w = p.width // max(num_cols, 1)

    # Draw grid
    for i in range(num_rows + 1):
        y = p.y + i * row_h
        draw.line([(p.x, y), (p.x + p.width, y)], fill=(180, 180, 180), width=1)
    for j in range(num_cols + 1):
        x = p.x + j * col_w
        draw.line([(x, p.y), (x, p.y + p.height)], fill=(180, 180, 180), width=1)

    # Draw cell text
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            if j >= num_cols:
                break
            x = p.x + j * col_w + cell_pad
            y = p.y + i * row_h + cell_pad
            # Truncate if too long
            text = str(cell)
            bbox = font.getbbox(text)
            if bbox[2] - bbox[0] > col_w - 2 * cell_pad:
                while len(text) > 1 and font.getbbox(text + "..")[2] - font.getbbox(text + "..")[0] > col_w - 2 * cell_pad:
                    text = text[:-1]
                text += ".."
            draw.text((x, y), text, font=font, fill=(0, 0, 0))


# ---------------------------------------------------------------------------
# DynamicCanvas baseline rendering (current method)
# ---------------------------------------------------------------------------

def render_baseline(blocks: List[ContentBlock], patch_size: int = 640) -> Image.Image:
    """Render using the old DynamicCanvas approach: fixed-width single column stacked patches."""
    padding = 20
    gap = 15
    content_w = patch_size - 2 * padding

    # Calculate total height
    y = padding
    for b in blocks:
        if b.type == BlockType.TEXT:
            h = reflow_text_height(b, content_w)
            y += h + gap
        elif b.type == BlockType.IMAGE:
            scale = min(content_w / b.ref_width, (patch_size // 2) / b.ref_height, 1.0)
            y += int(b.ref_height * scale) + gap
        elif b.type == BlockType.TABLE:
            scale = min(content_w / b.ref_width, 1.0)
            y += int(b.ref_height * scale) + gap
    y += padding

    # Round up to full patches
    num_patches = max(1, math.ceil(y / patch_size))
    canvas_h = num_patches * patch_size

    canvas = Image.new("RGB", (patch_size, canvas_h), (255, 255, 255))

    # Render
    cur_y = padding
    for b in blocks:
        if b.type == BlockType.TEXT:
            p = PlacedBlock(b, padding, cur_y, content_w, reflow_text_height(b, content_w))
            render_text_block(canvas, p)
            cur_y += p.height + gap
        elif b.type == BlockType.IMAGE:
            scale = min(content_w / b.ref_width, (patch_size // 2) / b.ref_height, 1.0)
            w = int(b.ref_width * scale)
            h = int(b.ref_height * scale)
            x = padding + (content_w - w) // 2
            p = PlacedBlock(b, x, cur_y, w, h)
            render_image_block(canvas, p)
            cur_y += h + gap
        elif b.type == BlockType.TABLE:
            scale = min(content_w / b.ref_width, 1.0)
            w = int(b.ref_width * scale)
            h = int(b.ref_height * scale)
            p = PlacedBlock(b, padding, cur_y, w, h)
            render_table_block(canvas, p)
            cur_y += h + gap

    # Draw patch boundaries
    draw = ImageDraw.Draw(canvas)
    for i in range(1, num_patches):
        y_line = i * patch_size
        draw.line([(0, y_line), (patch_size, y_line)], fill=(200, 200, 200), width=2)

    return canvas


# ---------------------------------------------------------------------------
# ScienceQA sample → content blocks
# ---------------------------------------------------------------------------

def sample_to_blocks(sample: Dict, img: Optional[Image.Image] = None) -> List[ContentBlock]:
    """Convert a ScienceQA sample into content blocks for layout."""
    blocks = []

    # Header
    subj = sample.get("subject", "")
    topic = sample.get("topic", "")
    if subj or topic:
        blocks.append(measure_text(f"[{subj} / {topic}]", font_size=14))

    # Context/hint
    hint = sample.get("hint", "")
    if hint:
        blocks.append(measure_text(f"Context: {hint}", font_size=16))

    # Image
    if img is not None:
        blocks.append(measure_image(img, max_dim=500))

    # Question + choices
    q_text = f"Q: {sample['question']}\n"
    for i, c in enumerate(sample["choices"]):
        letter = chr(65 + i)
        marker = "✓" if i == sample["answer"] else "○"
        q_text += f"  {marker} {letter}. {c}\n"
    blocks.append(measure_text(q_text.strip(), font_size=18))

    # Lecture (background knowledge)
    lecture = sample.get("lecture", "")
    if lecture:
        # Check if it contains table-like content
        if "|" in lecture and lecture.count("|") >= 4:
            # Parse pipe table
            table_lines = [l.strip() for l in lecture.split("\n") if "|" in l]
            table_data = []
            for line in table_lines:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells and not all(set(c) <= {'-', ' '} for c in cells):
                    table_data.append(cells)
            if table_data:
                # Non-table part of lecture
                non_table = "\n".join(l for l in lecture.split("\n") if "|" not in l).strip()
                if non_table:
                    blocks.append(measure_text(f"Knowledge: {non_table}", font_size=15))
                blocks.append(measure_table(table_data, font_size=15))
            else:
                blocks.append(measure_text(f"Knowledge: {lecture}", font_size=15))
        else:
            blocks.append(measure_text(f"Knowledge: {lecture}", font_size=15))

    # Solution
    solution = sample.get("solution", "")
    if solution:
        blocks.append(measure_text(f"Solution: {solution}", font_size=15))

    return blocks


# ---------------------------------------------------------------------------
# Demo: before/after comparison
# ---------------------------------------------------------------------------

def create_comparison(sample: Dict, img: Optional[Image.Image], sample_id: int, output_dir: str):
    """Create a before/after comparison image for a sample."""
    blocks = sample_to_blocks(sample, img)

    # Baseline (old method)
    baseline = render_baseline(blocks)

    # Smart layout
    best = choose_best_layout(blocks)
    smart = render_layout(best)

    # Create comparison image
    margin = 30
    label_h = 40
    total_w = baseline.width + smart.width + 3 * margin
    total_h = max(baseline.height, smart.height) + 2 * margin + label_h

    comparison = Image.new("RGB", (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(comparison)
    font = get_font(18)
    font_small = get_font(14)

    # Left: baseline
    lx = margin
    draw.text((lx, margin // 2), f"BEFORE: DynamicCanvas (640×{baseline.height})", font=font, fill=(200, 0, 0))
    comparison.paste(baseline, (lx, label_h))
    # Border
    draw.rectangle([lx - 1, label_h - 1, lx + baseline.width, label_h + baseline.height], outline=(200, 0, 0), width=2)

    # Right: smart
    rx = margin + baseline.width + margin
    sq_pct = best.squareness * 100
    util_pct = best.utilization * 100
    draw.text((rx, margin // 2),
              f"AFTER: {best.name} ({best.width}×{best.height}, sq={sq_pct:.0f}%, util={util_pct:.0f}%)",
              font=font, fill=(0, 150, 0))
    comparison.paste(smart, (rx, label_h))
    draw.rectangle([rx - 1, label_h - 1, rx + smart.width, label_h + smart.height], outline=(0, 150, 0), width=2)

    # Info
    info = f"Sample #{sample_id} | {sample.get('subject','')} / {sample.get('topic','')} | {len(blocks)} blocks"
    draw.text((margin, total_h - margin + 5), info, font=font_small, fill=(100, 100, 100))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"comparison_{sample_id}.png")
    comparison.save(out_path)
    print(f"  Saved: {out_path} ({total_w}×{total_h})")
    print(f"    Baseline: {baseline.width}×{baseline.height} | Smart: {best.width}×{best.height} ({best.name})")
    print(f"    Squareness: {sq_pct:.0f}% | Utilization: {util_pct:.0f}%")

    # Also save individual canvases
    baseline.save(os.path.join(output_dir, f"baseline_{sample_id}.png"))
    smart.save(os.path.join(output_dir, f"smart_{sample_id}.png"))

    return baseline, smart, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset

    print("=" * 60)
    print("SmartCanvasLayout — Before/After Demo")
    print("=" * 60)

    # Load ScienceQA cached data
    cache_path = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"
    with open(cache_path, "rb") as f:
        train_data, test_data = pickle.load(f)

    # Load HF dataset for images
    print("Loading ScienceQA images from HF dataset...")
    ds = load_dataset("derek-thomas/ScienceQA", split="train")

    output_dir = "/home/cyf/codex/canvas_layout_demo"

    # Target sample indices (rich multimodal content)
    target_indices = [4332, 9, 8805, 264, 7595]

    for idx in target_indices:
        if idx >= len(train_data):
            print(f"\nSkipping idx {idx}: out of range")
            continue

        sample = train_data[idx]
        print(f"\n{'='*60}")
        print(f"Sample #{idx}: {sample.get('subject','')} / {sample.get('topic','')}")
        print(f"  Q: {sample['question'][:80]}...")
        print(f"  hint={len(sample.get('hint',''))} lecture={len(sample.get('lecture',''))} "
              f"solution={len(sample.get('solution',''))} choices={len(sample['choices'])}")

        # Get image from HF dataset
        # The cached data uses sequential indices that may differ from HF indices
        # Try to get image directly
        img = None
        hf_idx = sample.get("hf_index", idx)
        try:
            hf_sample = ds[hf_idx]
            if hf_sample.get("image") is not None:
                img = hf_sample["image"].convert("RGB")
                print(f"  Image: {img.size}")
        except Exception as e:
            print(f"  No image: {e}")

        # If no image from HF index, try matching by question
        if img is None:
            # Search by question text
            for hi in range(min(len(ds), 15000)):
                if ds[hi]["question"] == sample["question"]:
                    if ds[hi].get("image") is not None:
                        img = ds[hi]["image"].convert("RGB")
                        print(f"  Image found at HF idx {hi}: {img.size}")
                        break

        create_comparison(sample, img, idx, output_dir)

    print(f"\n{'='*60}")
    print(f"All comparisons saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
