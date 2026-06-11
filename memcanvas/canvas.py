"""Canvas construction and rendering utilities for MemCanvas."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


class BlockType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class ContentBlock:
    type: BlockType
    data: Any
    ref_width: int = 0
    ref_height: int = 0
    area: float = 0.0
    aspect_ratio: float = 1.0
    char_count: int = 0
    font_size: int = 18
    title: str | None = None


@dataclass
class PlacedBlock:
    block: ContentBlock
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class Layout:
    name: str
    width: int
    height: int
    placements: list[PlacedBlock] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def squareness(self) -> float:
        ratio = self.aspect_ratio
        return min(ratio, 1 / ratio) if ratio > 0 else 0.0

    @property
    def total_area(self) -> int:
        return self.width * self.height

    @property
    def content_area(self) -> int:
        return sum(p.width * p.height for p in self.placements)

    @property
    def utilization(self) -> float:
        return self.content_area / max(self.total_area, 1)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

PADDING = 15
GAP = 10
BACKGROUND = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
BORDER_COLOR = (210, 210, 210)

_font_cache: dict[tuple[str, int], ImageFont.ImageFont] = {}


def _font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def get_font(size: int) -> ImageFont.ImageFont:
    path = _font_path() or "default"
    key = (path, size)
    if key not in _font_cache:
        if path == "default":
            _font_cache[key] = ImageFont.load_default()
        else:
            _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        current_width = 0
        for char in paragraph:
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0] if bbox else max(8, getattr(font, "size", 12) // 2)
            if current and current_width + char_width > max_width:
                lines.append(current)
                current = char
                current_width = char_width
            else:
                current += char
                current_width += char_width
        if current:
            lines.append(current)
    return lines


def measure_text(text: str, font_size: int = 18, ref_width: int = 600, title: str | None = None) -> ContentBlock:
    font = get_font(font_size)
    line_height = int(font_size * 1.35)
    lines = wrap_text(text, font, ref_width)
    height = max(line_height, len(lines) * line_height)
    return ContentBlock(
        type=BlockType.TEXT,
        data=text,
        ref_width=ref_width,
        ref_height=height,
        area=ref_width * height,
        char_count=len(str(text)),
        font_size=font_size,
        title=title,
    )


def measure_image(image: Image.Image | str | Path, max_dim: int = 500, title: str | None = None) -> ContentBlock:
    img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    width, height = img.size
    scale = min(max_dim / max(width, height), 1.0)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return ContentBlock(
        type=BlockType.IMAGE,
        data=img,
        ref_width=new_width,
        ref_height=new_height,
        area=new_width * new_height,
        aspect_ratio=new_width / max(new_height, 1),
        title=title,
    )


def measure_table(rows: Iterable[Iterable[Any]], font_size: int = 16, title: str | None = None) -> ContentBlock:
    table = [[str(cell) for cell in row] for row in rows]
    if not table:
        table = [[""]]
    font = get_font(font_size)
    cell_pad = 8
    row_height = int(font_size * 1.35) + 2 * cell_pad
    n_cols = max(len(row) for row in table)
    col_widths = [60] * n_cols
    for row in table:
        for idx, cell in enumerate(row):
            bbox = font.getbbox(cell[:80])
            col_widths[idx] = max(col_widths[idx], min(260, bbox[2] - bbox[0] + 2 * cell_pad))
    width = sum(col_widths) + 4
    height = len(table) * row_height + 4
    return ContentBlock(
        type=BlockType.TABLE,
        data=table,
        ref_width=width,
        ref_height=height,
        area=width * height,
        font_size=font_size,
        title=title,
    )


def reflow_text_height(block: ContentBlock, target_width: int) -> int:
    font = get_font(block.font_size)
    line_height = int(block.font_size * 1.35)
    return max(line_height, len(wrap_text(block.data, font, target_width)) * line_height)


def layout_single_column(blocks: list[ContentBlock], target_width: int | None = None) -> Layout:
    if not blocks:
        blocks = [measure_text("(empty)")]
    width = target_width or min(960, max(360, max(block.ref_width for block in blocks) + 2 * PADDING))
    content_width = width - 2 * PADDING
    placements: list[PlacedBlock] = []
    y = PADDING
    for block in blocks:
        if block.type == BlockType.TEXT:
            height = reflow_text_height(block, content_width)
            placements.append(PlacedBlock(block, PADDING, y, content_width, height))
        elif block.type == BlockType.IMAGE:
            scale = min(content_width / block.ref_width, 1.0)
            height = max(1, int(block.ref_height * scale))
            placements.append(PlacedBlock(block, PADDING, y, int(block.ref_width * scale), height))
        else:
            scale = min(content_width / block.ref_width, 1.0)
            height = max(1, int(block.ref_height * scale))
            placements.append(PlacedBlock(block, PADDING, y, int(block.ref_width * scale), height))
        y += placements[-1].height + GAP
    return Layout("single_column", width, y + PADDING - GAP, placements)


def layout_two_column(blocks: list[ContentBlock], target_width: int = 960) -> Layout:
    left_x = PADDING
    col_width = (target_width - 2 * PADDING - GAP) // 2
    right_x = left_x + col_width + GAP
    ys = [PADDING, PADDING]
    placements: list[PlacedBlock] = []
    for block in sorted(blocks, key=lambda b: b.area, reverse=True):
        col = 0 if ys[0] <= ys[1] else 1
        x = left_x if col == 0 else right_x
        if block.type == BlockType.TEXT:
            height = reflow_text_height(block, col_width)
            width = col_width
        else:
            scale = min(col_width / block.ref_width, 1.0)
            width = int(block.ref_width * scale)
            height = max(1, int(block.ref_height * scale))
        placements.append(PlacedBlock(block, x, ys[col], width, height))
        ys[col] += height + GAP
    return Layout("two_column", target_width, max(ys) + PADDING - GAP, placements)


def score_layout(layout: Layout, min_size: int = 224, max_size: int = 1344) -> float:
    in_bounds = int(min_size <= layout.width <= max_size and min_size <= layout.height <= max_size)
    return 0.6 * layout.squareness + 0.3 * layout.utilization + 0.1 * in_bounds


def choose_best_layout(blocks: list[ContentBlock], max_width: int = 960) -> Layout:
    candidates = [
        layout_single_column(blocks, target_width=640),
        layout_single_column(blocks, target_width=830),
    ]
    if len(blocks) >= 3:
        candidates.append(layout_two_column(blocks, target_width=max_width))
    return max(candidates, key=score_layout)


def _draw_text(draw: ImageDraw.ImageDraw, placement: PlacedBlock) -> None:
    font = get_font(placement.block.font_size)
    line_height = int(placement.block.font_size * 1.35)
    y = placement.y
    for line in wrap_text(placement.block.data, font, placement.width):
        draw.text((placement.x, y), line, fill=TEXT_COLOR, font=font)
        y += line_height


def _draw_table(draw: ImageDraw.ImageDraw, placement: PlacedBlock) -> None:
    rows = placement.block.data
    font = get_font(placement.block.font_size)
    row_height = int(placement.block.font_size * 1.35) + 14
    n_cols = max(len(row) for row in rows)
    col_width = max(40, placement.width // max(n_cols, 1))
    for r, row in enumerate(rows):
        y = placement.y + r * row_height
        for c in range(n_cols):
            x = placement.x + c * col_width
            draw.rectangle([x, y, x + col_width, y + row_height], outline=BORDER_COLOR)
            value = row[c] if c < len(row) else ""
            draw.text((x + 6, y + 6), str(value)[:40], fill=TEXT_COLOR, font=font)


def render_layout(layout: Layout) -> Image.Image:
    image = Image.new("RGB", (max(1, layout.width), max(1, layout.height)), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for placement in layout.placements:
        draw.rounded_rectangle(
            [placement.x - 4, placement.y - 4, placement.x + placement.width + 4, placement.y + placement.height + 4],
            radius=8,
            outline=BORDER_COLOR,
            width=1,
        )
        if placement.block.type == BlockType.TEXT:
            _draw_text(draw, placement)
        elif placement.block.type == BlockType.IMAGE:
            block_image = placement.block.data.resize((placement.width, placement.height), Image.Resampling.LANCZOS)
            image.paste(block_image, (placement.x, placement.y))
        elif placement.block.type == BlockType.TABLE:
            _draw_table(draw, placement)
    return image


def render_canvas(blocks: list[ContentBlock], output_path: str | Path | None = None) -> Image.Image:
    layout = choose_best_layout(blocks)
    image = render_layout(layout)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    return image
