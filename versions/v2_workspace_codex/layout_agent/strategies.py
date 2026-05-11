#!/usr/bin/env python3
"""
strategies.py — 8 layout strategies + unified content block abstraction

Design principles:
  - Benchmark-agnostic: ContentBlock uses semantic_role, not field names
  - Parameterized: each strategy accepts (blocks, params) where params come from agent JSON
  - Deterministic: same (blocks, params) always produce the same Layout
"""

import io
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class BlockType(Enum):
    TEXT = "text"
    IMAGE = "image"


class SemanticRole(Enum):
    """Benchmark-agnostic content roles."""
    QUESTION = "question"
    CHOICES = "choices"
    CONTEXT = "context"          # hint, passage, surrounding text
    FIGURE = "figure"            # photo, diagram, illustration
    TABLE = "table"              # table rendered as image or structured
    CHART = "chart"              # chart / plot as image
    DOCUMENT = "document"        # document page as image
    KNOWLEDGE = "knowledge"      # lecture, background knowledge
    EXPLANATION = "explanation"   # solution, rationale
    HEADER = "header"            # subject/topic metadata
    OTHER = "other"


# Priority order: higher priority blocks rendered first / more prominently
ROLE_PRIORITY = {
    SemanticRole.QUESTION: 0,
    SemanticRole.CHOICES: 1,
    SemanticRole.FIGURE: 2,
    SemanticRole.TABLE: 3,
    SemanticRole.CHART: 3,
    SemanticRole.DOCUMENT: 3,
    SemanticRole.CONTEXT: 4,
    SemanticRole.KNOWLEDGE: 5,
    SemanticRole.EXPLANATION: 6,
    SemanticRole.HEADER: 7,
    SemanticRole.OTHER: 8,
}

# Width classes (pixels)
WIDTH_MAP = {"narrow": 480, "medium": 640, "wide": 860}
# Font profile offsets (added to base font size)
FONT_OFFSET = {"compact": -2, "normal": 0, "large": 2}
# Image scale factors
IMAGE_SCALE_MAP = {"small": 0.6, "medium": 0.8, "large": 1.0}
# Truncation ratios for knowledge/explanation blocks
TRUNCATION_MAP = {"none": 1.0, "light": 0.6, "heavy": 0.3}

# All valid strategy names
STRATEGY_NAMES = [
    "single_column", "two_column", "side_by_side",
    "compact", "emphasis", "grid", "text_wrap", "panoramic",
]

# All valid parameter values
PARAM_SPACE = {
    "strategy": STRATEGY_NAMES,
    "width_class": ["narrow", "medium", "wide"],
    "image_scale": ["small", "medium", "large"],
    "content_priority": ["question_first", "context_first", "balanced"],
    "truncation": ["none", "light", "heavy"],
    "font_profile": ["compact", "normal", "large"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """A measured content block — benchmark agnostic."""
    type: BlockType
    role: SemanticRole
    data: Any                    # str for text, PIL.Image for image
    ref_width: int = 0
    ref_height: int = 0
    area: float = 0.0
    aspect_ratio: float = 1.0   # w/h, only meaningful for images
    char_count: int = 0
    font_size: int = 18         # base font size


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


@dataclass
class LayoutParams:
    """Parameterized layout configuration — agent's action space."""
    strategy: str = "single_column"
    width_class: str = "medium"      # narrow / medium / wide
    image_scale: str = "medium"      # small / medium / large
    content_priority: str = "balanced"  # question_first / context_first / balanced
    truncation: str = "none"         # none / light / heavy
    font_profile: str = "normal"     # compact / normal / large

    @property
    def target_width(self) -> int:
        return WIDTH_MAP.get(self.width_class, 640)

    @property
    def font_offset(self) -> int:
        return FONT_OFFSET.get(self.font_profile, 0)

    @property
    def img_scale(self) -> float:
        return IMAGE_SCALE_MAP.get(self.image_scale, 0.8)

    @property
    def trunc_ratio(self) -> float:
        return TRUNCATION_MAP.get(self.truncation, 1.0)

    def config_hash(self) -> str:
        """Deterministic hash for reward table lookup."""
        return f"{self.strategy}_{self.width_class}_{self.image_scale}_{self.content_priority}_{self.truncation}_{self.font_profile}"

    @staticmethod
    def from_dict(d: Dict) -> "LayoutParams":
        return LayoutParams(
            strategy=d.get("strategy", "single_column"),
            width_class=d.get("width_class", "medium"),
            image_scale=d.get("image_scale", "medium"),
            content_priority=d.get("content_priority", "balanced"),
            truncation=d.get("truncation", "none"),
            font_profile=d.get("font_profile", "normal"),
        )

    def is_valid(self) -> bool:
        return all([
            self.strategy in STRATEGY_NAMES,
            self.width_class in WIDTH_MAP,
            self.image_scale in IMAGE_SCALE_MAP,
            self.content_priority in ("question_first", "context_first", "balanced"),
            self.truncation in TRUNCATION_MAP,
            self.font_profile in FONT_OFFSET,
        ])


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
    size = max(size, 8)
    if size not in _font_cache:
        if _font_path:
            _font_cache[size] = ImageFont.truetype(_font_path, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
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
    return lines if lines else ['']


# ---------------------------------------------------------------------------
# Measure helpers
# ---------------------------------------------------------------------------

def measure_text(text: str, role: SemanticRole = SemanticRole.OTHER,
                 font_size: int = 18, ref_width: int = 600) -> ContentBlock:
    font = get_font(font_size)
    line_height = int(font_size * 1.3)
    lines = wrap_text(text, font, ref_width)
    ref_height = max(len(lines) * line_height, line_height)
    return ContentBlock(
        type=BlockType.TEXT, role=role, data=text,
        ref_width=ref_width, ref_height=ref_height,
        area=ref_width * ref_height, char_count=len(text),
        font_size=font_size,
    )


def measure_image(img: Image.Image, role: SemanticRole = SemanticRole.FIGURE,
                  max_dim: int = 500) -> ContentBlock:
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return ContentBlock(
        type=BlockType.IMAGE, role=role, data=img,
        ref_width=new_w, ref_height=new_h,
        area=new_w * new_h, aspect_ratio=new_w / max(new_h, 1),
    )


# ---------------------------------------------------------------------------
# Content processing: ordering + truncation
# ---------------------------------------------------------------------------

def order_blocks(blocks: List[ContentBlock], priority: str) -> List[ContentBlock]:
    """Reorder blocks based on content_priority setting."""
    if priority == "question_first":
        key = lambda b: (0 if b.role in (SemanticRole.QUESTION, SemanticRole.CHOICES) else
                         1 if b.role in (SemanticRole.FIGURE, SemanticRole.TABLE,
                                         SemanticRole.CHART, SemanticRole.DOCUMENT) else 2,
                         ROLE_PRIORITY.get(b.role, 8))
    elif priority == "context_first":
        key = lambda b: (0 if b.role in (SemanticRole.FIGURE, SemanticRole.TABLE,
                                         SemanticRole.CHART, SemanticRole.DOCUMENT,
                                         SemanticRole.CONTEXT) else
                         1 if b.role in (SemanticRole.QUESTION, SemanticRole.CHOICES) else 2,
                         ROLE_PRIORITY.get(b.role, 8))
    else:  # balanced — use default role priority
        key = lambda b: ROLE_PRIORITY.get(b.role, 8)
    return sorted(blocks, key=key)


def truncate_blocks(blocks: List[ContentBlock], ratio: float) -> List[ContentBlock]:
    """Truncate knowledge/explanation text blocks by ratio."""
    if ratio >= 1.0:
        return blocks
    result = []
    for b in blocks:
        if b.type == BlockType.TEXT and b.role in (SemanticRole.KNOWLEDGE, SemanticRole.EXPLANATION):
            keep = max(int(len(b.data) * ratio), 50)
            truncated_text = b.data[:keep] + "..." if len(b.data) > keep else b.data
            result.append(measure_text(truncated_text, b.role, b.font_size, b.ref_width))
        else:
            result.append(b)
    return result


def apply_font_offset(blocks: List[ContentBlock], offset: int) -> List[ContentBlock]:
    """Adjust font sizes by offset."""
    if offset == 0:
        return blocks
    result = []
    for b in blocks:
        if b.type == BlockType.TEXT:
            new_fs = max(b.font_size + offset, 10)
            result.append(measure_text(b.data, b.role, new_fs, b.ref_width))
        else:
            result.append(b)
    return result


def scale_images(blocks: List[ContentBlock], scale: float) -> List[ContentBlock]:
    """Scale image blocks."""
    if abs(scale - 1.0) < 0.01:
        return blocks
    result = []
    for b in blocks:
        if b.type == BlockType.IMAGE:
            new_w = max(int(b.ref_width * scale), 50)
            new_h = max(int(b.ref_height * scale), 50)
            result.append(ContentBlock(
                type=BlockType.IMAGE, role=b.role, data=b.data,
                ref_width=new_w, ref_height=new_h,
                area=new_w * new_h, aspect_ratio=b.aspect_ratio,
            ))
        else:
            result.append(b)
    return result


def preprocess_blocks(blocks: List[ContentBlock], params: LayoutParams) -> List[ContentBlock]:
    """Apply all preprocessing: truncation, font offset, image scaling, ordering."""
    blocks = truncate_blocks(blocks, params.trunc_ratio)
    blocks = apply_font_offset(blocks, params.font_offset)
    blocks = scale_images(blocks, params.img_scale)
    blocks = order_blocks(blocks, params.content_priority)
    return blocks


# ---------------------------------------------------------------------------
# Reflow helper
# ---------------------------------------------------------------------------

def reflow_text_height(block: ContentBlock, target_width: int) -> int:
    font = get_font(block.font_size)
    line_height = int(block.font_size * 1.3)
    lines = wrap_text(block.data, font, target_width)
    return max(len(lines) * line_height, line_height)


# ---------------------------------------------------------------------------
# 8 Layout strategies
# ---------------------------------------------------------------------------

PADDING = 15
GAP = 10


def _place_text(b: ContentBlock, x: int, y: int, w: int) -> Tuple[PlacedBlock, int]:
    h = reflow_text_height(b, w)
    return PlacedBlock(b, x, y, w, h), y + h + GAP


def _place_image(b: ContentBlock, x: int, y: int, max_w: int,
                 center: bool = True) -> Tuple[PlacedBlock, int]:
    scale = min(max_w / b.ref_width, 1.0)
    w, h = int(b.ref_width * scale), int(b.ref_height * scale)
    px = x + (max_w - w) // 2 if center else x
    return PlacedBlock(b, px, y, w, h), y + h + GAP


# --- Strategy 1: single_column ---

def layout_single_column(blocks: List[ContentBlock], tw: int) -> Layout:
    """Single column vertical stack."""
    cw = tw - 2 * PADDING
    placements = []
    y = PADDING
    for b in blocks:
        if b.type == BlockType.TEXT:
            p, y = _place_text(b, PADDING, y, cw)
        else:
            p, y = _place_image(b, PADDING, y, cw)
        placements.append(p)
    return Layout("single_column", tw, y + PADDING - GAP, placements)


# --- Strategy 2: two_column ---

def layout_two_column(blocks: List[ContentBlock], tw: int) -> Layout:
    """Images span full width, text split into two columns."""
    cw = tw - 2 * PADDING
    col_w = (cw - GAP) // 2
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]

    placements = []
    y = PADDING

    # Full-width images
    for b in images:
        p, y = _place_image(b, PADDING, y, cw)
        placements.append(p)

    # Two columns for text
    if texts:
        mid = len(texts) // 2 + 1
        left, right = texts[:mid], texts[mid:]
        ly, ry = y, y
        for b in left:
            p, ly = _place_text(b, PADDING, ly, col_w)
            placements.append(p)
        rx = PADDING + col_w + GAP
        for b in right:
            p, ry = _place_text(b, rx, ry, col_w)
            placements.append(p)
        y = max(ly, ry)

    return Layout("two_column", tw, y + PADDING - GAP, placements)


# --- Strategy 3: side_by_side ---

def layout_side_by_side(blocks: List[ContentBlock], tw: int) -> Layout:
    """Image on left, text on right. Falls back to single_column if no image."""
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]

    if not images:
        return layout_single_column(blocks, tw)

    cw = tw - 2 * PADDING
    img = images[0]
    img_frac = max(0.3, min(0.5, img.ref_width / cw))
    img_w = int(cw * img_frac)
    txt_w = cw - img_w - GAP

    placements = []
    y = PADDING

    # Image left
    scale = min(img_w / img.ref_width, 1.0)
    iw, ih = int(img.ref_width * scale), int(img.ref_height * scale)
    placements.append(PlacedBlock(img, PADDING + (img_w - iw) // 2, y, iw, ih))

    # Text right
    tx = PADDING + img_w + GAP
    ty = y
    for b in texts:
        p, ty = _place_text(b, tx, ty, txt_w)
        placements.append(p)

    # Extra images below
    below_y = max(y + ih + GAP, ty)
    for b in images[1:]:
        p, below_y = _place_image(b, PADDING, below_y, cw)
        placements.append(p)

    total_h = max(y + ih, ty, below_y) + PADDING
    return Layout("side_by_side", tw, total_h, placements)


# --- Strategy 4: compact ---

def layout_compact(blocks: List[ContentBlock], tw: int) -> Layout:
    """Tight spacing, maximum information density."""
    pad, gap = 8, 5
    cw = tw - 2 * pad
    placements = []
    y = pad
    for b in blocks:
        if b.type == BlockType.TEXT:
            h = reflow_text_height(b, cw)
            placements.append(PlacedBlock(b, pad, y, cw, h))
            y += h + gap
        else:
            p, y_new = _place_image(b, pad, y, cw)
            placements.append(p)
            y = y_new - GAP + gap  # tighter gap
    return Layout("compact", tw, y + pad - gap, placements)


# --- Strategy 5: emphasis ---

def layout_emphasis(blocks: List[ContentBlock], tw: int) -> Layout:
    """Question/choices prominent (larger), background smaller."""
    cw = tw - 2 * PADDING
    placements = []
    y = PADDING

    for b in blocks:
        if b.type == BlockType.IMAGE:
            p, y = _place_image(b, PADDING, y, cw)
            placements.append(p)
            continue

        # Emphasize question + choices: +2pt font
        if b.role in (SemanticRole.QUESTION, SemanticRole.CHOICES):
            emph_fs = b.font_size + 2
            font = get_font(emph_fs)
            lh = int(emph_fs * 1.3)
            lines = wrap_text(b.data, font, cw)
            h = max(len(lines) * lh, lh)
            # Store with modified font for rendering
            emph_block = ContentBlock(
                type=BlockType.TEXT, role=b.role, data=b.data,
                ref_width=b.ref_width, ref_height=h,
                area=cw * h, char_count=b.char_count, font_size=emph_fs,
            )
            placements.append(PlacedBlock(emph_block, PADDING, y, cw, h))
            y += h + GAP
        else:
            # De-emphasize: -2pt font
            de_fs = max(b.font_size - 2, 10)
            font = get_font(de_fs)
            lh = int(de_fs * 1.3)
            lines = wrap_text(b.data, font, cw)
            h = max(len(lines) * lh, lh)
            de_block = ContentBlock(
                type=BlockType.TEXT, role=b.role, data=b.data,
                ref_width=b.ref_width, ref_height=h,
                area=cw * h, char_count=b.char_count, font_size=de_fs,
            )
            placements.append(PlacedBlock(de_block, PADDING, y, cw, h))
            y += h + GAP

    return Layout("emphasis", tw, y + PADDING - GAP, placements)


# --- Strategy 6: grid ---

def layout_grid(blocks: List[ContentBlock], tw: int) -> Layout:
    """Grid layout — images in grid, text below."""
    cw = tw - 2 * PADDING
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]

    placements = []
    y = PADDING

    if len(images) >= 2:
        cols = min(len(images), 3)
        cell_w = (cw - (cols - 1) * GAP) // cols
        row_imgs = [images[i:i + cols] for i in range(0, len(images), cols)]

        for row in row_imgs:
            row_h = 0
            for j, b in enumerate(row):
                scale = min(cell_w / b.ref_width, 1.0)
                w, h = int(b.ref_width * scale), int(b.ref_height * scale)
                x = PADDING + j * (cell_w + GAP) + (cell_w - w) // 2
                placements.append(PlacedBlock(b, x, y, w, h))
                row_h = max(row_h, h)
            y += row_h + GAP
    elif images:
        # Single image, full width
        p, y = _place_image(images[0], PADDING, y, cw)
        placements.append(p)

    # Text below grid
    for b in texts:
        p, y = _place_text(b, PADDING, y, cw)
        placements.append(p)

    return Layout("grid", tw, y + PADDING - GAP, placements)


# --- Strategy 7: text_wrap ---

def layout_text_wrap(blocks: List[ContentBlock], tw: int) -> Layout:
    """Image floats right, text wraps around it."""
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]

    if not images:
        return layout_single_column(blocks, tw)

    cw = tw - 2 * PADDING
    img = images[0]

    # Image takes ~40% width, floats right
    img_w = min(int(cw * 0.4), img.ref_width)
    scale = img_w / img.ref_width
    img_h = int(img.ref_height * scale)
    img_x = PADDING + cw - img_w

    placements = []
    y = PADDING

    placements.append(PlacedBlock(img, img_x, y, img_w, img_h))

    # Text wraps: narrower beside image, full width below
    wrap_w = cw - img_w - GAP
    ty = y
    remaining_texts = list(texts)

    # Phase 1: text beside image (narrower)
    while remaining_texts and ty < y + img_h:
        b = remaining_texts.pop(0)
        h = reflow_text_height(b, wrap_w)
        if ty + h > y + img_h and ty > y:
            remaining_texts.insert(0, b)
            break
        placements.append(PlacedBlock(b, PADDING, ty, wrap_w, h))
        ty += h + GAP

    # Phase 2: text below image (full width)
    below_y = max(ty, y + img_h + GAP)
    for b in remaining_texts:
        p, below_y = _place_text(b, PADDING, below_y, cw)
        placements.append(p)

    # Extra images below
    for b in images[1:]:
        p, below_y = _place_image(b, PADDING, below_y, cw)
        placements.append(p)

    return Layout("text_wrap", tw, below_y + PADDING - GAP, placements)


# --- Strategy 8: panoramic ---

def layout_panoramic(blocks: List[ContentBlock], tw: int) -> Layout:
    """Wide format (~2:1 aspect ratio). Image left, multi-column text right."""
    # Force wider canvas
    wide_tw = max(tw, int(tw * 1.5))
    cw = wide_tw - 2 * PADDING
    images = [b for b in blocks if b.type == BlockType.IMAGE]
    texts = [b for b in blocks if b.type == BlockType.TEXT]

    placements = []
    y = PADDING

    if images:
        img = images[0]
        img_w = int(cw * 0.35)
        scale = min(img_w / img.ref_width, 1.0)
        iw, ih = int(img.ref_width * scale), int(img.ref_height * scale)
        placements.append(PlacedBlock(img, PADDING, y, iw, ih))

        # Text in 2 columns to the right
        txt_area_w = cw - img_w - GAP
        col_w = (txt_area_w - GAP) // 2
        tx_left = PADDING + img_w + GAP
        tx_right = tx_left + col_w + GAP

        mid = len(texts) // 2 + 1
        left_texts, right_texts = texts[:mid], texts[mid:]
        ly, ry = y, y

        for b in left_texts:
            p, ly = _place_text(b, tx_left, ly, col_w)
            placements.append(p)
        for b in right_texts:
            p, ry = _place_text(b, tx_right, ry, col_w)
            placements.append(p)

        y = max(y + ih, ly, ry)

        # Extra images below
        for b in images[1:]:
            p, y = _place_image(b, PADDING, y + GAP, cw)
            placements.append(p)
    else:
        # No images: 3-column text
        col_w = (cw - 2 * GAP) // 3
        thirds = [texts[i::3] for i in range(3)]
        col_ys = [y, y, y]
        for ci, col_blocks in enumerate(thirds):
            cx = PADDING + ci * (col_w + GAP)
            for b in col_blocks:
                p, col_ys[ci] = _place_text(b, cx, col_ys[ci], col_w)
                placements.append(p)
        y = max(col_ys)

    return Layout("panoramic", wide_tw, y + PADDING, placements)


# Strategy dispatch table
STRATEGY_FN = {
    "single_column": layout_single_column,
    "two_column": layout_two_column,
    "side_by_side": layout_side_by_side,
    "compact": layout_compact,
    "emphasis": layout_emphasis,
    "grid": layout_grid,
    "text_wrap": layout_text_wrap,
    "panoramic": layout_panoramic,
}


# ---------------------------------------------------------------------------
# Main entry: generate layout from params
# ---------------------------------------------------------------------------

def generate_layout(blocks: List[ContentBlock], params: LayoutParams) -> Layout:
    """Generate a layout from content blocks and agent-specified parameters."""
    processed = preprocess_blocks(blocks, params)
    fn = STRATEGY_FN.get(params.strategy, layout_single_column)
    return fn(processed, params.target_width)


def generate_all_candidates(blocks: List[ContentBlock]) -> List[Tuple[LayoutParams, Layout]]:
    """Generate all 24 standard candidates (8 strategies × 3 widths) for reward table."""
    candidates = []
    for strategy in STRATEGY_NAMES:
        for width_class in ["narrow", "medium", "wide"]:
            params = LayoutParams(strategy=strategy, width_class=width_class)
            layout = generate_layout(blocks, params)
            candidates.append((params, layout))
    return candidates


# ---------------------------------------------------------------------------
# Quality score (deterministic, no model)
# ---------------------------------------------------------------------------

def compute_quality_score(layout: Layout) -> float:
    """Compute heuristic quality score in [0, 0.3]."""
    # 1. Squareness
    sq = layout.squareness

    # 2. Utilization
    util = min(layout.utilization, 1.0)

    # 3. Size OK
    w, h = layout.width, layout.height
    size_ok = 1.0 if (300 <= w <= 1200 and 300 <= h <= 1200) else 0.5

    # 4. Readability: min text column width >= 200px
    text_widths = [p.width for p in layout.placements if p.block.type == BlockType.TEXT]
    if text_widths:
        min_col = min(text_widths)
        readability = min(min_col / 200.0, 1.0)
    else:
        readability = 1.0  # no text to read

    # 5. Vision token efficiency
    # Qwen2VL: 28×28 patch, every 4 patches merge to 1 token
    tokens = (math.ceil(w / 28) * math.ceil(h / 28)) // 4
    token_eff = 1.0 if tokens <= 1280 else max(0, 1.0 - (tokens - 1280) / 1280)

    raw = (sq * 0.25 + util * 0.25 + size_ok * 0.15
           + readability * 0.20 + token_eff * 0.15)
    return raw * 0.3


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_layout(layout: Layout) -> Image.Image:
    canvas = Image.new("RGB", (layout.width, layout.height), (255, 255, 255))
    for p in layout.placements:
        if p.block.type == BlockType.TEXT:
            _render_text(canvas, p)
        else:
            _render_image(canvas, p)
    return canvas


def _render_text(canvas: Image.Image, p: PlacedBlock):
    font = get_font(p.block.font_size)
    lh = int(p.block.font_size * 1.3)
    draw = ImageDraw.Draw(canvas)
    lines = wrap_text(p.block.data, font, p.width)
    y = p.y
    for line in lines:
        if y + lh > p.y + p.height + 5:
            draw.text((p.x, y), "...", font=font, fill=(150, 150, 150))
            break
        draw.text((p.x, y), line, font=font, fill=(0, 0, 0))
        y += lh


def _render_image(canvas: Image.Image, p: PlacedBlock):
    img = p.block.data
    if not isinstance(img, Image.Image):
        return
    resized = img.resize((max(p.width, 1), max(p.height, 1)), Image.Resampling.LANCZOS)
    canvas.paste(resized, (p.x, p.y))


# ---------------------------------------------------------------------------
# Benchmark adapters — convert benchmark-specific data to generic blocks
# ---------------------------------------------------------------------------

def adapt_scienceqa(sample: Dict, img: Optional[Image.Image] = None) -> List[ContentBlock]:
    """ScienceQA → generic ContentBlock list."""
    blocks = []

    subj = sample.get("subject", "")
    topic = sample.get("topic", "")
    if subj or topic:
        blocks.append(measure_text(f"[{subj} / {topic}]", SemanticRole.HEADER, 14))

    hint = sample.get("hint", "")
    if hint:
        blocks.append(measure_text(f"Context: {hint}", SemanticRole.CONTEXT, 16))

    if img is not None:
        blocks.append(measure_image(img, SemanticRole.FIGURE))

    q_text = f"Q: {sample['question']}\n"
    for i, c in enumerate(sample.get("choices", [])):
        letter = chr(65 + i)
        q_text += f"  {letter}. {c}\n"
    blocks.append(measure_text(q_text.strip(), SemanticRole.QUESTION, 18))

    lecture = sample.get("lecture", "")
    if lecture:
        blocks.append(measure_text(f"Knowledge: {lecture}", SemanticRole.KNOWLEDGE, 15))

    solution = sample.get("solution", "")
    if solution:
        blocks.append(measure_text(f"Solution: {solution}", SemanticRole.EXPLANATION, 15))

    return blocks


def adapt_scienceqa_knowledge_only(sample: Dict,
                                    img: Optional[Image.Image] = None) -> List[ContentBlock]:
    """ScienceQA → knowledge-only canvas (no question/choices/solution).
    Used for QA reward evaluation: canvas = reference material,
    question asked separately so VLM must comprehend the layout."""
    blocks = []

    subj = sample.get("subject", "")
    topic = sample.get("topic", "")
    if subj or topic:
        blocks.append(measure_text(f"[{subj} / {topic}]", SemanticRole.HEADER, 14))

    hint = sample.get("hint", "")
    if hint:
        blocks.append(measure_text(f"Context: {hint}", SemanticRole.CONTEXT, 16))

    if img is not None:
        blocks.append(measure_image(img, SemanticRole.FIGURE))

    lecture = sample.get("lecture", "")
    if lecture:
        blocks.append(measure_text(f"Knowledge: {lecture}", SemanticRole.KNOWLEDGE, 15))

    return blocks


def adapt_mmqa(sample: Dict, images: Optional[List[Image.Image]] = None,
               tables: Optional[List[Image.Image]] = None) -> List[ContentBlock]:
    """MultimodalQA → generic ContentBlock list.
    Tables and documents are pre-rendered as images."""
    blocks = []

    question = sample.get("question", "")
    if question:
        blocks.append(measure_text(f"Q: {question}", SemanticRole.QUESTION, 18))

    # Text passages
    for ctx in sample.get("text_contexts", []):
        text = ctx if isinstance(ctx, str) else ctx.get("text", "")
        if text:
            blocks.append(measure_text(text, SemanticRole.CONTEXT, 15))

    # Images (photos, diagrams)
    if images:
        for img in images:
            blocks.append(measure_image(img, SemanticRole.FIGURE))

    # Tables rendered as images
    if tables:
        for tbl_img in tables:
            blocks.append(measure_image(tbl_img, SemanticRole.TABLE))

    return blocks


def adapt_infographicvqa(sample: Dict,
                         infographic: Optional[Image.Image] = None) -> List[ContentBlock]:
    """InfographicVQA → generic ContentBlock list."""
    blocks = []

    question = sample.get("question", "")
    if question:
        blocks.append(measure_text(f"Q: {question}", SemanticRole.QUESTION, 18))

    if infographic is not None:
        blocks.append(measure_image(infographic, SemanticRole.DOCUMENT))

    return blocks


def adapt_okvqa(sample: Dict,
                img: Optional[Image.Image] = None) -> List[ContentBlock]:
    """OK-VQA → generic ContentBlock list."""
    blocks = []

    question = sample.get("question", "")
    if question:
        blocks.append(measure_text(f"Q: {question}", SemanticRole.QUESTION, 18))

    if img is not None:
        blocks.append(measure_image(img, SemanticRole.FIGURE))

    return blocks


def adapt_generic(question: str, images: Optional[List[Image.Image]] = None,
                  text_contexts: Optional[List[str]] = None) -> List[ContentBlock]:
    """Generic adapter for any benchmark."""
    blocks = []
    if question:
        blocks.append(measure_text(f"Q: {question}", SemanticRole.QUESTION, 18))
    if images:
        for img in images:
            blocks.append(measure_image(img, SemanticRole.FIGURE))
    if text_contexts:
        for ctx in text_contexts:
            blocks.append(measure_text(ctx, SemanticRole.CONTEXT, 15))
    return blocks


# Adapter registry
BENCHMARK_ADAPTERS = {
    "scienceqa": adapt_scienceqa,
    "mmqa": adapt_mmqa,
    "infographicvqa": adapt_infographicvqa,
    "okvqa": adapt_okvqa,
}


# ---------------------------------------------------------------------------
# Agent prompt builder (benchmark-agnostic)
# ---------------------------------------------------------------------------

def build_agent_prompt(blocks: List[ContentBlock]) -> str:
    """Build the text prompt that the Layout Agent sees.
    Describes blocks by type, role, and dimensions — no benchmark-specific info."""
    lines = ["You are a canvas layout agent. Given content blocks, output the optimal layout as JSON.",
             "", "Content blocks:"]
    for i, b in enumerate(blocks):
        if b.type == BlockType.TEXT:
            lines.append(f"  #{i} [text/{b.role.value}] {b.char_count} chars, font={b.font_size}")
        else:
            lines.append(f"  #{i} [image/{b.role.value}] {b.ref_width}x{b.ref_height}, ar={b.aspect_ratio:.2f}")

    lines.append("")
    lines.append("Available strategies: " + ", ".join(STRATEGY_NAMES))
    lines.append("Output JSON with keys: strategy, width_class(narrow/medium/wide), "
                 "image_scale(small/medium/large), content_priority(question_first/context_first/balanced), "
                 "truncation(none/light/heavy), font_profile(compact/normal/large)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Strategies — self test")
    print("=" * 60)

    # Synthetic blocks
    blocks = [
        measure_text("[science / biology]", SemanticRole.HEADER, 14),
        measure_text("Context: A butterfly was observed in the garden.", SemanticRole.CONTEXT, 16),
        measure_text("Q: What type of insect is this?\n  A. Bee\n  B. Butterfly\n  C. Moth", SemanticRole.QUESTION, 18),
        measure_text("Knowledge: Butterflies belong to the order Lepidoptera. " * 5, SemanticRole.KNOWLEDGE, 15),
        measure_text("Solution: The answer is B because...", SemanticRole.EXPLANATION, 15),
    ]

    # Test with a dummy image
    dummy_img = Image.new("RGB", (300, 200), (100, 150, 200))
    blocks.insert(2, measure_image(dummy_img, SemanticRole.FIGURE))

    print(f"\n{len(blocks)} blocks")
    print(build_agent_prompt(blocks))

    # Generate all 24 candidates
    candidates = generate_all_candidates(blocks)
    print(f"\nGenerated {len(candidates)} candidates:")
    for params, layout in candidates:
        q = compute_quality_score(layout)
        print(f"  {params.config_hash():40s}  {layout.width:4d}x{layout.height:<4d}  "
              f"sq={layout.squareness:.2f}  util={layout.utilization:.2f}  quality={q:.3f}")

    # Render best
    best_params, best_layout = max(candidates, key=lambda x: compute_quality_score(x[1]))
    print(f"\nBest: {best_params.config_hash()}")
    canvas = render_layout(best_layout)
    os.makedirs("/tmp/layout_test", exist_ok=True)
    canvas.save("/tmp/layout_test/best.png")
    print(f"Saved to /tmp/layout_test/best.png ({canvas.size})")
