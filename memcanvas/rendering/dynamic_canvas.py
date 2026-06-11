"""
DynamicCanvas - text

text：
- text patch text
- text patch text
- text，text
- text patches text vision tokens

text：
1. text - text patch text
2. text - text patches
3. text vision token text
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
import os


class ContentType(Enum):
    """text"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    SEPARATOR = "separator"


@dataclass
class ContentBlock:
    """text"""
    type: ContentType
    data: Any
    # text
    width: int = 0
    height: int = 0
    # text（text）
    rendered: Optional[Image.Image] = None


@dataclass
class DynamicCanvasConfig:
    """text"""
    # Patch text（text vision encoder text）
    patch_size: int = 640

    # text
    padding: int = 20
    # text
    content_gap: int = 15

    # text
    font_size: int = 20
    font_color: Tuple[int, int, int] = (0, 0, 0)
    line_spacing: float = 1.3

    # text
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # text patch text（text）
    show_patch_boundary: bool = False


@dataclass
class Patch:
    """text Patch"""
    index: int
    image: Image.Image
    # text patch text
    content_summary: List[str] = field(default_factory=list)
    # text
    is_full: bool = False
    # text
    remaining_height: int = 0


class DynamicCanvas:
    """
    text

    text：
    1. text patch
    2. text，text patch text
    3. text，text patch
    4. text，text patch text
    5. text patch text

    text：
    ```python
    canvas = DynamicCanvas(DynamicCanvasConfig(patch_size=640))

    # text
    canvas.add_text("text", font_size=32)
    canvas.add_text("text...")
    canvas.add_image(some_image)
    canvas.add_table(table_data)

    # text patches
    patches = canvas.get_patches()

    # text vision token text
    for patch in patches:
        tokens = extractor.extract_single(patch.image)
    ```
    """

    # text
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def __init__(self, config: Optional[DynamicCanvasConfig] = None):
        self.config = config or DynamicCanvasConfig()

        # Patch text
        self.patches: List[Patch] = []

        # text patch text
        self._current_patch_idx: int = -1
        self._cursor_y: int = 0  # text patch text y text

        # text
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path = self._find_font()

        # text
        self._total_content_blocks = 0

        # text patch
        self._create_new_patch()

    def _find_font(self) -> Optional[str]:
        """text"""
        for fp in self.FONT_CANDIDATES:
            if os.path.exists(fp):
                return fp
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """text"""
        if size not in self._font_cache:
            if self._font_path:
                try:
                    self._font_cache[size] = ImageFont.truetype(self._font_path, size)
                except:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _create_new_patch(self) -> Patch:
        """text patch"""
        # text patch text（text get_compact_image text）
        if self.patches:
            self.current_patch.remaining_height = self.available_height

        size = self.config.patch_size
        image = Image.new('RGB', (size, size), self.config.background_color)

        # text
        if self.config.show_patch_boundary:
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, size-1, size-1], outline=(200, 200, 200), width=2)
            # text patch text
            font = self._get_font(12)
            draw.text((5, 5), f"P{len(self.patches)}", fill=(150, 150, 150), font=font)

        patch = Patch(
            index=len(self.patches),
            image=image,
            remaining_height=size - 2 * self.config.padding
        )
        self.patches.append(patch)
        self._current_patch_idx = len(self.patches) - 1
        self._cursor_y = self.config.padding

        return patch

    @property
    def current_patch(self) -> Patch:
        """text patch"""
        return self.patches[self._current_patch_idx]

    @property
    def available_height(self) -> int:
        """text patch text"""
        return self.config.patch_size - self._cursor_y - self.config.padding

    @property
    def content_width(self) -> int:
        """text"""
        return self.config.patch_size - 2 * self.config.padding

    # ==================== text ====================

    def add_text(
        self,
        text: str,
        font_size: Optional[int] = None,
        font_color: Optional[Tuple[int, int, int]] = None,
        bold: bool = False
    ) -> int:
        """
        text

        Args:
            text: text
            font_size: text
            font_color: text
            bold: text（text，text）

        Returns:
            text patch text
        """
        font_size = font_size or self.config.font_size
        font_color = font_color or self.config.font_color
        font = self._get_font(font_size)

        # text
        line_height = int(font_size * self.config.line_spacing)

        # text
        lines = self._wrap_text(text, font)

        patches_used = 0
        start_patch = self._current_patch_idx

        for line in lines:
            # text patch text
            if self.available_height < line_height:
                # text patch text
                self.current_patch.is_full = True
                self.current_patch.remaining_height = self.available_height
                # text patch
                self._create_new_patch()
                patches_used += 1

            # text
            draw = ImageDraw.Draw(self.current_patch.image)
            draw.text(
                (self.config.padding, self._cursor_y),
                line,
                font=font,
                fill=font_color
            )
            self._cursor_y += line_height

        # text
        self._cursor_y += self.config.content_gap

        # text
        self._total_content_blocks += 1
        self.current_patch.content_summary.append(f"text:{len(text)}chars")

        return self._current_patch_idx - start_patch + 1

    def add_image(
        self,
        image: Image.Image,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        caption: Optional[str] = None
    ) -> int:
        """
        text

        Args:
            image: PIL Image
            max_width: text（text）
            max_height: text（text patch text）
            caption: text

        Returns:
            text patch text
        """
        max_width = max_width or self.content_width
        max_height = max_height or (self.config.patch_size // 2)

        # text
        img_w, img_h = image.size
        scale = min(max_width / img_w, max_height / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if scale < 1.0:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # text
        total_height = new_h
        if caption:
            caption_height = int(self.config.font_size * self.config.line_spacing)
            total_height += caption_height + 5

        # text patch
        patches_used = 0
        start_patch = self._current_patch_idx

        if self.available_height < total_height:
            self.current_patch.is_full = True
            self._create_new_patch()
            patches_used += 1

        # text
        x = self.config.padding + (self.content_width - new_w) // 2  # text
        self.current_patch.image.paste(image, (x, self._cursor_y))
        self._cursor_y += new_h

        # text
        if caption:
            draw = ImageDraw.Draw(self.current_patch.image)
            font = self._get_font(self.config.font_size - 2)
            draw.text(
                (self.config.padding, self._cursor_y + 5),
                caption,
                font=font,
                fill=(100, 100, 100)
            )
            self._cursor_y += caption_height + 5

        self._cursor_y += self.config.content_gap
        self._total_content_blocks += 1
        self.current_patch.content_summary.append(f"image:{new_w}x{new_h}")

        return self._current_patch_idx - start_patch + 1

    def _truncate_text_to_width(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> str:
        """text，text"""
        if max_width <= 0:
            return ""
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return text

        # text
        ellipsis = ".."
        ell_bbox = font.getbbox(ellipsis)
        ell_width = ell_bbox[2] - ell_bbox[0]
        target_width = max_width - ell_width

        if target_width <= 0:
            return ""

        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            bbox = font.getbbox(text[:mid])
            w = bbox[2] - bbox[0]
            if w <= target_width:
                lo = mid
            else:
                hi = mid - 1

        return text[:lo] + ellipsis if lo < len(text) else text

    def add_table(
        self,
        data: List[List[str]],
        headers: Optional[List[str]] = None,
        cell_padding: int = 8
    ) -> int:
        """
        text

        Args:
            data: text [[row1], [row2], ...]
            headers: text
            cell_padding: text

        Returns:
            text patch text
        """
        if not data:
            return 0

        font = self._get_font(self.config.font_size - 2)

        # text
        all_rows = ([headers] if headers else []) + data
        num_cols = max(len(row) for row in all_rows)

        col_widths = [0] * num_cols
        for row in all_rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    bbox = font.getbbox(str(cell))
                    cell_width = (bbox[2] - bbox[0]) + 2 * cell_padding
                    col_widths[i] = max(col_widths[i], cell_width)

        # text，text
        total_width = sum(col_widths)
        if total_width > self.content_width:
            scale = self.content_width / total_width
            col_widths = [max(int(w * scale), 2 * cell_padding + 10) for w in col_widths]
            # text
            while sum(col_widths) > self.content_width:
                widest = max(range(num_cols), key=lambda i: col_widths[i])
                col_widths[widest] -= 1

        # text
        row_height = int(self.config.font_size * self.config.line_spacing) + 2 * cell_padding

        # text
        patches_used = 0
        start_patch = self._current_patch_idx

        rows_to_render = ([headers] if headers else []) + data
        is_header = True if headers else False

        for row_idx, row in enumerate(rows_to_render):
            # text patch
            if self.available_height < row_height:
                self.current_patch.is_full = True
                self._create_new_patch()
                patches_used += 1

            draw = ImageDraw.Draw(self.current_patch.image)

            # text
            x = self.config.padding
            for col_idx, cell in enumerate(row):
                if col_idx >= len(col_widths):
                    break

                cell_width = col_widths[col_idx]

                # text
                draw.rectangle(
                    [x, self._cursor_y, x + cell_width, self._cursor_y + row_height],
                    outline=(200, 200, 200),
                    fill=(240, 240, 240) if (is_header and row_idx == 0) else None
                )

                # text，text
                avail_text_width = cell_width - 2 * cell_padding
                cell_text = self._truncate_text_to_width(
                    str(cell), font, avail_text_width
                )
                draw.text(
                    (x + cell_padding, self._cursor_y + cell_padding),
                    cell_text,
                    font=font,
                    fill=self.config.font_color
                )

                x += cell_width

            self._cursor_y += row_height
            is_header = False

        self._cursor_y += self.config.content_gap
        self._total_content_blocks += 1
        self.current_patch.content_summary.append(f"table:{len(data)}rows")

        return self._current_patch_idx - start_patch + 1

    def add_html(
        self,
        content: str,
        content_type: str = "markdown",
        max_height: Optional[int] = None,
    ) -> int:
        """
        text HTML/Markdown text

        text HtmlRenderer text。
        Playwright text fallback text add_text()。

        Args:
            content: Markdown、HTML text
            content_type: "markdown", "html", text "text"
            max_height: text

        Returns:
            text patch text
        """
        try:
            from .renderers.html_renderer import HtmlRenderer, HtmlStyle

            style = HtmlStyle(viewport_width=self.content_width)
            renderer = HtmlRenderer(style=style)
            img = renderer.render(content, content_type)

            # text content_width
            if img.width != self.content_width:
                scale = self.content_width / img.width
                new_h = int(img.height * scale)
                img = img.resize(
                    (self.content_width, new_h), Image.Resampling.LANCZOS
                )

            return self.add_image(
                img,
                max_width=self.content_width,
                max_height=max_height,
            )
        except (ImportError, Exception):
            # Fallback: text
            return self.add_text(content)

    def add_tall_image(
        self,
        image: Image.Image,
        max_sections: int = 3,
        overlap: int = 50,
        gap: int = 6,
        label_height: int = 22,
    ) -> int:
        """
        text，text

        text > 2.0 text，text，
        text，text。
        text，text。

        text：596×5107 text → 3 text，text 196×559，
        text 600×579，text 640×640 patch。

        Args:
            image: PIL Image（text）
            max_sections: text（text token text）
            overlap: text（text）
            gap: text（text）
            label_height: text（text）

        Returns:
            text patch text
        """
        img_w, img_h = image.size
        aspect_ratio = img_h / img_w if img_w > 0 else 1

        # text，text
        if aspect_ratio <= 2.0:
            return self.add_image(image)

        # text（2~max_sections）
        n_sections = min(max(2, round(aspect_ratio / 2.5)), max_sections)

        # text
        section_h = (img_h + (n_sections - 1) * overlap) // n_sections
        effective_step = section_h - overlap

        sections = []
        for i in range(n_sections):
            y_start = i * effective_step
            y_end = min(y_start + section_h, img_h)
            sections.append(image.crop((0, y_start, img_w, y_end)))

        # text：text
        avail_w = self.content_width - (n_sections - 1) * gap
        per_w = avail_w // n_sections
        scale = per_w / img_w
        per_h = int(section_h * scale)

        # text（text + text）
        total_w = n_sections * per_w + (n_sections - 1) * gap
        total_h = per_h + label_height
        composed = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        font = self._get_font(11)
        draw = ImageDraw.Draw(composed)

        for i, sec in enumerate(sections):
            # text
            sec_h_actual = sec.size[1]
            scaled_h = int(sec_h_actual * scale)
            sec_resized = sec.resize((per_w, scaled_h), Image.Resampling.LANCZOS)

            # text
            x = i * (per_w + gap)
            composed.paste(sec_resized, (x, 0))

            # text
            label = f"{i + 1}/{n_sections}"
            bbox = font.getbbox(label)
            lw = bbox[2] - bbox[0]
            lx = x + (per_w - lw) // 2
            ly = per_h + 3
            draw.text((lx, ly), label, fill=(120, 120, 120), font=font)

        # text
        return self.add_image(
            composed,
            max_width=self.content_width,
            max_height=self.config.patch_size - 2 * self.config.padding,
        )

    def add_separator(self, style: str = "line") -> int:
        """text"""
        height = 20 if style == "line" else 40

        if self.available_height < height:
            self.current_patch.is_full = True
            self._create_new_patch()

        draw = ImageDraw.Draw(self.current_patch.image)

        if style == "line":
            y = self._cursor_y + 10
            draw.line(
                [(self.config.padding, y), (self.config.patch_size - self.config.padding, y)],
                fill=(200, 200, 200),
                width=1
            )
        elif style == "dots":
            y = self._cursor_y + 20
            for x in range(self.config.padding, self.config.patch_size - self.config.padding, 10):
                draw.ellipse([x, y-2, x+4, y+2], fill=(180, 180, 180))

        self._cursor_y += height
        return 0

    # ==================== text ====================

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont) -> List[str]:
        """text（text）"""
        lines = []
        paragraphs = text.split('\n')

        for para in paragraphs:
            if not para.strip():
                lines.append('')
                continue

            current_line = ''
            current_width = 0

            for char in para:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0] if bbox else self.config.font_size // 2

                if current_width + char_width <= self.content_width:
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

    # ==================== text ====================

    def get_patches(self) -> List[Patch]:
        """text patches"""
        # text patch text
        self.current_patch.remaining_height = self.available_height
        return self.patches

    def get_images(self) -> List[Image.Image]:
        """text patch text"""
        return [p.image for p in self.patches]

    def get_combined_image(self, direction: str = "vertical") -> Image.Image:
        """
        text

        Args:
            direction: "vertical" text "horizontal"

        Returns:
            text
        """
        if not self.patches:
            return Image.new('RGB', (self.config.patch_size, self.config.patch_size), self.config.background_color)

        images = self.get_images()

        if direction == "vertical":
            total_height = sum(img.height for img in images)
            combined = Image.new('RGB', (self.config.patch_size, total_height), self.config.background_color)
            y = 0
            for img in images:
                combined.paste(img, (0, y))
                y += img.height
        else:
            total_width = sum(img.width for img in images)
            combined = Image.new('RGB', (total_width, self.config.patch_size), self.config.background_color)
            x = 0
            for img in images:
                combined.paste(img, (x, 0))
                x += img.width

        return combined

    def get_compact_image(self) -> Image.Image:
        """
        text（text）

        text patch_size，text。
        text patch text，text patch text。

        Returns:
            text RGB text
        """
        if not self.patches:
            return Image.new(
                'RGB', (self.config.patch_size, self.config.padding * 2),
                self.config.background_color,
            )

        width = self.config.patch_size

        # text patch text
        cropped = []
        for i, patch in enumerate(self.patches):
            if i == self._current_patch_idx:
                # text patch：text + padding
                h = self._cursor_y + self.config.padding
            elif patch.is_full:
                # text patch：text
                h = self.config.patch_size - patch.remaining_height
            else:
                h = self.config.patch_size
            h = max(h, self.config.padding * 2)  # text
            cropped.append(patch.image.crop((0, 0, width, h)))

        total_h = sum(c.height for c in cropped)
        combined = Image.new('RGB', (width, total_h), self.config.background_color)
        y = 0
        for c in cropped:
            combined.paste(c, (0, y))
            y += c.height

        return combined

    def get_statistics(self) -> Dict[str, Any]:
        """text"""
        total_used = sum(
            self.config.patch_size - p.remaining_height
            for p in self.patches
        )
        total_capacity = len(self.patches) * self.config.patch_size

        return {
            "num_patches": len(self.patches),
            "patch_size": self.config.patch_size,
            "total_content_blocks": self._total_content_blocks,
            "space_utilization": total_used / total_capacity * 100 if total_capacity > 0 else 0,
            "patches_summary": [
                {
                    "index": p.index,
                    "is_full": p.is_full,
                    "remaining_height": p.remaining_height,
                    "content": p.content_summary
                }
                for p in self.patches
            ]
        }

    def clear(self):
        """text"""
        self.patches = []
        self._current_patch_idx = -1
        self._cursor_y = 0
        self._total_content_blocks = 0
        self._create_new_patch()


def create_dynamic_canvas(
    patch_size: int = 640,
    font_size: int = 20,
    padding: int = 20
) -> DynamicCanvas:
    """
    text

    Args:
        patch_size: Patch text（text vision encoder text）
        font_size: text
        padding: text

    Returns:
        DynamicCanvas text
    """
    config = DynamicCanvasConfig(
        patch_size=patch_size,
        font_size=font_size,
        padding=padding
    )
    return DynamicCanvas(config)
