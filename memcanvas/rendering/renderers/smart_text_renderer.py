"""
SmartTextRenderer - text

text：
1. text
2. text
3. text（text）
4. text
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import os
import re
import math


@dataclass
class SmartTextStyle:
    """text"""
    # text
    font_path: Optional[str] = None
    font_size: int = 24
    min_font_size: int = 12      # text（text）
    max_font_size: int = 48      # text
    font_color: Tuple[int, int, int] = (0, 0, 0)

    # text
    line_spacing: float = 1.3    # text
    padding: int = 20            # text
    paragraph_spacing: int = 10  # text

    # text
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # text
    auto_fit: bool = True        # text
    allow_pagination: bool = True # text


@dataclass
class RenderResult:
    """text"""
    images: List[Image.Image]    # text（text）
    total_pages: int
    font_size_used: int
    chars_rendered: int
    chars_total: int
    is_truncated: bool
    overflow_text: Optional[str] = None  # text


class SmartTextRenderer:
    """
    text

    text：
    1. text
    2. text
    3. text
    4. text
    """

    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def __init__(self, style: Optional[SmartTextStyle] = None):
        self.style = style or SmartTextStyle()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path = self._find_font()

    def _find_font(self) -> Optional[str]:
        """text"""
        if self.style.font_path and os.path.exists(self.style.font_path):
            return self.style.font_path

        for font_path in self.FONT_CANDIDATES:
            if os.path.exists(font_path):
                return font_path

        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """text（text）"""
        if size not in self._font_cache:
            if self._font_path:
                try:
                    self._font_cache[size] = ImageFont.truetype(self._font_path, size)
                except:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def estimate_capacity(
        self,
        width: int,
        height: int,
        font_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        text

        Args:
            width: text
            height: text
            font_size: text（Nonetext）

        Returns:
            text
        """
        font_size = font_size or self.style.font_size
        font = self._get_font(font_size)

        # text
        usable_width = width - 2 * self.style.padding
        usable_height = height - 2 * self.style.padding

        # text
        line_height = int(font_size * self.style.line_spacing)

        # text
        max_lines = usable_height // line_height

        # text（text）
        test_bbox = font.getbbox("text")
        char_width = test_bbox[2] - test_bbox[0] if test_bbox else font_size
        chars_per_line = usable_width // char_width

        # text
        total_chars = max_lines * chars_per_line

        return {
            "font_size": font_size,
            "line_height": line_height,
            "max_lines": max_lines,
            "chars_per_line": chars_per_line,
            "total_chars_estimate": total_chars,
            "usable_width": usable_width,
            "usable_height": usable_height
        }

    def check_fit(
        self,
        text: str,
        width: int,
        height: int,
        font_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        text

        Returns:
            {
                "fits": bool,
                "text_lines": int,
                "max_lines": int,
                "overflow_lines": int,
                "recommended_font_size": int
            }
        """
        font_size = font_size or self.style.font_size

        # text
        lines = self._smart_wrap(text, width, font_size)
        text_lines = len(lines)

        # text
        capacity = self.estimate_capacity(width, height, font_size)
        max_lines = capacity["max_lines"]

        fits = text_lines <= max_lines
        overflow = max(0, text_lines - max_lines)

        # text（textfit）
        recommended_size = font_size
        if not fits and self.style.auto_fit:
            recommended_size = self._calculate_optimal_font_size(
                text, width, height
            )

        return {
            "fits": fits,
            "text_lines": text_lines,
            "max_lines": max_lines,
            "overflow_lines": overflow,
            "current_font_size": font_size,
            "recommended_font_size": recommended_size
        }

    def _calculate_optimal_font_size(
        self,
        text: str,
        width: int,
        height: int
    ) -> int:
        """text"""
        min_size = self.style.min_font_size
        max_size = self.style.max_font_size
        optimal_size = min_size

        while min_size <= max_size:
            mid_size = (min_size + max_size) // 2

            lines = self._smart_wrap(text, width, mid_size)
            capacity = self.estimate_capacity(width, height, mid_size)

            if len(lines) <= capacity["max_lines"]:
                optimal_size = mid_size
                min_size = mid_size + 1
            else:
                max_size = mid_size - 1

        return optimal_size

    def _smart_wrap(
        self,
        text: str,
        width: int,
        font_size: int
    ) -> List[str]:
        """
        text（text）

        texttextwraptext，text，text。
        """
        font = self._get_font(font_size)
        usable_width = width - 2 * self.style.padding

        lines = []
        paragraphs = text.split('\n')

        for para in paragraphs:
            if not para.strip():
                lines.append('')
                continue

            # text
            current_line = ''
            current_width = 0

            for char in para:
                char_bbox = font.getbbox(char)
                char_width = char_bbox[2] - char_bbox[0] if char_bbox else font_size // 2

                if current_width + char_width <= usable_width:
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

    def render(
        self,
        text: str,
        width: int = 1024,
        height: int = 1024,
        font_size: Optional[int] = None
    ) -> RenderResult:
        """
        text

        text：
        1. auto_fit=True: text
        2. allow_pagination=True: text
        3. text: text

        Args:
            text: text
            width: text
            height: text
            font_size: text（Nonetext）

        Returns:
            RenderResult
        """
        original_text = text

        # text
        if font_size is None:
            font_size = self.style.font_size

        # text
        check = self.check_fit(text, width, height, font_size)

        if not check["fits"]:
            if self.style.auto_fit:
                font_size = check["recommended_font_size"]
                check = self.check_fit(text, width, height, font_size)

        # text
        font = self._get_font(font_size)
        line_height = int(font_size * self.style.line_spacing)

        # text
        all_lines = self._smart_wrap(text, width, font_size)
        capacity = self.estimate_capacity(width, height, font_size)
        max_lines_per_page = capacity["max_lines"]

        # text
        images = []
        total_lines = len(all_lines)
        chars_rendered = 0

        if self.style.allow_pagination:
            # text
            for page_start in range(0, total_lines, max_lines_per_page):
                page_lines = all_lines[page_start:page_start + max_lines_per_page]
                img = self._render_page(page_lines, width, height, font, line_height)
                images.append(img)
                chars_rendered += sum(len(line) for line in page_lines)
        else:
            # text（text）
            page_lines = all_lines[:max_lines_per_page]
            img = self._render_page(page_lines, width, height, font, line_height)
            images.append(img)
            chars_rendered = sum(len(line) for line in page_lines)

        # text
        total_chars = len(text.replace('\n', ''))
        is_truncated = chars_rendered < total_chars and not self.style.allow_pagination

        overflow_text = None
        if is_truncated:
            # text
            rendered_text = '\n'.join(all_lines[:max_lines_per_page])
            overflow_start = len(rendered_text)
            overflow_text = original_text[overflow_start:] if overflow_start < len(original_text) else None

        return RenderResult(
            images=images,
            total_pages=len(images),
            font_size_used=font_size,
            chars_rendered=chars_rendered,
            chars_total=total_chars,
            is_truncated=is_truncated,
            overflow_text=overflow_text
        )

    def _render_page(
        self,
        lines: List[str],
        width: int,
        height: int,
        font: ImageFont.FreeTypeFont,
        line_height: int
    ) -> Image.Image:
        """text"""
        image = Image.new('RGB', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        y = self.style.padding
        for line in lines:
            draw.text(
                (self.style.padding, y),
                line,
                font=font,
                fill=self.style.font_color
            )
            y += line_height

        return image

    def render_with_info(
        self,
        text: str,
        width: int = 1024,
        height: int = 1024
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        text

        Returns:
            (image, info_dict)
        """
        result = self.render(text, width, height)

        info = {
            "total_pages": result.total_pages,
            "font_size": result.font_size_used,
            "chars_rendered": result.chars_rendered,
            "chars_total": result.chars_total,
            "is_truncated": result.is_truncated,
            "information_preserved": result.chars_rendered / max(result.chars_total, 1) * 100
        }

        # text，text
        if result.total_pages > 1:
            total_height = height * result.total_pages
            combined = Image.new('RGB', (width, total_height), self.style.background_color)
            for i, img in enumerate(result.images):
                combined.paste(img, (0, i * height))
            return combined, info

        return result.images[0], info


def demo():
    """text"""

    # text
    renderer = SmartTextRenderer(SmartTextStyle(
        font_size=24,
        auto_fit=True,
        allow_pagination=True
    ))

    # text
    long_text = """text，text。

text，text：
1. text（textauto_fit）
2. text，text（textallow_pagination）
3. text，text

text：
- text
- text
- text

text：
- text：12px - 48px
- text：1.3text
- text：20px

English text is also supported, and the line breaking algorithm handles mixed Chinese-English text properly by calculating the actual pixel width of each character.

text。"""

    # text
    print("="*50)
    print("1. text")
    print("="*50)

    capacity = renderer.estimate_capacity(800, 600, 24)
    print(f"text 800x600, text 24px:")
    print(f"  text: {capacity['max_lines']}")
    print(f"  text: {capacity['chars_per_line']}")
    print(f"  text: {capacity['total_chars_estimate']} text")

    # textfit
    print("\n" + "="*50)
    print("2. text")
    print("="*50)

    fit_check = renderer.check_fit(long_text, 800, 600)
    print(f"text: {fit_check['text_lines']}")
    print(f"text: {fit_check['max_lines']}")
    print(f"text: {fit_check['fits']}")
    print(f"text: {fit_check['recommended_font_size']}px")

    # text
    print("\n" + "="*50)
    print("3. text")
    print("="*50)

    result = renderer.render(long_text, 800, 600)
    print(f"text: {result.total_pages}")
    print(f"text: {result.font_size_used}px")
    print(f"text: {result.chars_rendered}")
    print(f"text: {result.chars_total}")
    print(f"text: {result.is_truncated}")
    print(f"text: {result.chars_rendered/result.chars_total*100:.1f}%")

    # text
    for i, img in enumerate(result.images):
        img.save(f"smart_text_page_{i+1}.png")
        print(f"text: smart_text_page_{i+1}.png")


if __name__ == "__main__":
    demo()
