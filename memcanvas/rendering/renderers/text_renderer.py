"""
TextRenderer - text

text，text。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os


@dataclass
class TextStyle:
    """text"""
    font_path: Optional[str] = None  # text，None text
    font_size: int = 24
    font_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    line_spacing: float = 1.4  # text
    padding: int = 20
    max_chars_per_line: int = 80  # text


class TextRenderer:
    """
    text

    text。
    """

    # text（text）
    FONT_CANDIDATES = [
        # Linux text
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        # text monospace
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]

    def __init__(self, style: Optional[TextStyle] = None):
        self.style = style or TextStyle()
        self._font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """text"""
        # text
        if self.style.font_path and os.path.exists(self.style.font_path):
            try:
                return ImageFont.truetype(self.style.font_path, self.style.font_size)
            except Exception:
                pass

        # text
        for font_path in self.FONT_CANDIDATES:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, self.style.font_size)
                except Exception:
                    continue

        # text PIL text
        try:
            return ImageFont.truetype("DejaVuSans.ttf", self.style.font_size)
        except Exception:
            return ImageFont.load_default()

    def render(
        self,
        text: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        auto_height: bool = True
    ) -> Image.Image:
        """
        text

        Args:
            text: text
            width: text（None text）
            height: text（None text auto_height=True text）
            auto_height: text

        Returns:
            text RGBA text
        """
        # text
        lines = self._wrap_text(text, width)

        # text
        line_height = int(self.style.font_size * self.style.line_spacing)
        text_height = len(lines) * line_height

        if width is None:
            # text
            max_width = 0
            for line in lines:
                bbox = self._font.getbbox(line)
                line_width = bbox[2] - bbox[0] if bbox else 0
                max_width = max(max_width, line_width)
            width = max_width + 2 * self.style.padding

        if height is None and auto_height:
            height = text_height + 2 * self.style.padding

        height = height or 200

        # text
        image = Image.new('RGBA', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        # text
        y = self.style.padding
        for line in lines:
            draw.text(
                (self.style.padding, y),
                line,
                font=self._font,
                fill=self.style.font_color
            )
            y += line_height

        return image

    def _wrap_text(self, text: str, width: Optional[int] = None) -> List[str]:
        """
        text

        Args:
            text: text
            width: text（text）

        Returns:
            text
        """
        # text
        paragraphs = text.split('\n')

        lines = []
        for para in paragraphs:
            if not para.strip():
                lines.append('')
                continue

            # text
            if width:
                # text
                test_char = "text"  # text（text）
                bbox = self._font.getbbox(test_char)
                char_width = (bbox[2] - bbox[0]) if bbox else self.style.font_size
                chars_per_line = max(10, (width - 2 * self.style.padding) // char_width)
            else:
                chars_per_line = self.style.max_chars_per_line

            # text textwrap text
            wrapped = textwrap.wrap(para, width=chars_per_line)
            lines.extend(wrapped if wrapped else [''])

        return lines

    def render_markdown(
        self,
        markdown_text: str,
        width: int = 800
    ) -> Image.Image:
        """
        text Markdown text（text）

        text，text Markdown text。
        """
        # text：text Markdown text
        import re

        text = markdown_text
        # text
        text = re.sub(r'```[\s\S]*?```', '[text]', text)
        # text
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # text/text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # text，text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        return self.render(text, width=width)

    def estimate_height(self, text: str, width: int) -> int:
        """
        text

        Args:
            text: text
            width: text

        Returns:
            text（text）
        """
        lines = self._wrap_text(text, width)
        line_height = int(self.style.font_size * self.style.line_spacing)
        return len(lines) * line_height + 2 * self.style.padding
