"""
TextRenderer - 文本渲染器

将文本内容渲染为图像，支持多种样式和布局。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os


@dataclass
class TextStyle:
    """文本样式配置"""
    font_path: Optional[str] = None  # 字体路径，None 使用默认
    font_size: int = 24
    font_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    line_spacing: float = 1.4  # 行间距倍数
    padding: int = 20
    max_chars_per_line: int = 80  # 每行最大字符数


class TextRenderer:
    """
    文本渲染器

    将纯文本或结构化文本渲染为图像。
    """

    # 尝试加载的字体列表（按优先级）
    FONT_CANDIDATES = [
        # Linux 中文字体
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
        # 通用 monospace
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]

    def __init__(self, style: Optional[TextStyle] = None):
        self.style = style or TextStyle()
        self._font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """加载字体"""
        # 如果指定了字体路径
        if self.style.font_path and os.path.exists(self.style.font_path):
            try:
                return ImageFont.truetype(self.style.font_path, self.style.font_size)
            except Exception:
                pass

        # 尝试候选字体
        for font_path in self.FONT_CANDIDATES:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, self.style.font_size)
                except Exception:
                    continue

        # 使用 PIL 默认字体
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
        渲染文本为图像

        Args:
            text: 要渲染的文本
            width: 图像宽度（None 则自动计算）
            height: 图像高度（None 且 auto_height=True 则自动计算）
            auto_height: 是否自动计算高度

        Returns:
            渲染后的 RGBA 图像
        """
        # 处理文本换行
        lines = self._wrap_text(text, width)

        # 计算尺寸
        line_height = int(self.style.font_size * self.style.line_spacing)
        text_height = len(lines) * line_height

        if width is None:
            # 计算最大行宽
            max_width = 0
            for line in lines:
                bbox = self._font.getbbox(line)
                line_width = bbox[2] - bbox[0] if bbox else 0
                max_width = max(max_width, line_width)
            width = max_width + 2 * self.style.padding

        if height is None and auto_height:
            height = text_height + 2 * self.style.padding

        height = height or 200

        # 创建图像
        image = Image.new('RGBA', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        # 绘制文本
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
        文本换行处理

        Args:
            text: 原始文本
            width: 目标宽度（用于计算每行字符数）

        Returns:
            换行后的文本行列表
        """
        # 先按原有换行符分割
        paragraphs = text.split('\n')

        lines = []
        for para in paragraphs:
            if not para.strip():
                lines.append('')
                continue

            # 根据宽度计算每行字符数
            if width:
                # 估算每个字符的平均宽度
                test_char = "中"  # 用中文字符估算（较宽）
                bbox = self._font.getbbox(test_char)
                char_width = (bbox[2] - bbox[0]) if bbox else self.style.font_size
                chars_per_line = max(10, (width - 2 * self.style.padding) // char_width)
            else:
                chars_per_line = self.style.max_chars_per_line

            # 使用 textwrap 处理
            wrapped = textwrap.wrap(para, width=chars_per_line)
            lines.extend(wrapped if wrapped else [''])

        return lines

    def render_markdown(
        self,
        markdown_text: str,
        width: int = 800
    ) -> Image.Image:
        """
        渲染 Markdown 文本（简化版）

        目前只处理基本格式，复杂 Markdown 建议用专门的库渲染后截图。
        """
        # 简单处理：移除 Markdown 标记
        import re

        text = markdown_text
        # 移除代码块标记
        text = re.sub(r'```[\s\S]*?```', '[代码块]', text)
        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 移除加粗/斜体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # 移除链接，保留文字
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 移除标题标记
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        return self.render(text, width=width)

    def estimate_height(self, text: str, width: int) -> int:
        """
        估算渲染所需高度

        Args:
            text: 要渲染的文本
            width: 目标宽度

        Returns:
            估算的高度（像素）
        """
        lines = self._wrap_text(text, width)
        line_height = int(self.style.font_size * self.style.line_spacing)
        return len(lines) * line_height + 2 * self.style.padding
