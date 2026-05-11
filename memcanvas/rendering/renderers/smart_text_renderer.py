"""
SmartTextRenderer - 智能文本渲染器

解决信息丢失问题：
1. 自动调整字体大小以适应画布
2. 文本溢出时自动分页
3. 支持精确的字符级换行（中英文混合）
4. 提供渲染前的容量检查
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import os
import re
import math


@dataclass
class SmartTextStyle:
    """智能文本样式配置"""
    # 字体
    font_path: Optional[str] = None
    font_size: int = 24
    min_font_size: int = 12      # 最小字体（自动调整时）
    max_font_size: int = 48      # 最大字体
    font_color: Tuple[int, int, int] = (0, 0, 0)

    # 布局
    line_spacing: float = 1.3    # 行间距倍数
    padding: int = 20            # 边距
    paragraph_spacing: int = 10  # 段落间距

    # 背景
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # 自动调整选项
    auto_fit: bool = True        # 是否自动调整字体大小
    allow_pagination: bool = True # 是否允许分页


@dataclass
class RenderResult:
    """渲染结果"""
    images: List[Image.Image]    # 渲染的图像列表（可能多页）
    total_pages: int
    font_size_used: int
    chars_rendered: int
    chars_total: int
    is_truncated: bool
    overflow_text: Optional[str] = None  # 溢出的文本


class SmartTextRenderer:
    """
    智能文本渲染器

    特点：
    1. 自动调整字体大小以最大化信息保留
    2. 精确的中英文混合换行
    3. 文本溢出时支持分页
    4. 渲染前容量检查
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
        """查找可用字体"""
        if self.style.font_path and os.path.exists(self.style.font_path):
            return self.style.font_path

        for font_path in self.FONT_CANDIDATES:
            if os.path.exists(font_path):
                return font_path

        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """获取指定大小的字体（带缓存）"""
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
        估算画布容量

        Args:
            width: 画布宽度
            height: 画布高度
            font_size: 字体大小（None使用默认）

        Returns:
            容量信息字典
        """
        font_size = font_size or self.style.font_size
        font = self._get_font(font_size)

        # 计算可用区域
        usable_width = width - 2 * self.style.padding
        usable_height = height - 2 * self.style.padding

        # 计算每行高度
        line_height = int(font_size * self.style.line_spacing)

        # 计算行数
        max_lines = usable_height // line_height

        # 估算每行字符数（用中文字符估算）
        test_bbox = font.getbbox("中")
        char_width = test_bbox[2] - test_bbox[0] if test_bbox else font_size
        chars_per_line = usable_width // char_width

        # 总容量
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
        检查文本是否能完整放入画布

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

        # 计算换行后的行数
        lines = self._smart_wrap(text, width, font_size)
        text_lines = len(lines)

        # 计算容量
        capacity = self.estimate_capacity(width, height, font_size)
        max_lines = capacity["max_lines"]

        fits = text_lines <= max_lines
        overflow = max(0, text_lines - max_lines)

        # 计算推荐字体大小（如果不fit）
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
        """二分查找最优字体大小"""
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
        智能换行（支持中英文混合）

        与textwrap不同，这里按像素宽度换行，更精确。
        """
        font = self._get_font(font_size)
        usable_width = width - 2 * self.style.padding

        lines = []
        paragraphs = text.split('\n')

        for para in paragraphs:
            if not para.strip():
                lines.append('')
                continue

            # 按像素宽度换行
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
        渲染文本

        如果文本太长：
        1. auto_fit=True: 自动减小字体
        2. allow_pagination=True: 分页渲染
        3. 都不允许: 截断并返回溢出文本

        Args:
            text: 要渲染的文本
            width: 画布宽度
            height: 画布高度
            font_size: 字体大小（None则自动）

        Returns:
            RenderResult
        """
        original_text = text

        # 确定字体大小
        if font_size is None:
            font_size = self.style.font_size

        # 检查是否需要调整
        check = self.check_fit(text, width, height, font_size)

        if not check["fits"]:
            if self.style.auto_fit:
                font_size = check["recommended_font_size"]
                check = self.check_fit(text, width, height, font_size)

        # 获取字体
        font = self._get_font(font_size)
        line_height = int(font_size * self.style.line_spacing)

        # 换行
        all_lines = self._smart_wrap(text, width, font_size)
        capacity = self.estimate_capacity(width, height, font_size)
        max_lines_per_page = capacity["max_lines"]

        # 分页
        images = []
        total_lines = len(all_lines)
        chars_rendered = 0

        if self.style.allow_pagination:
            # 分页模式
            for page_start in range(0, total_lines, max_lines_per_page):
                page_lines = all_lines[page_start:page_start + max_lines_per_page]
                img = self._render_page(page_lines, width, height, font, line_height)
                images.append(img)
                chars_rendered += sum(len(line) for line in page_lines)
        else:
            # 单页模式（可能截断）
            page_lines = all_lines[:max_lines_per_page]
            img = self._render_page(page_lines, width, height, font, line_height)
            images.append(img)
            chars_rendered = sum(len(line) for line in page_lines)

        # 计算溢出
        total_chars = len(text.replace('\n', ''))
        is_truncated = chars_rendered < total_chars and not self.style.allow_pagination

        overflow_text = None
        if is_truncated:
            # 找出溢出的文本
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
        """渲染单页"""
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
        渲染并返回详细信息

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

        # 如果有多页，合并成一张长图
        if result.total_pages > 1:
            total_height = height * result.total_pages
            combined = Image.new('RGB', (width, total_height), self.style.background_color)
            for i, img in enumerate(result.images):
                combined.paste(img, (0, i * height))
            return combined, info

        return result.images[0], info


def demo():
    """演示智能文本渲染"""

    # 创建渲染器
    renderer = SmartTextRenderer(SmartTextStyle(
        font_size=24,
        auto_fit=True,
        allow_pagination=True
    ))

    # 测试文本
    long_text = """这是一段很长的测试文本，用于演示智能文本渲染器的功能。

当文本内容超过画布容量时，渲染器会：
1. 首先尝试自动调整字体大小（如果启用auto_fit）
2. 如果字体已经是最小值还是放不下，则分页渲染（如果启用allow_pagination）
3. 如果都不允许，则截断并返回溢出的文本

这种设计确保了：
- 信息尽可能不丢失
- 可以追踪哪些内容被渲染了
- 支持中英文混合的精确换行

下面是一些技术细节：
- 字体大小范围：12px - 48px
- 行间距：1.3倍
- 边距：20px

English text is also supported, and the line breaking algorithm handles mixed Chinese-English text properly by calculating the actual pixel width of each character.

这是最后一段。"""

    # 检查容量
    print("="*50)
    print("1. 容量检查")
    print("="*50)

    capacity = renderer.estimate_capacity(800, 600, 24)
    print(f"画布 800x600, 字体 24px:")
    print(f"  最大行数: {capacity['max_lines']}")
    print(f"  每行字符: {capacity['chars_per_line']}")
    print(f"  总容量估计: {capacity['total_chars_estimate']} 字符")

    # 检查是否fit
    print("\n" + "="*50)
    print("2. 文本适配检查")
    print("="*50)

    fit_check = renderer.check_fit(long_text, 800, 600)
    print(f"文本行数: {fit_check['text_lines']}")
    print(f"最大行数: {fit_check['max_lines']}")
    print(f"是否适配: {fit_check['fits']}")
    print(f"推荐字体: {fit_check['recommended_font_size']}px")

    # 渲染
    print("\n" + "="*50)
    print("3. 渲染结果")
    print("="*50)

    result = renderer.render(long_text, 800, 600)
    print(f"总页数: {result.total_pages}")
    print(f"使用字体: {result.font_size_used}px")
    print(f"已渲染字符: {result.chars_rendered}")
    print(f"总字符数: {result.chars_total}")
    print(f"是否截断: {result.is_truncated}")
    print(f"信息保留率: {result.chars_rendered/result.chars_total*100:.1f}%")

    # 保存示例
    for i, img in enumerate(result.images):
        img.save(f"smart_text_page_{i+1}.png")
        print(f"保存: smart_text_page_{i+1}.png")


if __name__ == "__main__":
    demo()
