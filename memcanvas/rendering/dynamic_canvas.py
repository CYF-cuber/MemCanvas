"""
DynamicCanvas - 动态画布系统

核心思想：
- 以 patch 为基本单位进行填充
- 填满一个 patch 后自动创建下一个
- 不浪费空间，不丢失信息
- 生成的 patches 可直接提取 vision tokens

优势：
1. 空间利用率高 - 每个 patch 都被充分利用
2. 无信息丢失 - 内容多就创建更多 patches
3. 与 vision token 提取天然对齐
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
import os


class ContentType(Enum):
    """内容类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    SEPARATOR = "separator"


@dataclass
class ContentBlock:
    """内容块"""
    type: ContentType
    data: Any
    # 渲染后的尺寸
    width: int = 0
    height: int = 0
    # 渲染后的图像（缓存）
    rendered: Optional[Image.Image] = None


@dataclass
class DynamicCanvasConfig:
    """动态画布配置"""
    # Patch 尺寸（与 vision encoder 的输入对齐）
    patch_size: int = 640

    # 内容边距
    padding: int = 20
    # 内容间距
    content_gap: int = 15

    # 文本渲染配置
    font_size: int = 20
    font_color: Tuple[int, int, int] = (0, 0, 0)
    line_spacing: float = 1.3

    # 背景色
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # 是否在 patch 边界显示标记（调试用）
    show_patch_boundary: bool = False


@dataclass
class Patch:
    """单个 Patch"""
    index: int
    image: Image.Image
    # 该 patch 包含的内容摘要
    content_summary: List[str] = field(default_factory=list)
    # 填充状态
    is_full: bool = False
    # 剩余空间
    remaining_height: int = 0


class DynamicCanvas:
    """
    动态画布

    核心工作流：
    1. 创建初始 patch
    2. 添加内容时，检查当前 patch 是否能容纳
    3. 如果能容纳，渲染到当前 patch
    4. 如果不能容纳，创建新 patch 继续渲染
    5. 最终输出 patch 列表

    使用示例：
    ```python
    canvas = DynamicCanvas(DynamicCanvasConfig(patch_size=640))

    # 添加内容
    canvas.add_text("标题", font_size=32)
    canvas.add_text("这是一段很长的文本...")
    canvas.add_image(some_image)
    canvas.add_table(table_data)

    # 获取所有 patches
    patches = canvas.get_patches()

    # 直接用于 vision token 提取
    for patch in patches:
        tokens = extractor.extract_single(patch.image)
    ```
    """

    # 字体候选列表
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

        # Patch 列表
        self.patches: List[Patch] = []

        # 当前 patch 索引和游标位置
        self._current_patch_idx: int = -1
        self._cursor_y: int = 0  # 当前 patch 中的 y 位置

        # 字体缓存
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path = self._find_font()

        # 统计
        self._total_content_blocks = 0

        # 创建第一个 patch
        self._create_new_patch()

    def _find_font(self) -> Optional[str]:
        """查找可用字体"""
        for fp in self.FONT_CANDIDATES:
            if os.path.exists(fp):
                return fp
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """获取字体"""
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
        """创建新的 patch"""
        # 保存当前 patch 的实际剩余高度（供 get_compact_image 使用）
        if self.patches:
            self.current_patch.remaining_height = self.available_height

        size = self.config.patch_size
        image = Image.new('RGB', (size, size), self.config.background_color)

        # 如果启用边界显示
        if self.config.show_patch_boundary:
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, size-1, size-1], outline=(200, 200, 200), width=2)
            # 显示 patch 编号
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
        """当前 patch"""
        return self.patches[self._current_patch_idx]

    @property
    def available_height(self) -> int:
        """当前 patch 剩余可用高度"""
        return self.config.patch_size - self._cursor_y - self.config.padding

    @property
    def content_width(self) -> int:
        """内容可用宽度"""
        return self.config.patch_size - 2 * self.config.padding

    # ==================== 添加内容 ====================

    def add_text(
        self,
        text: str,
        font_size: Optional[int] = None,
        font_color: Optional[Tuple[int, int, int]] = None,
        bold: bool = False
    ) -> int:
        """
        添加文本

        Args:
            text: 文本内容
            font_size: 字体大小
            font_color: 字体颜色
            bold: 是否加粗（暂不支持，预留）

        Returns:
            使用的 patch 数量
        """
        font_size = font_size or self.config.font_size
        font_color = font_color or self.config.font_color
        font = self._get_font(font_size)

        # 计算行高
        line_height = int(font_size * self.config.line_spacing)

        # 文本换行
        lines = self._wrap_text(text, font)

        patches_used = 0
        start_patch = self._current_patch_idx

        for line in lines:
            # 检查当前 patch 是否能容纳这一行
            if self.available_height < line_height:
                # 标记当前 patch 已满
                self.current_patch.is_full = True
                self.current_patch.remaining_height = self.available_height
                # 创建新 patch
                self._create_new_patch()
                patches_used += 1

            # 渲染这一行
            draw = ImageDraw.Draw(self.current_patch.image)
            draw.text(
                (self.config.padding, self._cursor_y),
                line,
                font=font,
                fill=font_color
            )
            self._cursor_y += line_height

        # 添加内容间距
        self._cursor_y += self.config.content_gap

        # 更新统计
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
        添加图像

        Args:
            image: PIL Image
            max_width: 最大宽度（默认为内容宽度）
            max_height: 最大高度（默认为 patch 高度的一半）
            caption: 图片说明

        Returns:
            使用的 patch 数量
        """
        max_width = max_width or self.content_width
        max_height = max_height or (self.config.patch_size // 2)

        # 缩放图像
        img_w, img_h = image.size
        scale = min(max_width / img_w, max_height / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if scale < 1.0:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 计算需要的总高度
        total_height = new_h
        if caption:
            caption_height = int(self.config.font_size * self.config.line_spacing)
            total_height += caption_height + 5

        # 检查是否需要新 patch
        patches_used = 0
        start_patch = self._current_patch_idx

        if self.available_height < total_height:
            self.current_patch.is_full = True
            self._create_new_patch()
            patches_used += 1

        # 粘贴图像
        x = self.config.padding + (self.content_width - new_w) // 2  # 居中
        self.current_patch.image.paste(image, (x, self._cursor_y))
        self._cursor_y += new_h

        # 添加说明
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
        """将文本截断到指定像素宽度，超出部分用省略号替换"""
        if max_width <= 0:
            return ""
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return text

        # 二分查找合适的截断位置
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
        添加表格

        Args:
            data: 表格数据 [[row1], [row2], ...]
            headers: 表头
            cell_padding: 单元格内边距

        Returns:
            使用的 patch 数量
        """
        if not data:
            return 0

        font = self._get_font(self.config.font_size - 2)

        # 计算列宽
        all_rows = ([headers] if headers else []) + data
        num_cols = max(len(row) for row in all_rows)

        col_widths = [0] * num_cols
        for row in all_rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    bbox = font.getbbox(str(cell))
                    cell_width = (bbox[2] - bbox[0]) + 2 * cell_padding
                    col_widths[i] = max(col_widths[i], cell_width)

        # 检查表格是否太宽，等比缩放列宽
        total_width = sum(col_widths)
        if total_width > self.content_width:
            scale = self.content_width / total_width
            col_widths = [max(int(w * scale), 2 * cell_padding + 10) for w in col_widths]
            # 确保总宽度不超出
            while sum(col_widths) > self.content_width:
                widest = max(range(num_cols), key=lambda i: col_widths[i])
                col_widths[widest] -= 1

        # 行高
        row_height = int(self.config.font_size * self.config.line_spacing) + 2 * cell_padding

        # 渲染表格
        patches_used = 0
        start_patch = self._current_patch_idx

        rows_to_render = ([headers] if headers else []) + data
        is_header = True if headers else False

        for row_idx, row in enumerate(rows_to_render):
            # 检查是否需要新 patch
            if self.available_height < row_height:
                self.current_patch.is_full = True
                self._create_new_patch()
                patches_used += 1

            draw = ImageDraw.Draw(self.current_patch.image)

            # 绘制单元格
            x = self.config.padding
            for col_idx, cell in enumerate(row):
                if col_idx >= len(col_widths):
                    break

                cell_width = col_widths[col_idx]

                # 绘制边框
                draw.rectangle(
                    [x, self._cursor_y, x + cell_width, self._cursor_y + row_height],
                    outline=(200, 200, 200),
                    fill=(240, 240, 240) if (is_header and row_idx == 0) else None
                )

                # 截断文本到列宽内，避免溢出
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
        添加 HTML/Markdown 渲染内容

        使用 HtmlRenderer 将内容渲染为图像后放入画布。
        Playwright 不可用时 fallback 到 add_text()。

        Args:
            content: Markdown、HTML 或纯文本
            content_type: "markdown", "html", 或 "text"
            max_height: 渲染图像最大高度

        Returns:
            使用的 patch 数量
        """
        try:
            from .renderers.html_renderer import HtmlRenderer, HtmlStyle

            style = HtmlStyle(viewport_width=self.content_width)
            renderer = HtmlRenderer(style=style)
            img = renderer.render(content, content_type)

            # 缩放到 content_width
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
            # Fallback: 纯文本渲染
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
        智能分段裁切长图，横向排列

        当图片纵横比 > 2.0 时，将长图等分为多段，
        每段等比缩放后横向排列在画布中，底部标注段号。
        这样能最大化利用画布空间，保持内容可读。

        例：596×5107 信息图 → 3 段横排，每段 196×559，
        总占 600×579，几乎填满一个 640×640 patch。

        Args:
            image: PIL Image（长图）
            max_sections: 最大段数（控制 token 预算）
            overlap: 段间重叠像素（避免切断重要内容）
            gap: 横排段间间距（像素）
            label_height: 段号标签高度（像素）

        Returns:
            使用的 patch 数量
        """
        img_w, img_h = image.size
        aspect_ratio = img_h / img_w if img_w > 0 else 1

        # 纵横比不高，直接作为普通图像添加
        if aspect_ratio <= 2.0:
            return self.add_image(image)

        # 根据纵横比决定分段数（2~max_sections）
        n_sections = min(max(2, round(aspect_ratio / 2.5)), max_sections)

        # 计算每段在原图中的裁切区域
        section_h = (img_h + (n_sections - 1) * overlap) // n_sections
        effective_step = section_h - overlap

        sections = []
        for i in range(n_sections):
            y_start = i * effective_step
            y_end = min(y_start + section_h, img_h)
            sections.append(image.crop((0, y_start, img_w, y_end)))

        # 计算横排布局：每段等宽排列
        avail_w = self.content_width - (n_sections - 1) * gap
        per_w = avail_w // n_sections
        scale = per_w / img_w
        per_h = int(section_h * scale)

        # 组装横排图像（各段 + 底部标签）
        total_w = n_sections * per_w + (n_sections - 1) * gap
        total_h = per_h + label_height
        composed = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        font = self._get_font(11)
        draw = ImageDraw.Draw(composed)

        for i, sec in enumerate(sections):
            # 缩放段落
            sec_h_actual = sec.size[1]
            scaled_h = int(sec_h_actual * scale)
            sec_resized = sec.resize((per_w, scaled_h), Image.Resampling.LANCZOS)

            # 粘贴到横排位置
            x = i * (per_w + gap)
            composed.paste(sec_resized, (x, 0))

            # 底部段号标签
            label = f"{i + 1}/{n_sections}"
            bbox = font.getbbox(label)
            lw = bbox[2] - bbox[0]
            lx = x + (per_w - lw) // 2
            ly = per_h + 3
            draw.text((lx, ly), label, fill=(120, 120, 120), font=font)

        # 作为单张图添加到画布
        return self.add_image(
            composed,
            max_width=self.content_width,
            max_height=self.config.patch_size - 2 * self.config.padding,
        )

    def add_separator(self, style: str = "line") -> int:
        """添加分隔符"""
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

    # ==================== 工具方法 ====================

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont) -> List[str]:
        """文本换行（按像素宽度）"""
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

    # ==================== 输出 ====================

    def get_patches(self) -> List[Patch]:
        """获取所有 patches"""
        # 更新最后一个 patch 的剩余空间
        self.current_patch.remaining_height = self.available_height
        return self.patches

    def get_images(self) -> List[Image.Image]:
        """获取所有 patch 图像"""
        return [p.image for p in self.patches]

    def get_combined_image(self, direction: str = "vertical") -> Image.Image:
        """
        获取合并后的图像

        Args:
            direction: "vertical" 或 "horizontal"

        Returns:
            合并后的图像
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
        获取紧凑合并图像（无多余空白）

        宽度固定为 patch_size，高度根据实际内容裁剪。
        已满的 patch 裁剪掉底部空白，最后一个 patch 裁剪到光标位置。

        Returns:
            紧凑的 RGB 图像
        """
        if not self.patches:
            return Image.new(
                'RGB', (self.config.patch_size, self.config.padding * 2),
                self.config.background_color,
            )

        width = self.config.patch_size

        # 计算每个 patch 的实际内容高度
        cropped = []
        for i, patch in enumerate(self.patches):
            if i == self._current_patch_idx:
                # 最后一个活跃 patch：裁剪到光标位置 + padding
                h = self._cursor_y + self.config.padding
            elif patch.is_full:
                # 已满 patch：裁剪掉底部剩余空白
                h = self.config.patch_size - patch.remaining_height
            else:
                h = self.config.patch_size
            h = max(h, self.config.padding * 2)  # 最小高度
            cropped.append(patch.image.crop((0, 0, width, h)))

        total_h = sum(c.height for c in cropped)
        combined = Image.new('RGB', (width, total_h), self.config.background_color)
        y = 0
        for c in cropped:
            combined.paste(c, (0, y))
            y += c.height

        return combined

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
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
        """清空画布"""
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
    快速创建动态画布

    Args:
        patch_size: Patch 尺寸（建议与 vision encoder 输入对齐）
        font_size: 默认字体大小
        padding: 边距

    Returns:
        DynamicCanvas 实例
    """
    config = DynamicCanvasConfig(
        patch_size=patch_size,
        font_size=font_size,
        padding=padding
    )
    return DynamicCanvas(config)
