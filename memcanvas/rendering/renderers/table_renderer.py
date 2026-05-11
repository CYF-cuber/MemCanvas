"""
TableRenderer - 表格渲染器

将表格数据（list/dict/DataFrame）渲染为图像。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont
import os


@dataclass
class TableStyle:
    """表格样式配置"""
    font_path: Optional[str] = None
    font_size: int = 16
    header_font_size: int = 18
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    header_text_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    header_background: Tuple[int, int, int, int] = (66, 133, 244, 255)
    row_alt_background: Tuple[int, int, int, int] = (245, 245, 245, 255)
    border_color: Tuple[int, int, int, int] = (200, 200, 200, 255)
    cell_padding: int = 10
    border_width: int = 1


class TableRenderer:
    """
    表格渲染器

    将结构化数据渲染为表格图像。
    """

    # 字体候选列表
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    def __init__(self, style: Optional[TableStyle] = None):
        self.style = style or TableStyle()
        self._font = self._load_font(self.style.font_size)
        self._header_font = self._load_font(self.style.header_font_size)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """加载字体"""
        if self.style.font_path and os.path.exists(self.style.font_path):
            try:
                return ImageFont.truetype(self.style.font_path, size)
            except Exception:
                pass

        for font_path in self.FONT_CANDIDATES:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue

        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def render(
        self,
        data: Union[List[List[Any]], List[Dict[str, Any]], "pd.DataFrame"],
        headers: Optional[List[str]] = None,
        max_width: Optional[int] = None,
        max_col_width: int = 300
    ) -> Image.Image:
        """
        渲染表格

        Args:
            data: 表格数据（二维列表、字典列表或 DataFrame）
            headers: 表头（如果 data 是二维列表）
            max_width: 最大宽度
            max_col_width: 单列最大宽度

        Returns:
            表格 RGBA 图像
        """
        # 标准化数据
        rows, headers = self._normalize_data(data, headers)

        if not rows and not headers:
            return self._render_empty_table()

        # 计算列宽
        col_widths = self._calculate_column_widths(rows, headers, max_col_width)

        # 计算尺寸
        table_width = sum(col_widths) + self.style.border_width * (len(col_widths) + 1)
        row_height = self.style.font_size + 2 * self.style.cell_padding
        header_height = self.style.header_font_size + 2 * self.style.cell_padding
        table_height = header_height + row_height * len(rows) + self.style.border_width * (len(rows) + 2)

        if max_width and table_width > max_width:
            # 需要缩放
            scale = max_width / table_width
            col_widths = [int(w * scale) for w in col_widths]
            table_width = max_width

        # 创建图像
        image = Image.new('RGBA', (table_width, table_height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        # 绘制表格
        y = 0

        # 绘制表头
        if headers:
            self._draw_row(
                draw, headers, col_widths, y, header_height,
                self.style.header_background,
                self.style.header_text_color,
                self._header_font,
                is_header=True
            )
            y += header_height + self.style.border_width

        # 绘制数据行
        for idx, row in enumerate(rows):
            bg_color = self.style.row_alt_background if idx % 2 == 1 else self.style.background_color
            self._draw_row(
                draw, row, col_widths, y, row_height,
                bg_color,
                self.style.text_color,
                self._font
            )
            y += row_height + self.style.border_width

        return image

    def _normalize_data(
        self,
        data: Union[List[List[Any]], List[Dict[str, Any]], Any],
        headers: Optional[List[str]]
    ) -> Tuple[List[List[str]], List[str]]:
        """标准化数据格式"""
        # 尝试处理 DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                headers = list(data.columns)
                rows = [[str(cell) for cell in row] for row in data.values.tolist()]
                return rows, headers
        except ImportError:
            pass

        if not data:
            return [], headers or []

        # 字典列表
        if isinstance(data[0], dict):
            if headers is None:
                headers = list(data[0].keys())
            rows = [[str(row.get(h, "")) for h in headers] for row in data]
            return rows, headers

        # 二维列表
        rows = [[str(cell) for cell in row] for row in data]
        return rows, headers or []

    def _calculate_column_widths(
        self,
        rows: List[List[str]],
        headers: List[str],
        max_col_width: int
    ) -> List[int]:
        """计算每列宽度"""
        if not rows and not headers:
            return []

        num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
        widths = [0] * num_cols

        # 计算表头宽度
        for i, h in enumerate(headers):
            if i < num_cols:
                bbox = self._header_font.getbbox(str(h))
                widths[i] = max(widths[i], bbox[2] - bbox[0] if bbox else 0)

        # 计算数据宽度
        for row in rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    bbox = self._font.getbbox(str(cell))
                    widths[i] = max(widths[i], bbox[2] - bbox[0] if bbox else 0)

        # 添加 padding 并限制最大宽度
        widths = [min(w + 2 * self.style.cell_padding, max_col_width) for w in widths]

        return widths

    def _draw_row(
        self,
        draw: ImageDraw.ImageDraw,
        cells: List[str],
        col_widths: List[int],
        y: int,
        height: int,
        bg_color: Tuple[int, int, int, int],
        text_color: Tuple[int, int, int, int],
        font: ImageFont.FreeTypeFont,
        is_header: bool = False
    ):
        """绘制一行"""
        x = 0

        for i, (cell, width) in enumerate(zip(cells, col_widths)):
            # 绘制背景
            draw.rectangle(
                [x, y, x + width, y + height],
                fill=bg_color,
                outline=self.style.border_color,
                width=self.style.border_width
            )

            # 绘制文字（居中或左对齐）
            text = str(cell)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0] if bbox else 0

            # 截断过长文本
            if text_width > width - 2 * self.style.cell_padding:
                while text and text_width > width - 2 * self.style.cell_padding - 20:
                    text = text[:-1]
                    bbox = font.getbbox(text + "...")
                    text_width = bbox[2] - bbox[0] if bbox else 0
                text += "..."
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0] if bbox else 0

            text_x = x + self.style.cell_padding
            if is_header:
                text_x = x + (width - text_width) // 2  # 表头居中

            text_y = y + (height - (bbox[3] - bbox[1] if bbox else font.size)) // 2

            draw.text((text_x, text_y), text, fill=text_color, font=font)

            x += width + self.style.border_width

    def _render_empty_table(self) -> Image.Image:
        """渲染空表格"""
        width, height = 200, 50
        image = Image.new('RGBA', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width - 1, height - 1], outline=self.style.border_color)
        draw.text((10, 15), "[空表格]", fill=self.style.text_color, font=self._font)
        return image

    def render_dict(
        self,
        data: Dict[str, Any],
        key_header: str = "Key",
        value_header: str = "Value"
    ) -> Image.Image:
        """
        渲染字典为两列表格

        Args:
            data: 字典数据
            key_header: 键列表头
            value_header: 值列表头

        Returns:
            表格图像
        """
        rows = [[str(k), str(v)] for k, v in data.items()]
        return self.render(rows, headers=[key_header, value_header])
