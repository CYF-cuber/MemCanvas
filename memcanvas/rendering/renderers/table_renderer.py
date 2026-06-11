"""
TableRenderer - text

text（list/dict/DataFrame）text。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont
import os


@dataclass
class TableStyle:
    """text"""
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
    text

    text。
    """

    # text
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
        """text"""
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
        text

        Args:
            data: text（text、text DataFrame）
            headers: text（text data text）
            max_width: text
            max_col_width: text

        Returns:
            text RGBA text
        """
        # text
        rows, headers = self._normalize_data(data, headers)

        if not rows and not headers:
            return self._render_empty_table()

        # text
        col_widths = self._calculate_column_widths(rows, headers, max_col_width)

        # text
        table_width = sum(col_widths) + self.style.border_width * (len(col_widths) + 1)
        row_height = self.style.font_size + 2 * self.style.cell_padding
        header_height = self.style.header_font_size + 2 * self.style.cell_padding
        table_height = header_height + row_height * len(rows) + self.style.border_width * (len(rows) + 2)

        if max_width and table_width > max_width:
            # text
            scale = max_width / table_width
            col_widths = [int(w * scale) for w in col_widths]
            table_width = max_width

        # text
        image = Image.new('RGBA', (table_width, table_height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        # text
        y = 0

        # text
        if headers:
            self._draw_row(
                draw, headers, col_widths, y, header_height,
                self.style.header_background,
                self.style.header_text_color,
                self._header_font,
                is_header=True
            )
            y += header_height + self.style.border_width

        # text
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
        """text"""
        # text DataFrame
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

        # text
        if isinstance(data[0], dict):
            if headers is None:
                headers = list(data[0].keys())
            rows = [[str(row.get(h, "")) for h in headers] for row in data]
            return rows, headers

        # text
        rows = [[str(cell) for cell in row] for row in data]
        return rows, headers or []

    def _calculate_column_widths(
        self,
        rows: List[List[str]],
        headers: List[str],
        max_col_width: int
    ) -> List[int]:
        """text"""
        if not rows and not headers:
            return []

        num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
        widths = [0] * num_cols

        # text
        for i, h in enumerate(headers):
            if i < num_cols:
                bbox = self._header_font.getbbox(str(h))
                widths[i] = max(widths[i], bbox[2] - bbox[0] if bbox else 0)

        # text
        for row in rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    bbox = self._font.getbbox(str(cell))
                    widths[i] = max(widths[i], bbox[2] - bbox[0] if bbox else 0)

        # text padding text
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
        """text"""
        x = 0

        for i, (cell, width) in enumerate(zip(cells, col_widths)):
            # text
            draw.rectangle(
                [x, y, x + width, y + height],
                fill=bg_color,
                outline=self.style.border_color,
                width=self.style.border_width
            )

            # text（text）
            text = str(cell)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0] if bbox else 0

            # text
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
                text_x = x + (width - text_width) // 2  # text

            text_y = y + (height - (bbox[3] - bbox[1] if bbox else font.size)) // 2

            draw.text((text_x, text_y), text, fill=text_color, font=font)

            x += width + self.style.border_width

    def _render_empty_table(self) -> Image.Image:
        """text"""
        width, height = 200, 50
        image = Image.new('RGBA', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width - 1, height - 1], outline=self.style.border_color)
        draw.text((10, 15), "[text]", fill=self.style.text_color, font=self._font)
        return image

    def render_dict(
        self,
        data: Dict[str, Any],
        key_header: str = "Key",
        value_header: str = "Value"
    ) -> Image.Image:
        """
        text

        Args:
            data: text
            key_header: text
            value_header: text

        Returns:
            text
        """
        rows = [[str(k), str(v)] for k, v in data.items()]
        return self.render(rows, headers=[key_header, value_header])
