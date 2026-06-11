"""
ImageRenderer - text

text、text、text。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from PIL import Image
from pathlib import Path


@dataclass
class ImageStyle:
    """text"""
    fit_mode: str = "contain"  # contain, cover, stretch, none
    alignment: str = "center"  # left, center, right
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    border_width: int = 0
    border_color: Tuple[int, int, int, int] = (200, 200, 200, 255)
    padding: int = 0


class ImageRenderer:
    """
    text

    text、text、text。
    """

    def __init__(self, style: Optional[ImageStyle] = None):
        self.style = style or ImageStyle()

    def render(
        self,
        source: Union[str, Path, Image.Image],
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        text

        Args:
            source: text PIL Image object
            target_width: text
            target_height: text

        Returns:
            text RGBA text
        """
        # text
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            image = source

        # text RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # text（text padding text border）
        available_width = target_width - 2 * (self.style.padding + self.style.border_width)
        available_height = target_height - 2 * (self.style.padding + self.style.border_width)

        # text fit_mode text
        if self.style.fit_mode == "contain":
            fitted = self._fit_contain(image, available_width, available_height)
        elif self.style.fit_mode == "cover":
            fitted = self._fit_cover(image, available_width, available_height)
        elif self.style.fit_mode == "stretch":
            fitted = image.resize((available_width, available_height), Image.Resampling.LANCZOS)
        else:  # none
            fitted = image

        # text
        result = Image.new('RGBA', (target_width, target_height), self.style.background_color)

        # text（text）
        if self.style.border_width > 0:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(result)
            draw.rectangle(
                [
                    self.style.padding,
                    self.style.padding,
                    target_width - self.style.padding - 1,
                    target_height - self.style.padding - 1
                ],
                outline=self.style.border_color,
                width=self.style.border_width
            )

        # text（text）
        offset = self.style.padding + self.style.border_width

        # text
        if self.style.alignment == "left":
            paste_x = offset
        elif self.style.alignment == "right":
            paste_x = offset + (available_width - fitted.width)
        else:  # center
            paste_x = offset + (available_width - fitted.width) // 2

        # text
        paste_y = offset + (available_height - fitted.height) // 2

        # text
        result.paste(fitted, (paste_x, paste_y), fitted)

        return result

    def _fit_contain(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        Contain text：text，text

        text，text。
        """
        # text
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        ratio = min(width_ratio, height_ratio)

        # text
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _fit_cover(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        Cover text：text，text

        text。
        """
        # text
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        ratio = max(width_ratio, height_ratio)

        # text
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # text
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return scaled.crop((left, top, right, bottom))

    def load(self, source: Union[str, Path]) -> Image.Image:
        """
        text

        Args:
            source: text

        Returns:
            PIL Image object
        """
        image = Image.open(source)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return image

    def create_thumbnail(
        self,
        source: Union[str, Path, Image.Image],
        size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        text

        Args:
            source: text
            size: text

        Returns:
            text
        """
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            image = source.copy()

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image

    def create_grid(
        self,
        images: list,
        grid_size: Tuple[int, int],
        cell_size: Tuple[int, int],
        gap: int = 10
    ) -> Image.Image:
        """
        text

        Args:
            images: text
            grid_size: (text, text)
            cell_size: text
            gap: text

        Returns:
            text
        """
        cols, rows = grid_size
        cell_w, cell_h = cell_size

        # text
        total_width = cols * cell_w + (cols - 1) * gap
        total_height = rows * cell_h + (rows - 1) * gap

        result = Image.new('RGBA', (total_width, total_height), self.style.background_color)

        for idx, img in enumerate(images):
            if idx >= cols * rows:
                break

            row = idx // cols
            col = idx % cols

            x = col * (cell_w + gap)
            y = row * (cell_h + gap)

            # text
            cell_img = self.render(img, cell_w, cell_h)
            result.paste(cell_img, (x, y), cell_img)

        return result
