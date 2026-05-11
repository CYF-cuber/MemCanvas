"""
ImageRenderer - 图像渲染器

处理图像的缩放、裁剪、适配画布等操作。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from PIL import Image
from pathlib import Path


@dataclass
class ImageStyle:
    """图像样式配置"""
    fit_mode: str = "contain"  # contain, cover, stretch, none
    alignment: str = "center"  # left, center, right
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    border_width: int = 0
    border_color: Tuple[int, int, int, int] = (200, 200, 200, 255)
    padding: int = 0


class ImageRenderer:
    """
    图像渲染器

    处理图像的加载、缩放、适配等操作。
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
        渲染图像到目标尺寸

        Args:
            source: 图像路径或 PIL Image 对象
            target_width: 目标宽度
            target_height: 目标高度

        Returns:
            处理后的 RGBA 图像
        """
        # 加载图像
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            image = source

        # 转换为 RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # 计算实际可用区域（减去 padding 和 border）
        available_width = target_width - 2 * (self.style.padding + self.style.border_width)
        available_height = target_height - 2 * (self.style.padding + self.style.border_width)

        # 根据 fit_mode 处理
        if self.style.fit_mode == "contain":
            fitted = self._fit_contain(image, available_width, available_height)
        elif self.style.fit_mode == "cover":
            fitted = self._fit_cover(image, available_width, available_height)
        elif self.style.fit_mode == "stretch":
            fitted = image.resize((available_width, available_height), Image.Resampling.LANCZOS)
        else:  # none
            fitted = image

        # 创建目标画布
        result = Image.new('RGBA', (target_width, target_height), self.style.background_color)

        # 绘制边框（如果有）
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

        # 计算粘贴位置（根据对齐方式）
        offset = self.style.padding + self.style.border_width

        # 水平对齐
        if self.style.alignment == "left":
            paste_x = offset
        elif self.style.alignment == "right":
            paste_x = offset + (available_width - fitted.width)
        else:  # center
            paste_x = offset + (available_width - fitted.width) // 2

        # 垂直居中
        paste_y = offset + (available_height - fitted.height) // 2

        # 粘贴图像
        result.paste(fitted, (paste_x, paste_y), fitted)

        return result

    def _fit_contain(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        Contain 模式：保持比例，完整显示图像

        图像会被缩放以完全适应目标区域，可能有留白。
        """
        # 计算缩放比例
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        ratio = min(width_ratio, height_ratio)

        # 计算新尺寸
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
        Cover 模式：保持比例，填满目标区域

        图像会被缩放并裁剪以填满目标区域。
        """
        # 计算缩放比例
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        ratio = max(width_ratio, height_ratio)

        # 先缩放
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 居中裁剪
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return scaled.crop((left, top, right, bottom))

    def load(self, source: Union[str, Path]) -> Image.Image:
        """
        加载图像

        Args:
            source: 图像路径

        Returns:
            PIL Image 对象
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
        创建缩略图

        Args:
            source: 图像源
            size: 缩略图尺寸

        Returns:
            缩略图
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
        创建图像网格

        Args:
            images: 图像列表
            grid_size: (列数, 行数)
            cell_size: 每个单元格尺寸
            gap: 单元格间距

        Returns:
            网格图像
        """
        cols, rows = grid_size
        cell_w, cell_h = cell_size

        # 计算总尺寸
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

            # 渲染单个图像到单元格
            cell_img = self.render(img, cell_w, cell_h)
            result.paste(cell_img, (x, y), cell_img)

        return result
