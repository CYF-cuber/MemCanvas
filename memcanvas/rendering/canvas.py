"""
MemoryCanvas - 核心画布类

负责管理多层画布的创建、渲染和合成。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime


@dataclass
class CanvasConfig:
    """画布配置"""
    width: int = 2048
    height: int = 2048
    patch_size: int = 14  # ViT/SigLIP/CLIP 常用 patch 大小
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    # 各层配置
    main_layer_opacity: float = 1.0
    structure_layer_opacity: float = 0.8
    anchor_layer_opacity: float = 0.6

    def __post_init__(self):
        """确保画布尺寸是 patch_size 的整数倍"""
        self.width = self.align_to_patch(self.width)
        self.height = self.align_to_patch(self.height)

    def align_to_patch(self, value: int) -> int:
        """将值对齐到 patch_size 的整数倍（向下取整）"""
        return (value // self.patch_size) * self.patch_size

    def align_to_patch_ceil(self, value: int) -> int:
        """将值对齐到 patch_size 的整数倍（向上取整）"""
        return ((value + self.patch_size - 1) // self.patch_size) * self.patch_size


@dataclass
class CanvasMetadata:
    """画布元数据"""
    memory_id: str
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    source: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class MemoryCanvas:
    """
    统一记忆画布

    将多模态数据渲染到三层画布上:
    1. 主内容层 (main): 实际内容（文本渲染、图片、频谱图等）
    2. 结构层 (structure): 布局信息（网格、分隔线、时间轴等）
    3. 锚点层 (anchor): 定位标记（时间戳、页码、来源标签等）
    """

    def __init__(
        self,
        config: Optional[CanvasConfig] = None,
        metadata: Optional[CanvasMetadata] = None
    ):
        self.config = config or CanvasConfig()
        self.metadata = metadata or CanvasMetadata(memory_id=self._generate_id())

        # 初始化三层画布 (RGBA)
        self._main_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            self.config.background_color
        )
        self._structure_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)  # 透明
        )
        self._anchor_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)  # 透明
        )

        # 绘制上下文
        self._main_draw = ImageDraw.Draw(self._main_layer)
        self._structure_draw = ImageDraw.Draw(self._structure_layer)
        self._anchor_draw = ImageDraw.Draw(self._anchor_layer)

        # 布局追踪
        self._content_regions: List[Dict[str, Any]] = []
        self._current_y = 0  # 当前渲染位置（自动布局用）

    @property
    def patch_size(self) -> int:
        """获取 patch 大小"""
        return self.config.patch_size

    def align_to_patch(self, value: int) -> int:
        """将值对齐到 patch_size 的整数倍（向下取整）"""
        return self.config.align_to_patch(value)

    def align_to_patch_ceil(self, value: int) -> int:
        """将值对齐到 patch_size 的整数倍（向上取整）"""
        return self.config.align_to_patch_ceil(value)

    def _generate_id(self) -> str:
        """生成记忆 ID"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mem_{ts}"

    @property
    def main_layer(self) -> Image.Image:
        """获取主内容层"""
        return self._main_layer

    @property
    def structure_layer(self) -> Image.Image:
        """获取结构层"""
        return self._structure_layer

    @property
    def anchor_layer(self) -> Image.Image:
        """获取锚点层"""
        return self._anchor_layer

    @property
    def main_draw(self) -> ImageDraw.ImageDraw:
        """获取主层绘制上下文"""
        return self._main_draw

    @property
    def structure_draw(self) -> ImageDraw.ImageDraw:
        """获取结构层绘制上下文"""
        return self._structure_draw

    @property
    def anchor_draw(self) -> ImageDraw.ImageDraw:
        """获取锚点层绘制上下文"""
        return self._anchor_draw

    def allocate_region(
        self,
        width: Optional[int] = None,
        height: int = 200,
        modality: str = "unknown",
        num_patches_height: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        """
        分配一个内容区域（自动布局，对齐到 patch_size）

        Args:
            width: 宽度（None 则使用画布宽度）
            height: 高度（会被对齐到 patch_size 的整数倍）
            modality: 模态类型
            num_patches_height: 直接指定高度为多少个 patch（优先于 height）

        Returns:
            (x, y, width, height) 区域坐标，所有值都是 patch_size 的整数倍
        """
        ps = self.config.patch_size

        # X 坐标从 0 开始（patch 对齐）
        x = 0

        # Y 坐标对齐到 patch
        y = self.align_to_patch_ceil(self._current_y)

        # 宽度对齐
        w = width or self.config.width
        w = self.align_to_patch(w)

        # 高度对齐（支持直接指定 patch 数量）
        if num_patches_height is not None:
            h = num_patches_height * ps
        else:
            h = self.align_to_patch_ceil(height)

        # 确保至少有一个 patch
        h = max(h, ps)
        w = max(w, ps)

        # 检查是否超出画布
        if y + h > self.config.height:
            raise ValueError(f"画布空间不足: 需要 y={y + h}, 可用高度={self.config.height}")

        region = {
            "bbox": (x, y, w, h),
            "modality": modality,
            "index": len(self._content_regions),
            "patches": (w // ps, h // ps)  # 记录 patch 数量 (cols, rows)
        }
        self._content_regions.append(region)
        self._current_y = y + h

        return (x, y, w, h)

    def paste_image(
        self,
        image: Image.Image,
        position: Tuple[int, int],
        layer: str = "main"
    ):
        """
        将图像粘贴到指定层

        Args:
            image: 要粘贴的图像
            position: (x, y) 位置
            layer: 目标层 ("main", "structure", "anchor")
        """
        target = {
            "main": self._main_layer,
            "structure": self._structure_layer,
            "anchor": self._anchor_layer
        }.get(layer, self._main_layer)

        # 如果图像有 alpha 通道，使用它作为 mask
        if image.mode == 'RGBA':
            target.paste(image, position, image)
        else:
            target.paste(image, position)

    def composite(self) -> Image.Image:
        """
        合成所有层为最终图像

        Returns:
            合成后的 RGBA 图像
        """
        # 从主层开始
        result = self._main_layer.copy()

        # 叠加结构层
        if self.config.structure_layer_opacity < 1.0:
            structure = self._structure_layer.copy()
            # 调整透明度
            alpha = structure.split()[3]
            alpha = alpha.point(lambda x: int(x * self.config.structure_layer_opacity))
            structure.putalpha(alpha)
            result = Image.alpha_composite(result, structure)
        else:
            result = Image.alpha_composite(result, self._structure_layer)

        # 叠加锚点层
        if self.config.anchor_layer_opacity < 1.0:
            anchor = self._anchor_layer.copy()
            alpha = anchor.split()[3]
            alpha = alpha.point(lambda x: int(x * self.config.anchor_layer_opacity))
            anchor.putalpha(alpha)
            result = Image.alpha_composite(result, anchor)
        else:
            result = Image.alpha_composite(result, self._anchor_layer)

        return result

    def save(self, path: str, format: str = "PNG"):
        """保存合成后的画布"""
        result = self.composite()
        result.save(path, format)

    def save_layers(self, prefix: str):
        """分别保存各层"""
        self._main_layer.save(f"{prefix}_main.png")
        self._structure_layer.save(f"{prefix}_structure.png")
        self._anchor_layer.save(f"{prefix}_anchor.png")
        self.composite().save(f"{prefix}_composite.png")

    def get_content_regions(self) -> List[Dict[str, Any]]:
        """获取所有内容区域信息"""
        return self._content_regions.copy()

    def to_numpy(self) -> np.ndarray:
        """将合成画布转换为 numpy 数组"""
        return np.array(self.composite())

    def reset(self):
        """重置画布"""
        self._main_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            self.config.background_color
        )
        self._structure_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)
        )
        self._anchor_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)
        )
        self._main_draw = ImageDraw.Draw(self._main_layer)
        self._structure_draw = ImageDraw.Draw(self._structure_layer)
        self._anchor_draw = ImageDraw.Draw(self._anchor_layer)
        self._content_regions = []
        self._current_y = 0

    def __repr__(self) -> str:
        return (
            f"MemoryCanvas(id={self.metadata.memory_id}, "
            f"size={self.config.width}x{self.config.height}, "
            f"regions={len(self._content_regions)})"
        )
