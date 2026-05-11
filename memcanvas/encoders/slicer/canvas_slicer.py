"""
CanvasSlicer - 画布切分器

实现类似DeepSeek-OCR的画布切分逻辑：
1. 固定分辨率模式: 512/640/1024/1280 -> 64/100/256/400 vision tokens
2. Gundam模式: n×640×640 patches + 1×1024×1024 global view
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import math


@dataclass
class SliceConfig:
    """切分配置"""
    # 基础分辨率 (用于全局视图)
    base_size: int = 1024
    # patch分辨率
    patch_size: int = 640
    # 是否启用crop模式 (Gundam模式)
    crop_mode: bool = True
    # vision encoder的patch大小 (用于计算token数)
    vision_patch_size: int = 64
    # 最大patch数量
    max_patches: int = 16
    # 重叠比例 (0-1)
    overlap_ratio: float = 0.0


@dataclass
class SliceResult:
    """切分结果"""
    # 全局视图 (base_size × base_size)
    global_view: Image.Image
    # 局部patches列表 (patch_size × patch_size each)
    patches: List[Image.Image]
    # 每个patch的位置信息 (x, y, w, h) 在原图中的坐标
    patch_positions: List[Tuple[int, int, int, int]]
    # 原始图像尺寸
    original_size: Tuple[int, int]
    # 预估的vision token数量
    estimated_tokens: int
    # 配置信息
    config: SliceConfig


class CanvasSlicer:
    """
    画布切分器

    支持两种模式:
    1. 固定分辨率: 直接缩放到目标尺寸
    2. Gundam模式: 切分为多个patches + 全局视图
    """

    # 预定义的分辨率模式及对应的vision token数
    RESOLUTION_MODES = {
        "tiny": {"size": 512, "tokens": 64},
        "small": {"size": 640, "tokens": 100},
        "base": {"size": 1024, "tokens": 256},
        "large": {"size": 1280, "tokens": 400},
    }

    def __init__(self, config: Optional[SliceConfig] = None):
        self.config = config or SliceConfig()

    def slice(
        self,
        source: Union[str, Path, Image.Image]
    ) -> SliceResult:
        """
        切分画布

        Args:
            source: 图像路径或PIL Image

        Returns:
            SliceResult 包含全局视图和patches
        """
        # 加载图像
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            image = source

        if image.mode != 'RGB':
            image = image.convert('RGB')

        original_size = image.size

        if self.config.crop_mode:
            return self._slice_gundam(image, original_size)
        else:
            return self._slice_fixed(image, original_size)

    def _slice_fixed(
        self,
        image: Image.Image,
        original_size: Tuple[int, int]
    ) -> SliceResult:
        """固定分辨率模式"""
        target_size = self.config.base_size

        # 缩放到目标尺寸
        global_view = image.resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS
        )

        # 计算token数
        tokens = (target_size // self.config.vision_patch_size) ** 2

        return SliceResult(
            global_view=global_view,
            patches=[],
            patch_positions=[],
            original_size=original_size,
            estimated_tokens=tokens,
            config=self.config
        )

    def _slice_gundam(
        self,
        image: Image.Image,
        original_size: Tuple[int, int]
    ) -> SliceResult:
        """
        Gundam模式: n×patch_size patches + 1×base_size global
        """
        w, h = original_size
        patch_size = self.config.patch_size
        base_size = self.config.base_size

        # 1. 创建全局视图
        global_view = self._create_global_view(image, base_size)

        # 2. 计算需要多少patches
        patches, positions = self._create_patches(image, patch_size)

        # 3. 计算token数
        global_tokens = (base_size // self.config.vision_patch_size) ** 2
        patch_tokens = len(patches) * (patch_size // self.config.vision_patch_size) ** 2
        total_tokens = global_tokens + patch_tokens

        return SliceResult(
            global_view=global_view,
            patches=patches,
            patch_positions=positions,
            original_size=original_size,
            estimated_tokens=total_tokens,
            config=self.config
        )

    def _create_global_view(
        self,
        image: Image.Image,
        target_size: int
    ) -> Image.Image:
        """创建全局视图 (保持宽高比，填充白色)"""
        w, h = image.size

        # 计算缩放比例
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建白色背景并居中放置
        result = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def _create_patches(
        self,
        image: Image.Image,
        patch_size: int
    ) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
        """
        将图像切分为patches

        使用滑动窗口方式，支持重叠
        """
        w, h = image.size
        patches = []
        positions = []

        # 计算步长 (考虑重叠)
        stride = int(patch_size * (1 - self.config.overlap_ratio))
        stride = max(stride, 1)

        # 计算需要多少行列
        cols = max(1, math.ceil((w - patch_size) / stride) + 1) if w > patch_size else 1
        rows = max(1, math.ceil((h - patch_size) / stride) + 1) if h > patch_size else 1

        # 限制最大patch数
        total_patches = cols * rows
        if total_patches > self.config.max_patches:
            # 需要缩小图像或减少patches
            scale = math.sqrt(self.config.max_patches / total_patches)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
            cols = max(1, math.ceil((w - patch_size) / stride) + 1) if w > patch_size else 1
            rows = max(1, math.ceil((h - patch_size) / stride) + 1) if h > patch_size else 1

        for row in range(rows):
            for col in range(cols):
                # 计算patch位置
                x = min(col * stride, max(0, w - patch_size))
                y = min(row * stride, max(0, h - patch_size))

                # 提取patch
                patch = self._extract_patch(image, x, y, patch_size)
                patches.append(patch)
                positions.append((x, y, patch_size, patch_size))

        return patches, positions

    def _extract_patch(
        self,
        image: Image.Image,
        x: int,
        y: int,
        size: int
    ) -> Image.Image:
        """提取单个patch，不足部分填充白色"""
        w, h = image.size

        # 计算实际可提取的区域
        actual_w = min(size, w - x)
        actual_h = min(size, h - y)

        # 提取区域
        crop = image.crop((x, y, x + actual_w, y + actual_h))

        # 如果不足size，填充白色
        if actual_w < size or actual_h < size:
            result = Image.new('RGB', (size, size), (255, 255, 255))
            result.paste(crop, (0, 0))
            return result

        return crop

    @classmethod
    def get_mode_config(cls, mode: str) -> SliceConfig:
        """获取预定义模式的配置"""
        if mode not in cls.RESOLUTION_MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(cls.RESOLUTION_MODES.keys())}")

        info = cls.RESOLUTION_MODES[mode]
        return SliceConfig(
            base_size=info["size"],
            patch_size=info["size"],
            crop_mode=False
        )

    @classmethod
    def estimate_tokens(cls, width: int, height: int, config: SliceConfig) -> int:
        """预估给定尺寸图像的vision token数量"""
        if not config.crop_mode:
            return (config.base_size // config.vision_patch_size) ** 2

        # Gundam模式
        global_tokens = (config.base_size // config.vision_patch_size) ** 2

        # 计算patches数量
        patch_size = config.patch_size
        stride = int(patch_size * (1 - config.overlap_ratio))
        cols = max(1, math.ceil((width - patch_size) / stride) + 1) if width > patch_size else 1
        rows = max(1, math.ceil((height - patch_size) / stride) + 1) if height > patch_size else 1
        num_patches = min(cols * rows, config.max_patches)

        patch_tokens = num_patches * (patch_size // config.vision_patch_size) ** 2

        return global_tokens + patch_tokens
