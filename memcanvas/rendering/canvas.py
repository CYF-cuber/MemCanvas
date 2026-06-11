"""
MemoryCanvas - text

Responsibilitiestext、text。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime


@dataclass
class CanvasConfig:
    """text"""
    width: int = 2048
    height: int = 2048
    patch_size: int = 14  # ViT/SigLIP/CLIP text patch text
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    # text
    main_layer_opacity: float = 1.0
    structure_layer_opacity: float = 0.8
    anchor_layer_opacity: float = 0.6

    def __post_init__(self):
        """text patch_size text"""
        self.width = self.align_to_patch(self.width)
        self.height = self.align_to_patch(self.height)

    def align_to_patch(self, value: int) -> int:
        """text patch_size text（text）"""
        return (value // self.patch_size) * self.patch_size

    def align_to_patch_ceil(self, value: int) -> int:
        """text patch_size text（text）"""
        return ((value + self.patch_size - 1) // self.patch_size) * self.patch_size


@dataclass
class CanvasMetadata:
    """text"""
    memory_id: str
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    source: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class MemoryCanvas:
    """
    text

    text:
    1. text (main): text（text、text、text）
    2. text (structure): text（text、text、text）
    3. text (anchor): text（text、text、text）
    """

    def __init__(
        self,
        config: Optional[CanvasConfig] = None,
        metadata: Optional[CanvasMetadata] = None
    ):
        self.config = config or CanvasConfig()
        self.metadata = metadata or CanvasMetadata(memory_id=self._generate_id())

        # text (RGBA)
        self._main_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            self.config.background_color
        )
        self._structure_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)  # text
        )
        self._anchor_layer = Image.new(
            'RGBA',
            (self.config.width, self.config.height),
            (0, 0, 0, 0)  # text
        )

        # text
        self._main_draw = ImageDraw.Draw(self._main_layer)
        self._structure_draw = ImageDraw.Draw(self._structure_layer)
        self._anchor_draw = ImageDraw.Draw(self._anchor_layer)

        # text
        self._content_regions: List[Dict[str, Any]] = []
        self._current_y = 0  # text（text）

    @property
    def patch_size(self) -> int:
        """text patch text"""
        return self.config.patch_size

    def align_to_patch(self, value: int) -> int:
        """text patch_size text（text）"""
        return self.config.align_to_patch(value)

    def align_to_patch_ceil(self, value: int) -> int:
        """text patch_size text（text）"""
        return self.config.align_to_patch_ceil(value)

    def _generate_id(self) -> str:
        """text ID"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mem_{ts}"

    @property
    def main_layer(self) -> Image.Image:
        """text"""
        return self._main_layer

    @property
    def structure_layer(self) -> Image.Image:
        """text"""
        return self._structure_layer

    @property
    def anchor_layer(self) -> Image.Image:
        """text"""
        return self._anchor_layer

    @property
    def main_draw(self) -> ImageDraw.ImageDraw:
        """text"""
        return self._main_draw

    @property
    def structure_draw(self) -> ImageDraw.ImageDraw:
        """text"""
        return self._structure_draw

    @property
    def anchor_draw(self) -> ImageDraw.ImageDraw:
        """text"""
        return self._anchor_draw

    def allocate_region(
        self,
        width: Optional[int] = None,
        height: int = 200,
        modality: str = "unknown",
        num_patches_height: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        """
        text（text，text patch_size）

        Args:
            width: text（None text）
            height: text（text patch_size text）
            modality: text
            num_patches_height: text patch（text height）

        Returns:
            (x, y, width, height) text，text patch_size text
        """
        ps = self.config.patch_size

        # X text 0 text（patch text）
        x = 0

        # Y text patch
        y = self.align_to_patch_ceil(self._current_y)

        # text
        w = width or self.config.width
        w = self.align_to_patch(w)

        # text（text patch text）
        if num_patches_height is not None:
            h = num_patches_height * ps
        else:
            h = self.align_to_patch_ceil(height)

        # text patch
        h = max(h, ps)
        w = max(w, ps)

        # text
        if y + h > self.config.height:
            raise ValueError(f"text: text y={y + h}, text={self.config.height}")

        region = {
            "bbox": (x, y, w, h),
            "modality": modality,
            "index": len(self._content_regions),
            "patches": (w // ps, h // ps)  # text patch text (cols, rows)
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
        text

        Args:
            image: text
            position: (x, y) text
            layer: text ("main", "structure", "anchor")
        """
        target = {
            "main": self._main_layer,
            "structure": self._structure_layer,
            "anchor": self._anchor_layer
        }.get(layer, self._main_layer)

        # text alpha text，text mask
        if image.mode == 'RGBA':
            target.paste(image, position, image)
        else:
            target.paste(image, position)

    def composite(self) -> Image.Image:
        """
        text

        Returns:
            text RGBA text
        """
        # text
        result = self._main_layer.copy()

        # text
        if self.config.structure_layer_opacity < 1.0:
            structure = self._structure_layer.copy()
            # text
            alpha = structure.split()[3]
            alpha = alpha.point(lambda x: int(x * self.config.structure_layer_opacity))
            structure.putalpha(alpha)
            result = Image.alpha_composite(result, structure)
        else:
            result = Image.alpha_composite(result, self._structure_layer)

        # text
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
        """text"""
        result = self.composite()
        result.save(path, format)

    def save_layers(self, prefix: str):
        """text"""
        self._main_layer.save(f"{prefix}_main.png")
        self._structure_layer.save(f"{prefix}_structure.png")
        self._anchor_layer.save(f"{prefix}_anchor.png")
        self.composite().save(f"{prefix}_composite.png")

    def get_content_regions(self) -> List[Dict[str, Any]]:
        """text"""
        return self._content_regions.copy()

    def to_numpy(self) -> np.ndarray:
        """text numpy text"""
        return np.array(self.composite())

    def reset(self):
        """text"""
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
