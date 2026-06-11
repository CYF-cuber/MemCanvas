"""
CanvasSlicer - text

textDeepSeek-OCRtext：
1. text: 512/640/1024/1280 -> 64/100/256/400 vision tokens
2. Gundamtext: n×640×640 patches + 1×1024×1024 global view
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import math


@dataclass
class SliceConfig:
    """text"""
    # text (text)
    base_size: int = 1024
    # patchtext
    patch_size: int = 640
    # textcroptext (Gundamtext)
    crop_mode: bool = True
    # vision encodertextpatchtext (texttokentext)
    vision_patch_size: int = 64
    # textpatchtext
    max_patches: int = 16
    # text (0-1)
    overlap_ratio: float = 0.0


@dataclass
class SliceResult:
    """text"""
    # text (base_size × base_size)
    global_view: Image.Image
    # textpatchestext (patch_size × patch_size each)
    patches: List[Image.Image]
    # textpatchtext (x, y, w, h) text
    patch_positions: List[Tuple[int, int, int, int]]
    # text
    original_size: Tuple[int, int]
    # textvision tokentext
    estimated_tokens: int
    # text
    config: SliceConfig


class CanvasSlicer:
    """
    text

    text:
    1. text: text
    2. Gundamtext: textpatches + text
    """

    # textvision tokentext
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
        text

        Args:
            source: textPIL Image

        Returns:
            SliceResult textpatches
        """
        # text
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
        """text"""
        target_size = self.config.base_size

        # text
        global_view = image.resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS
        )

        # texttokentext
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
        Gundamtext: n×patch_size patches + 1×base_size global
        """
        w, h = original_size
        patch_size = self.config.patch_size
        base_size = self.config.base_size

        # 1. text
        global_view = self._create_global_view(image, base_size)

        # 2. textpatches
        patches, positions = self._create_patches(image, patch_size)

        # 3. texttokentext
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
        """text (text，text)"""
        w, h = image.size

        # text
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # text
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # text
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
        textpatches

        text，text
        """
        w, h = image.size
        patches = []
        positions = []

        # text (text)
        stride = int(patch_size * (1 - self.config.overlap_ratio))
        stride = max(stride, 1)

        # text
        cols = max(1, math.ceil((w - patch_size) / stride) + 1) if w > patch_size else 1
        rows = max(1, math.ceil((h - patch_size) / stride) + 1) if h > patch_size else 1

        # textpatchtext
        total_patches = cols * rows
        if total_patches > self.config.max_patches:
            # textpatches
            scale = math.sqrt(self.config.max_patches / total_patches)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
            cols = max(1, math.ceil((w - patch_size) / stride) + 1) if w > patch_size else 1
            rows = max(1, math.ceil((h - patch_size) / stride) + 1) if h > patch_size else 1

        for row in range(rows):
            for col in range(cols):
                # textpatchtext
                x = min(col * stride, max(0, w - patch_size))
                y = min(row * stride, max(0, h - patch_size))

                # textpatch
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
        """textpatch，text"""
        w, h = image.size

        # text
        actual_w = min(size, w - x)
        actual_h = min(size, h - y)

        # text
        crop = image.crop((x, y, x + actual_w, y + actual_h))

        # textsize，text
        if actual_w < size or actual_h < size:
            result = Image.new('RGB', (size, size), (255, 255, 255))
            result.paste(crop, (0, 0))
            return result

        return crop

    @classmethod
    def get_mode_config(cls, mode: str) -> SliceConfig:
        """text"""
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
        """textvision tokentext"""
        if not config.crop_mode:
            return (config.base_size // config.vision_patch_size) ** 2

        # Gundamtext
        global_tokens = (config.base_size // config.vision_patch_size) ** 2

        # textpatchestext
        patch_size = config.patch_size
        stride = int(patch_size * (1 - config.overlap_ratio))
        cols = max(1, math.ceil((width - patch_size) / stride) + 1) if width > patch_size else 1
        rows = max(1, math.ceil((height - patch_size) / stride) + 1) if height > patch_size else 1
        num_patches = min(cols * rows, config.max_patches)

        patch_tokens = num_patches * (patch_size // config.vision_patch_size) ** 2

        return global_tokens + patch_tokens
