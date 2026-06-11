"""
MemoryToken - textTokentext

text，text:
- vision tokens: text
- key embedding: textvector
- metadata: text

textDeepSeektextAlignertext。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import numpy as np
import json


@dataclass
class MemoryMeta:
    """text"""
    memory_id: str
    created_at: datetime
    source: List[str]
    modalities: List[str]
    # text
    canvas_size: tuple
    # text
    num_patches: int
    total_tokens: int
    # text
    extra: Dict[str, Any] = field(default_factory=dict)
    # textAlignertext
    is_aligned: bool = False
    # text
    compress_mode: str = "none"


@dataclass
class MemoryToken:
    """textTokentext"""
    # Vision tokens [total_tokens, hidden_dim]
    tokens: np.ndarray
    # Key embeddingtext [key_dim]
    key_embedding: np.ndarray
    # text
    meta: MemoryMeta
    # textmask [total_tokens] True=text, False=text
    valid_mask: Optional[np.ndarray] = None

    def get_valid_tokens(self) -> np.ndarray:
        """texttokens"""
        if self.valid_mask is None:
            return self.tokens
        return self.tokens[self.valid_mask]

    def save(self, path: str):
        """text"""
        np.savez_compressed(
            path,
            tokens=self.tokens,
            key_embedding=self.key_embedding,
            valid_mask=self.valid_mask,
            meta=json.dumps(self._meta_to_dict())
        )

    def _meta_to_dict(self) -> dict:
        return {
            "memory_id": self.meta.memory_id,
            "created_at": self.meta.created_at.isoformat(),
            "source": self.meta.source,
            "modalities": self.meta.modalities,
            "canvas_size": self.meta.canvas_size,
            "num_patches": self.meta.num_patches,
            "total_tokens": self.meta.total_tokens,
            "extra": self.meta.extra,
            "is_aligned": self.meta.is_aligned,
            "compress_mode": self.meta.compress_mode
        }

    @classmethod
    def load(cls, path: str) -> "MemoryToken":
        """text"""
        data = np.load(path, allow_pickle=True)
        meta_dict = json.loads(str(data['meta']))
        meta = MemoryMeta(
            memory_id=meta_dict['memory_id'],
            created_at=datetime.fromisoformat(meta_dict['created_at']),
            source=meta_dict['source'],
            modalities=meta_dict['modalities'],
            canvas_size=tuple(meta_dict['canvas_size']),
            num_patches=meta_dict['num_patches'],
            total_tokens=meta_dict['total_tokens'],
            extra=meta_dict.get('extra', {}),
            is_aligned=meta_dict.get('is_aligned', False),
            compress_mode=meta_dict.get('compress_mode', 'none')
        )
        return cls(
            tokens=data['tokens'],
            key_embedding=data['key_embedding'],
            meta=meta,
            valid_mask=data['valid_mask'] if 'valid_mask' in data else None
        )


class MemoryTokenBuilder:
    """
    textMemoryTokentext

    text：
    1. text：textCLIPtexttokens
    2. DeepSeektext：CLIP + Alignertext + text
    """

    # text (text)
    BLANK_THRESHOLD = 100

    def __init__(
        self,
        vision_extractor=None,
        use_deepseek_style: bool = False,
        aligner_output_dim: int = 1024,
        compress_mode: str = "none",
        target_tokens: int = 64,
        device: str = "cuda"
    ):
        """
        Args:
            vision_extractor: textVisionTokenExtractortext
            use_deepseek_style: textDeepSeektext
            aligner_output_dim: Alignertext
            compress_mode: text (none, pooling, resampler, conv)
            target_tokens: texttokentext（textresampler）
            device: text
        """
        self.vision_extractor = vision_extractor
        self.use_deepseek_style = use_deepseek_style
        self.aligner_output_dim = aligner_output_dim
        self.compress_mode = compress_mode
        self.target_tokens = target_tokens
        self.device = device

        # textDeepSeektextextractor，text
        if use_deepseek_style and vision_extractor is None:
            self._init_deepseek_style_extractor()

    def _init_deepseek_style_extractor(self):
        """textDeepSeektextextractor"""
        from .vision_extractor import create_deepseek_style_extractor
        self.vision_extractor = create_deepseek_style_extractor(
            output_dim=self.aligner_output_dim,
            compress_mode=self.compress_mode,
            target_tokens=self.target_tokens,
            device=self.device
        )

    def build_from_canvas(
        self,
        canvas,
        slice_result,
        vision_tokens=None,
        memory_id: Optional[str] = None
    ) -> MemoryToken:
        """
        textMemoryToken

        Args:
            canvas: textobjecttextPIL.Image
            slice_result: SliceResulttext
            vision_tokens: textVisionTokens（text）
            memory_id: memory ID（text）
        """
        from PIL import Image

        # textcanvastext
        if hasattr(canvas, 'get_image'):
            canvas_img = canvas.get_image()
        elif isinstance(canvas, Image.Image):
            canvas_img = canvas
        else:
            raise ValueError("Invalid canvas type")

        # textvision_tokens，text
        if vision_tokens is None and self.vision_extractor is not None:
            vision_tokens = self.vision_extractor.extract(slice_result)

        # text
        valid_mask = self._detect_valid_regions(slice_result)

        # textkey embedding
        if vision_tokens is not None:
            key_emb = vision_tokens.key_embedding
            if key_emb is None:
                key_emb = vision_tokens.global_tokens.mean(axis=0)
            tokens = vision_tokens.all_tokens
            is_aligned = vision_tokens.is_aligned
            is_compressed = vision_tokens.is_compressed
        else:
            key_emb = np.zeros(768)  # placeholder
            tokens = np.zeros((1, 768))
            is_aligned = False
            is_compressed = False

        # textmeta
        meta = MemoryMeta(
            memory_id=memory_id or getattr(canvas, 'metadata', {}).get('memory_id', 'unknown'),
            created_at=datetime.now(),
            source=getattr(canvas, 'metadata', {}).get('source', []),
            modalities=getattr(canvas, 'metadata', {}).get('modalities', []),
            canvas_size=canvas_img.size,
            num_patches=len(slice_result.patches),
            total_tokens=tokens.shape[0],
            extra={
                "hidden_dim": tokens.shape[-1] if len(tokens.shape) > 1 else 0,
                "global_tokens": vision_tokens.num_global_tokens if vision_tokens else 0,
                "patch_tokens": vision_tokens.num_patch_tokens if vision_tokens else 0
            },
            is_aligned=is_aligned,
            compress_mode=self.compress_mode if is_compressed else "none"
        )

        return MemoryToken(
            tokens=tokens,
            key_embedding=key_emb,
            meta=meta,
            valid_mask=valid_mask
        )

    def _detect_valid_regions(self, slice_result) -> np.ndarray:
        """text"""
        valid_list = []

        # text
        valid_list.append(not self._is_blank(slice_result.global_view))

        # textpatch
        for patch in slice_result.patches:
            valid_list.append(not self._is_blank(patch))

        return np.array(valid_list, dtype=bool)

    def _is_blank(self, image) -> bool:
        """text"""
        arr = np.array(image)
        variance = arr.var()
        return variance < self.BLANK_THRESHOLD


def create_deepseek_memory_builder(
    output_dim: int = 1024,
    compress_mode: str = "none",
    target_tokens: int = 64,
    device: str = "cuda"
) -> MemoryTokenBuilder:
    """
    textDeepSeektextMemoryTokentext

    Args:
        output_dim: text
        compress_mode: text
        target_tokens: texttokentext
        device: text

    Returns:
        textMemoryTokenBuilder
    """
    return MemoryTokenBuilder(
        use_deepseek_style=True,
        aligner_output_dim=output_dim,
        compress_mode=compress_mode,
        target_tokens=target_tokens,
        device=device
    )
