"""
MemoryToken - 记忆Token数据结构

将画布转换为可存储的记忆单元，包含:
- vision tokens: 视觉特征
- key embedding: 用于检索的关键向量
- metadata: 元信息

支持DeepSeek风格的Aligner处理。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import numpy as np
import json


@dataclass
class MemoryMeta:
    """记忆元信息"""
    memory_id: str
    created_at: datetime
    source: List[str]
    modalities: List[str]
    # 原始画布尺寸
    canvas_size: tuple
    # 切分信息
    num_patches: int
    total_tokens: int
    # 自定义元数据
    extra: Dict[str, Any] = field(default_factory=dict)
    # 是否经过Aligner处理
    is_aligned: bool = False
    # 压缩模式
    compress_mode: str = "none"


@dataclass
class MemoryToken:
    """记忆Token单元"""
    # Vision tokens [total_tokens, hidden_dim]
    tokens: np.ndarray
    # Key embedding用于检索 [key_dim]
    key_embedding: np.ndarray
    # 元信息
    meta: MemoryMeta
    # 空白mask [total_tokens] True=有效, False=空白
    valid_mask: Optional[np.ndarray] = None

    def get_valid_tokens(self) -> np.ndarray:
        """获取非空白的有效tokens"""
        if self.valid_mask is None:
            return self.tokens
        return self.tokens[self.valid_mask]

    def save(self, path: str):
        """保存到文件"""
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
        """从文件加载"""
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
    构建MemoryToken的工具类

    支持两种模式：
    1. 基础模式：直接使用CLIP提取的tokens
    2. DeepSeek风格模式：CLIP + Aligner投影 + 可选压缩
    """

    # 空白检测阈值 (像素方差)
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
            vision_extractor: 现有的VisionTokenExtractor实例
            use_deepseek_style: 是否使用DeepSeek风格的处理
            aligner_output_dim: Aligner输出维度
            compress_mode: 压缩模式 (none, pooling, resampler, conv)
            target_tokens: 目标token数（用于resampler）
            device: 设备
        """
        self.vision_extractor = vision_extractor
        self.use_deepseek_style = use_deepseek_style
        self.aligner_output_dim = aligner_output_dim
        self.compress_mode = compress_mode
        self.target_tokens = target_tokens
        self.device = device

        # 如果需要DeepSeek风格但没有提供extractor，创建一个
        if use_deepseek_style and vision_extractor is None:
            self._init_deepseek_style_extractor()

    def _init_deepseek_style_extractor(self):
        """初始化DeepSeek风格的extractor"""
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
        从画布构建MemoryToken

        Args:
            canvas: 画布对象或PIL.Image
            slice_result: SliceResult切分结果
            vision_tokens: 预计算的VisionTokens（可选）
            memory_id: 记忆ID（可选）
        """
        from PIL import Image

        # 获取canvas图像
        if hasattr(canvas, 'get_image'):
            canvas_img = canvas.get_image()
        elif isinstance(canvas, Image.Image):
            canvas_img = canvas
        else:
            raise ValueError("Invalid canvas type")

        # 如果没有预计算的vision_tokens，则提取
        if vision_tokens is None and self.vision_extractor is not None:
            vision_tokens = self.vision_extractor.extract(slice_result)

        # 检测空白区域
        valid_mask = self._detect_valid_regions(slice_result)

        # 生成key embedding
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

        # 构建meta
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
        """检测非空白区域"""
        valid_list = []

        # 检测全局视图
        valid_list.append(not self._is_blank(slice_result.global_view))

        # 检测每个patch
        for patch in slice_result.patches:
            valid_list.append(not self._is_blank(patch))

        return np.array(valid_list, dtype=bool)

    def _is_blank(self, image) -> bool:
        """判断图像是否为空白"""
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
    创建DeepSeek风格的MemoryToken构建器

    Args:
        output_dim: 输出维度
        compress_mode: 压缩模式
        target_tokens: 目标token数
        device: 设备

    Returns:
        配置好的MemoryTokenBuilder
    """
    return MemoryTokenBuilder(
        use_deepseek_style=True,
        aligner_output_dim=output_dim,
        compress_mode=compress_mode,
        target_tokens=target_tokens,
        device=device
    )
