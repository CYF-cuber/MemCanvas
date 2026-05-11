"""
Canvas Slicer Module

实现类似DeepSeek-OCR的画布切分和vision token提取逻辑。

增强功能：
- DeepSeek风格的CLIP Aligner投影
- Token压缩（resampler/pooling/conv）
"""

from .canvas_slicer import CanvasSlicer, SliceConfig, SliceResult
from .vision_extractor import (
    VisionTokenExtractor, TokenExtractionConfig, VisionTokens,
    create_deepseek_style_extractor
)
from .memory_token import MemoryToken, MemoryMeta, MemoryTokenBuilder
from .clip_aligner import (
    CLIPAligner, TokenCompressor, CLIPVisionProcessor,
    AlignerConfig, CompressorConfig,
    create_deepseek_style_processor
)

__all__ = [
    # Canvas Slicer
    'CanvasSlicer', 'SliceConfig', 'SliceResult',
    # Vision Extractor
    'VisionTokenExtractor', 'TokenExtractionConfig', 'VisionTokens',
    'create_deepseek_style_extractor',
    # Memory Token
    'MemoryToken', 'MemoryMeta', 'MemoryTokenBuilder',
    # CLIP Aligner (DeepSeek风格)
    'CLIPAligner', 'TokenCompressor', 'CLIPVisionProcessor',
    'AlignerConfig', 'CompressorConfig',
    'create_deepseek_style_processor'
]
