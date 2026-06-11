"""
Canvas Slicer Module

textDeepSeek-OCRtextvision tokentext。

text：
- DeepSeektextCLIP Alignertext
- Tokentext（resampler/pooling/conv）
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
    # CLIP Aligner (DeepSeektext)
    'CLIPAligner', 'TokenCompressor', 'CLIPVisionProcessor',
    'AlignerConfig', 'CompressorConfig',
    'create_deepseek_style_processor'
]
