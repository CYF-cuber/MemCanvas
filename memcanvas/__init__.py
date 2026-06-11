"""MemCanvas: visual memory for lifelong multimodal agents."""

from .canvas import BlockType, ContentBlock, Layout, choose_best_layout, render_canvas, render_layout
from .forgetting import ProgressiveForgettingPolicy, QualityLevel
from .bank import MemoryBank, MemoryEntry
from .retrieval import hybrid_retrieval

__version__ = "0.1.0"

__all__ = [
    "BlockType",
    "ContentBlock",
    "Layout",
    "MemoryBank",
    "MemoryEntry",
    "ProgressiveForgettingPolicy",
    "QualityLevel",
    "choose_best_layout",
    "hybrid_retrieval",
    "render_canvas",
    "render_layout",
]
