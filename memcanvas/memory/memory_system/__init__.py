"""
Memory Canvas Memory System

基于 Vision Token 的多模态记忆系统，提供：
- 记忆存储（MemoryStore）
- 向量索引（MemoryIndex）
- 记忆检索（MemoryRetriever）
- 记忆管理（MemoryManager）
- 文本查询编码（TextQueryEncoder）

设计参考 MemVerse，但使用 Vision Token 而非文本描述。
"""

from .memory_store import MemoryStore, MemoryStoreConfig
from .memory_index import MemoryIndex, IndexConfig
from .memory_retriever import MemoryRetriever, RetrievalConfig, RetrievalResult
from .memory_manager import MemoryManager, ManagerConfig
from .memory_updater import MemoryUpdater, UpdateAction
from .text_query import TextQueryEncoder, TextQueryConfig, create_text_encoder

__all__ = [
    # Storage
    'MemoryStore', 'MemoryStoreConfig',
    # Index
    'MemoryIndex', 'IndexConfig',
    # Retrieval
    'MemoryRetriever', 'RetrievalConfig', 'RetrievalResult',
    # Manager
    'MemoryManager', 'ManagerConfig',
    # Updater
    'MemoryUpdater', 'UpdateAction',
    # Text Query
    'TextQueryEncoder', 'TextQueryConfig', 'create_text_encoder'
]
