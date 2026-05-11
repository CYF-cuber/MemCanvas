"""
MemoryManager - 记忆系统管理器

整合存储、索引、检索、更新的统一接口。
提供完整的记忆生命周期管理。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import uuid

from .memory_store import MemoryStore, MemoryStoreConfig
from .memory_index import MemoryIndex, IndexConfig
from .memory_retriever import MemoryRetriever, RetrievalConfig, RetrievalResult
from .memory_updater import MemoryUpdater, UpdateConfig, UpdateAction, UpdateResult

from ...encoders.slicer import (
    CanvasSlicer,
    MemoryTokenBuilder,
    SliceConfig,
    TokenExtractionConfig,
    VisionTokenExtractor,
)
from ...encoders.slicer.memory_token import MemoryMeta, MemoryToken


@dataclass
class ManagerConfig:
    """管理器配置"""
    # 基础路径
    base_path: str = "./memory_system_data"
    # 存储配置
    store_config: Optional[MemoryStoreConfig] = None
    # 索引配置
    index_config: Optional[IndexConfig] = None
    # 检索配置
    retrieval_config: Optional[RetrievalConfig] = None
    # 更新配置
    update_config: Optional[UpdateConfig] = None
    # 默认分类
    default_category: str = "default"
    # 自动保存间隔（秒，0表示不自动保存）
    auto_save_interval: int = 300


class MemoryManager:
    """
    记忆系统管理器

    提供统一的API来管理记忆的完整生命周期：
    1. 创建记忆（从Canvas或直接创建）
    2. 存储和索引
    3. 检索和查询
    4. 更新和遗忘
    5. 维护和统计

    用法示例：
    ```python
    manager = MemoryManager()

    # 从Canvas创建并存储记忆
    memory_id = manager.create_from_canvas(canvas, slice_result)

    # 检索相关记忆
    results = manager.retrieve(query_vector, top_k=5)

    # 获取统计信息
    stats = manager.get_statistics()
    ```
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        self._init_components()

    def _init_components(self):
        """初始化各个组件"""
        base_path = Path(self.config.base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # 存储
        store_config = self.config.store_config or MemoryStoreConfig(
            storage_path=str(base_path / "store")
        )
        self.store = MemoryStore(store_config)

        # 索引
        index_config = self.config.index_config or IndexConfig(
            index_path=str(base_path / "index")
        )
        self.index = MemoryIndex(index_config)

        # 检索器
        retrieval_config = self.config.retrieval_config or RetrievalConfig()
        self.retriever = MemoryRetriever(self.store, self.index, retrieval_config)

        # 更新器
        update_config = self.config.update_config or UpdateConfig()
        self.updater = MemoryUpdater(self.store, self.index, update_config)

        # 同步索引（加载已存储的记忆到索引）
        self._sync_index()

    def _sync_index(self):
        """同步索引和存储"""
        for memory_id in self.store.list_ids():
            if memory_id not in self.index:
                memory = self.store.load(memory_id)
                if memory:
                    self.index.add(memory_id, memory.key_embedding)

    # ==================== 创建记忆 ====================

    def create_from_canvas(
        self,
        canvas: Union[Image.Image, Any],
        slice_result=None,
        vision_tokens=None,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        从Canvas创建并存储记忆

        Args:
            canvas: Canvas对象或PIL.Image
            slice_result: CanvasSlicer的切分结果
            vision_tokens: 预计算的VisionTokens（可选）
            memory_id: 记忆ID（可选，自动生成）
            category: 分类
            extra_meta: 额外元数据

        Returns:
            memory_id
        """
        category = category or self.config.default_category
        memory_id = memory_id or self._generate_id()

        # 获取图像
        if hasattr(canvas, 'get_image'):
            canvas_image = canvas.get_image()
        elif isinstance(canvas, Image.Image):
            canvas_image = canvas
        else:
            raise ValueError("Invalid canvas type")

        # 如果没有切分结果，自动切分
        if slice_result is None:
            slicer = CanvasSlicer(SliceConfig(base_size=1024, patch_size=640))
            slice_result = slicer.slice(canvas_image)

        # 构建MemoryToken
        builder = MemoryTokenBuilder()
        memory_token = builder.build_from_canvas(
            canvas=canvas_image,
            slice_result=slice_result,
            vision_tokens=vision_tokens,
            memory_id=memory_id
        )

        # 添加额外元数据
        if extra_meta:
            memory_token.meta.extra.update(extra_meta)

        # 处理并存储
        result = self.updater.process_new_memory(memory_token, category)

        return result.memory_id

    def create_from_tokens(
        self,
        tokens: np.ndarray,
        key_embedding: np.ndarray,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        直接从tokens创建记忆

        Args:
            tokens: vision tokens [seq_len, hidden_dim]
            key_embedding: key embedding [hidden_dim]
            memory_id: 记忆ID
            category: 分类
            source: 来源列表
            modalities: 模态列表
            extra_meta: 额外元数据

        Returns:
            memory_id
        """
        category = category or self.config.default_category
        memory_id = memory_id or self._generate_id()

        meta = MemoryMeta(
            memory_id=memory_id,
            created_at=datetime.now(),
            source=source or [],
            modalities=modalities or [],
            canvas_size=(0, 0),
            num_patches=0,
            total_tokens=tokens.shape[0],
            extra=extra_meta or {},
            is_aligned=True
        )

        memory_token = MemoryToken(
            tokens=tokens,
            key_embedding=key_embedding,
            meta=meta
        )

        result = self.updater.process_new_memory(memory_token, category)
        return result.memory_id

    # ==================== 检索记忆 ====================

    def retrieve(
        self,
        query_vector: Optional[np.ndarray] = None,
        query_image: Optional[Image.Image] = None,
        top_k: int = 10,
        mode: str = "vector",
        time_range: Optional[tuple] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        检索记忆

        Args:
            query_vector: 查询向量
            query_image: 查询图像（会自动提取特征）
            top_k: 返回数量
            mode: 检索模式 (vector, temporal, metadata, hybrid)
            time_range: 时间范围
            metadata_filter: 元数据过滤
            category: 分类过滤

        Returns:
            检索结果列表
        """
        # 如果提供了查询图像，提取特征
        if query_image is not None and query_vector is None:
            query_vector = self._extract_query_features(query_image)

        return self.retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k,
            mode=mode,
            time_range=time_range,
            metadata_filter=metadata_filter,
            category=category
        )

    def retrieve_recent(self, n: int = 10, category: Optional[str] = None) -> List[RetrievalResult]:
        """检索最近的N条记忆"""
        return self.retriever.retrieve_recent(n, category)

    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryToken]:
        """按ID检索单条记忆"""
        return self.store.load(memory_id)

    def _extract_query_features(self, image: Image.Image) -> np.ndarray:
        """从查询图像提取特征"""
        try:
            config = TokenExtractionConfig(device="cuda")
            extractor = VisionTokenExtractor(config)

            # 简单提取
            if image.mode != 'RGB':
                image = image.convert('RGB')

            tokens = extractor._extract_single(image)
            # 使用mean pooling作为查询向量
            return tokens.mean(axis=0)

        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return np.zeros(768)

    # ==================== 更新和删除 ====================

    def update_memory(
        self,
        memory_id: str,
        new_tokens: Optional[np.ndarray] = None,
        new_key_embedding: Optional[np.ndarray] = None,
        new_meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新记忆

        Args:
            memory_id: 记忆ID
            new_tokens: 新的tokens（可选）
            new_key_embedding: 新的key embedding（可选）
            new_meta: 新的元数据（可选）

        Returns:
            是否成功
        """
        memory = self.store.load(memory_id)
        if memory is None:
            return False

        # 更新tokens
        if new_tokens is not None:
            memory.tokens = new_tokens
            memory.meta.total_tokens = new_tokens.shape[0]

        # 更新key embedding
        if new_key_embedding is not None:
            memory.key_embedding = new_key_embedding
            self.index.remove(memory_id)
            self.index.add(memory_id, new_key_embedding)

        # 更新元数据
        if new_meta:
            memory.meta.extra.update(new_meta)

        # 重新保存
        # 需要先获取category
        index_info = self.store._index.get(memory_id, "")
        category = index_info.split("/")[0] if "/" in index_info else "default"

        self.store.save(memory, category, overwrite=True)
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        return self.updater.forget_memory(memory_id)

    def delete_by_filter(
        self,
        time_range: Optional[tuple] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> int:
        """
        按条件批量删除

        Returns:
            删除数量
        """
        # 先检索符合条件的记忆
        results = self.retriever.retrieve(
            mode="hybrid" if metadata_filter else "temporal",
            time_range=time_range,
            metadata_filter=metadata_filter,
            category=category,
            top_k=10000  # 获取所有
        )

        deleted = 0
        for result in results:
            if self.delete_memory(result.memory_id):
                deleted += 1

        return deleted

    # ==================== 维护操作 ====================

    def save(self):
        """保存所有数据"""
        self.store._save_index()
        if self.config.index_config and self.config.index_config.index_path:
            self.index.save(self.config.index_config.index_path)

    def compact(self):
        """整理和压缩"""
        self.updater.compact()

    def backup(self, backup_path: str):
        """备份"""
        self.store.backup(backup_path)

    def clear(self, category: Optional[str] = None):
        """清空"""
        self.store.clear(category)
        if category is None:
            # 清空索引
            self.index._vectors.clear()
            self.index._id_list.clear()

    # ==================== 统计信息 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        store_stats = self.store.get_statistics()
        updater_stats = self.updater.get_statistics()

        return {
            "storage": store_stats,
            "index": {
                "type": self.index.config.index_type,
                "size": len(self.index),
                "vector_dim": self.index.config.vector_dim
            },
            "updater": updater_stats,
            "config": {
                "base_path": self.config.base_path,
                "default_category": self.config.default_category
            }
        }

    def list_memories(
        self,
        category: Optional[str] = None,
        include_meta: bool = True
    ) -> List[Dict[str, Any]]:
        """列出所有记忆"""
        memories = []
        for mid in self.store.list_ids(category):
            info = {"memory_id": mid}
            if include_meta:
                meta = self.store.get_meta(mid)
                if meta:
                    info["created_at"] = meta.created_at.isoformat()
                    info["source"] = meta.source
                    info["modalities"] = meta.modalities
                    info["total_tokens"] = meta.total_tokens
            memories.append(info)
        return memories

    # ==================== 工具方法 ====================

    def _generate_id(self) -> str:
        """生成唯一ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique = str(uuid.uuid4())[:8]
        return f"mem_{timestamp}_{unique}"

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self.store


def create_memory_manager(
    base_path: str = "./memory_data",
    index_type: str = "memory",
    vector_dim: int = 1024,
    duplicate_threshold: float = 0.85,
    capacity_limit: int = 0
) -> MemoryManager:
    """
    快速创建记忆管理器

    Args:
        base_path: 数据存储路径
        index_type: 索引类型 (memory, faiss, hnswlib)
        vector_dim: 向量维度
        duplicate_threshold: 重复检测阈值
        capacity_limit: 容量限制

    Returns:
        MemoryManager实例
    """
    config = ManagerConfig(
        base_path=base_path,
        store_config=MemoryStoreConfig(
            storage_path=f"{base_path}/store"
        ),
        index_config=IndexConfig(
            index_type=index_type,
            vector_dim=vector_dim,
            index_path=f"{base_path}/index"
        ),
        update_config=UpdateConfig(
            duplicate_threshold=duplicate_threshold,
            capacity_limit=capacity_limit
        )
    )
    return MemoryManager(config)
