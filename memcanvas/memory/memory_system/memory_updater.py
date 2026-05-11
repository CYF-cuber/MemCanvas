"""
MemoryUpdater - 记忆更新模块

提供记忆的更新和遗忘机制：
- 重复检测
- 记忆合并
- 记忆遗忘
- 记忆压缩
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import numpy as np

from .memory_store import MemoryStore
from .memory_index import MemoryIndex

from ...encoders.slicer.memory_token import MemoryMeta, MemoryToken


class UpdateAction(Enum):
    """更新动作"""
    ADD = "add"           # 添加新记忆
    UPDATE = "update"     # 更新现有记忆
    REMAIN = "remain"     # 保持不变（重复）
    MERGE = "merge"       # 合并记忆


@dataclass
class UpdateConfig:
    """更新配置"""
    # 重复检测相似度阈值
    duplicate_threshold: float = 0.85
    # 相似记忆合并阈值
    merge_threshold: float = 0.7
    # 是否自动检测重复
    auto_duplicate_check: bool = True
    # 遗忘策略: none, time_decay, capacity, importance
    forget_strategy: str = "none"
    # 容量限制（0表示无限制）
    capacity_limit: int = 0
    # 时间衰减天数（超过此天数的记忆可被遗忘）
    decay_days: int = 365
    # 重要性阈值（低于此值的记忆可被遗忘）
    importance_threshold: float = 0.3


@dataclass
class UpdateResult:
    """更新结果"""
    action: UpdateAction
    memory_id: str
    # 如果是更新或合并，原记忆ID
    original_id: Optional[str] = None
    # 相似度分数
    similarity: float = 0.0
    # 消息
    message: str = ""


class MemoryUpdater:
    """
    记忆更新器

    负责：
    1. 检测新记忆是否与现有记忆重复
    2. 决定是添加、更新还是合并记忆
    3. 执行记忆遗忘策略
    4. 记忆压缩和整理
    """

    def __init__(
        self,
        store: MemoryStore,
        index: MemoryIndex,
        config: Optional[UpdateConfig] = None
    ):
        self.store = store
        self.index = index
        self.config = config or UpdateConfig()

    def process_new_memory(
        self,
        new_memory: MemoryToken,
        category: str = "default"
    ) -> UpdateResult:
        """
        处理新记忆

        Args:
            new_memory: 新记忆
            category: 分类

        Returns:
            UpdateResult
        """
        memory_id = new_memory.meta.memory_id

        # 检查是否启用重复检测
        if not self.config.auto_duplicate_check or len(self.index) == 0:
            # 直接添加
            self.store.save(new_memory, category)
            self.index.add(memory_id, new_memory.key_embedding)
            return UpdateResult(
                action=UpdateAction.ADD,
                memory_id=memory_id,
                message="Added new memory"
            )

        # 检测重复和相似
        action, similar_id, similarity = self._detect_duplicate(
            new_memory.key_embedding
        )

        if action == UpdateAction.REMAIN:
            return UpdateResult(
                action=UpdateAction.REMAIN,
                memory_id=memory_id,
                original_id=similar_id,
                similarity=similarity,
                message=f"Duplicate detected (similarity={similarity:.3f})"
            )

        elif action == UpdateAction.UPDATE:
            # 更新现有记忆
            return self._update_existing(new_memory, similar_id, category)

        elif action == UpdateAction.MERGE:
            # 合并记忆
            return self._merge_memories(new_memory, similar_id, category)

        else:
            # 添加新记忆
            self.store.save(new_memory, category)
            self.index.add(memory_id, new_memory.key_embedding)

            # 检查容量限制
            self._check_capacity()

            return UpdateResult(
                action=UpdateAction.ADD,
                memory_id=memory_id,
                similarity=similarity,
                message="Added new memory"
            )

    def _detect_duplicate(
        self,
        query_embedding: np.ndarray
    ) -> Tuple[UpdateAction, Optional[str], float]:
        """
        检测是否重复

        Returns:
            (action, similar_memory_id, similarity)
        """
        # 搜索最相似的记忆
        results = self.index.search(query_embedding, top_k=1, threshold=0.0)

        if not results:
            return UpdateAction.ADD, None, 0.0

        most_similar = results[0]
        similarity = most_similar.score

        if similarity >= self.config.duplicate_threshold:
            # 高度相似，视为重复
            return UpdateAction.REMAIN, most_similar.memory_id, similarity

        elif similarity >= self.config.merge_threshold:
            # 中度相似，考虑合并
            return UpdateAction.MERGE, most_similar.memory_id, similarity

        else:
            # 相似度不高，添加新记忆
            return UpdateAction.ADD, most_similar.memory_id, similarity

    def _update_existing(
        self,
        new_memory: MemoryToken,
        existing_id: str,
        category: str
    ) -> UpdateResult:
        """更新现有记忆"""
        # 删除旧记忆
        self.store.delete(existing_id)
        self.index.remove(existing_id)

        # 添加新记忆
        new_id = new_memory.meta.memory_id
        self.store.save(new_memory, category)
        self.index.add(new_id, new_memory.key_embedding)

        return UpdateResult(
            action=UpdateAction.UPDATE,
            memory_id=new_id,
            original_id=existing_id,
            message=f"Updated memory {existing_id} -> {new_id}"
        )

    def _merge_memories(
        self,
        new_memory: MemoryToken,
        existing_id: str,
        category: str
    ) -> UpdateResult:
        """合并记忆"""
        existing_memory = self.store.load(existing_id)
        if existing_memory is None:
            # 如果现有记忆不存在，直接添加
            return self.process_new_memory(new_memory, category)

        # 合并tokens（简单拼接）
        merged_tokens = np.concatenate([
            existing_memory.tokens,
            new_memory.tokens
        ], axis=0)

        # 合并key embedding（加权平均）
        merged_embedding = (
            existing_memory.key_embedding + new_memory.key_embedding
        ) / 2

        # 合并元数据
        merged_meta = MemoryMeta(
            memory_id=f"merged_{new_memory.meta.memory_id}",
            created_at=datetime.now(),
            source=list(set(existing_memory.meta.source + new_memory.meta.source)),
            modalities=list(set(existing_memory.meta.modalities + new_memory.meta.modalities)),
            canvas_size=new_memory.meta.canvas_size,
            num_patches=existing_memory.meta.num_patches + new_memory.meta.num_patches,
            total_tokens=merged_tokens.shape[0],
            extra={
                "merged_from": [existing_id, new_memory.meta.memory_id],
                "merge_time": datetime.now().isoformat()
            },
            is_aligned=new_memory.meta.is_aligned,
            compress_mode=new_memory.meta.compress_mode
        )

        # 合并valid_mask
        merged_mask = None
        if existing_memory.valid_mask is not None and new_memory.valid_mask is not None:
            merged_mask = np.concatenate([
                existing_memory.valid_mask,
                new_memory.valid_mask
            ])

        merged_memory = MemoryToken(
            tokens=merged_tokens,
            key_embedding=merged_embedding,
            meta=merged_meta,
            valid_mask=merged_mask
        )

        # 删除旧记忆
        self.store.delete(existing_id)
        self.index.remove(existing_id)

        # 保存合并后的记忆
        self.store.save(merged_memory, category)
        self.index.add(merged_meta.memory_id, merged_embedding)

        return UpdateResult(
            action=UpdateAction.MERGE,
            memory_id=merged_meta.memory_id,
            original_id=existing_id,
            message=f"Merged {existing_id} + {new_memory.meta.memory_id}"
        )

    def _check_capacity(self):
        """检查容量限制，必要时执行遗忘"""
        if self.config.capacity_limit <= 0:
            return

        current_count = len(self.store)
        if current_count <= self.config.capacity_limit:
            return

        # 需要遗忘一些记忆
        num_to_forget = current_count - self.config.capacity_limit

        if self.config.forget_strategy == "time_decay":
            self._forget_by_time(num_to_forget)
        elif self.config.forget_strategy == "importance":
            self._forget_by_importance(num_to_forget)
        else:
            # 默认：删除最老的
            self._forget_oldest(num_to_forget)

    def _forget_oldest(self, n: int):
        """删除最老的N条记忆"""
        memories_with_time = []
        for mid in self.store.list_ids():
            meta = self.store.get_meta(mid)
            if meta:
                memories_with_time.append((mid, meta.created_at))

        # 按时间排序，最老的在前
        memories_with_time.sort(key=lambda x: x[1])

        # 删除最老的N条
        for mid, _ in memories_with_time[:n]:
            self.store.delete(mid)
            self.index.remove(mid)

    def _forget_by_time(self, n: int):
        """按时间衰减删除记忆"""
        cutoff_date = datetime.now() - timedelta(days=self.config.decay_days)

        deleted = 0
        for mid in list(self.store.list_ids()):
            if deleted >= n:
                break

            meta = self.store.get_meta(mid)
            if meta and meta.created_at < cutoff_date:
                self.store.delete(mid)
                self.index.remove(mid)
                deleted += 1

        # 如果还需要删除更多，删除最老的
        if deleted < n:
            self._forget_oldest(n - deleted)

    def _forget_by_importance(self, n: int):
        """按重要性删除记忆（基于embedding范数作为简单的重要性指标）"""
        memories_with_importance = []
        for mid in self.store.list_ids():
            vec = self.index.get_vector(mid)
            if vec is not None:
                # 使用embedding范数作为重要性代理
                importance = float(np.linalg.norm(vec))
                memories_with_importance.append((mid, importance))

        # 按重要性排序，最不重要的在前
        memories_with_importance.sort(key=lambda x: x[1])

        # 删除最不重要的N条
        for mid, _ in memories_with_importance[:n]:
            self.store.delete(mid)
            self.index.remove(mid)

    def forget_memory(self, memory_id: str) -> bool:
        """手动删除指定记忆"""
        if memory_id not in self.store:
            return False

        self.store.delete(memory_id)
        self.index.remove(memory_id)
        return True

    def compact(self):
        """
        整理和压缩记忆

        - 移除无效索引
        - 重建索引（如果使用FAISS）
        - 清理孤立文件
        """
        # 同步索引和存储
        store_ids = set(self.store.list_ids())
        index_ids = set(self.index._id_list)

        # 移除索引中不存在于存储的记录
        for mid in index_ids - store_ids:
            self.index.remove(mid)

        # 为存储中存在但索引中没有的记录重建索引
        for mid in store_ids - index_ids:
            memory = self.store.load(mid)
            if memory:
                self.index.add(mid, memory.key_embedding)

    def get_statistics(self) -> Dict[str, Any]:
        """获取更新统计"""
        return {
            "total_memories": len(self.store),
            "index_size": len(self.index),
            "config": {
                "duplicate_threshold": self.config.duplicate_threshold,
                "merge_threshold": self.config.merge_threshold,
                "forget_strategy": self.config.forget_strategy,
                "capacity_limit": self.config.capacity_limit
            }
        }
