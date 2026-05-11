"""
MemoryRetriever - 记忆检索模块

提供多种检索模式：
- 向量相似度检索
- 时间范围检索
- 元数据过滤检索
- 混合检索
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from .memory_store import MemoryStore
from .memory_index import MemoryIndex, SearchResult

from ...encoders.slicer.memory_token import MemoryToken


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 默认返回数量
    default_top_k: int = 10
    # 相似度阈值
    similarity_threshold: float = 0.2
    # 是否使用重排序
    use_rerank: bool = False
    # 重排序模型（如果使用）
    rerank_model: Optional[str] = None
    # 时间衰减因子（0表示不使用）
    time_decay_factor: float = 0.0
    # 最大token预算
    max_token_budget: int = 30000


@dataclass
class RetrievalResult:
    """检索结果"""
    memory_id: str
    memory: MemoryToken
    score: float
    rank: int
    # 检索方式
    retrieval_mode: str
    # 额外信息
    extra: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """
    记忆检索器

    支持多种检索模式：
    1. vector: 基于向量相似度
    2. temporal: 基于时间范围
    3. metadata: 基于元数据过滤
    4. hybrid: 混合检索（向量 + 时间 + 元数据）
    """

    def __init__(
        self,
        store: MemoryStore,
        index: MemoryIndex,
        config: Optional[RetrievalConfig] = None
    ):
        self.store = store
        self.index = index
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query_vector: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        mode: Literal["vector", "temporal", "metadata", "hybrid"] = "vector",
        time_range: Optional[tuple] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        检索记忆

        Args:
            query_vector: 查询向量（vector和hybrid模式需要）
            top_k: 返回数量
            mode: 检索模式
            time_range: 时间范围 (start_datetime, end_datetime)
            metadata_filter: 元数据过滤条件
            category: 分类过滤

        Returns:
            RetrievalResult列表
        """
        top_k = top_k or self.config.default_top_k

        if mode == "vector":
            return self._retrieve_by_vector(query_vector, top_k, category)
        elif mode == "temporal":
            return self._retrieve_by_time(time_range, top_k, category)
        elif mode == "metadata":
            return self._retrieve_by_metadata(metadata_filter, top_k, category)
        elif mode == "hybrid":
            return self._retrieve_hybrid(
                query_vector, time_range, metadata_filter, top_k, category
            )

        return []

    def _retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """向量相似度检索"""
        if query_vector is None:
            raise ValueError("query_vector is required for vector retrieval")

        # 搜索索引
        search_results = self.index.search(
            query_vector,
            top_k=top_k * 2,  # 多检索一些，后面可能会过滤
            threshold=self.config.similarity_threshold
        )

        results = []
        for sr in search_results:
            # 分类过滤
            if category:
                ids_in_category = set(self.store.list_ids(category))
                if sr.memory_id not in ids_in_category:
                    continue

            # 加载记忆
            memory = self.store.load(sr.memory_id)
            if memory is None:
                continue

            # 应用时间衰减
            score = sr.score
            if self.config.time_decay_factor > 0:
                score = self._apply_time_decay(score, memory.meta.created_at)

            results.append(RetrievalResult(
                memory_id=sr.memory_id,
                memory=memory,
                score=score,
                rank=len(results),
                retrieval_mode="vector"
            ))

            if len(results) >= top_k:
                break

        return results

    def _retrieve_by_time(
        self,
        time_range: Optional[tuple],
        top_k: int,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """时间范围检索"""
        start_time, end_time = None, None
        if time_range:
            start_time, end_time = time_range

        results = []
        memory_ids = self.store.list_ids(category)

        # 收集符合时间范围的记忆
        candidates = []
        for mid in memory_ids:
            meta = self.store.get_meta(mid)
            if meta is None:
                continue

            created_at = meta.created_at

            # 时间范围过滤
            if start_time and created_at < start_time:
                continue
            if end_time and created_at > end_time:
                continue

            candidates.append((mid, created_at))

        # 按时间倒序排列（最新的在前）
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 加载记忆
        for rank, (mid, created_at) in enumerate(candidates[:top_k]):
            memory = self.store.load(mid)
            if memory:
                results.append(RetrievalResult(
                    memory_id=mid,
                    memory=memory,
                    score=1.0,  # 时间检索没有相似度分数
                    rank=rank,
                    retrieval_mode="temporal",
                    extra={"created_at": created_at.isoformat()}
                ))

        return results

    def _retrieve_by_metadata(
        self,
        metadata_filter: Optional[Dict[str, Any]],
        top_k: int,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """元数据过滤检索"""
        if not metadata_filter:
            metadata_filter = {}

        results = []
        memory_ids = self.store.list_ids(category)

        for mid in memory_ids:
            meta = self.store.get_meta(mid)
            if meta is None:
                continue

            # 检查元数据匹配
            if not self._match_metadata(meta, metadata_filter):
                continue

            memory = self.store.load(mid)
            if memory:
                results.append(RetrievalResult(
                    memory_id=mid,
                    memory=memory,
                    score=1.0,
                    rank=len(results),
                    retrieval_mode="metadata"
                ))

            if len(results) >= top_k:
                break

        return results

    def _retrieve_hybrid(
        self,
        query_vector: Optional[np.ndarray],
        time_range: Optional[tuple],
        metadata_filter: Optional[Dict[str, Any]],
        top_k: int,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """混合检索"""
        # 首先按向量检索更多候选
        if query_vector is not None:
            candidates = self._retrieve_by_vector(
                query_vector, top_k * 3, category
            )
        else:
            # 如果没有向量，退化为时间检索
            candidates = self._retrieve_by_time(
                time_range, top_k * 3, category
            )

        results = []
        for candidate in candidates:
            memory = candidate.memory
            meta = memory.meta

            # 时间范围过滤
            if time_range:
                start_time, end_time = time_range
                if start_time and meta.created_at < start_time:
                    continue
                if end_time and meta.created_at > end_time:
                    continue

            # 元数据过滤
            if metadata_filter:
                if not self._match_metadata(meta, metadata_filter):
                    continue

            candidate.retrieval_mode = "hybrid"
            candidate.rank = len(results)
            results.append(candidate)

            if len(results) >= top_k:
                break

        return results

    def _match_metadata(self, meta, filter_dict: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for key, value in filter_dict.items():
            # 支持嵌套key，如 "extra.key"
            if "." in key:
                parts = key.split(".")
                obj = meta
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    elif isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        return False
                if obj != value:
                    return False
            else:
                if not hasattr(meta, key):
                    return False
                if getattr(meta, key) != value:
                    return False

        return True

    def _apply_time_decay(self, score: float, created_at: datetime) -> float:
        """应用时间衰减"""
        now = datetime.now()
        age_days = (now - created_at).days

        # 指数衰减
        decay = np.exp(-self.config.time_decay_factor * age_days)
        return score * decay

    def retrieve_recent(
        self,
        n: int = 10,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """检索最近的N条记忆"""
        return self._retrieve_by_time(None, n, category)

    def retrieve_by_ids(self, memory_ids: List[str]) -> List[RetrievalResult]:
        """按ID列表检索"""
        results = []
        for rank, mid in enumerate(memory_ids):
            memory = self.store.load(mid)
            if memory:
                results.append(RetrievalResult(
                    memory_id=mid,
                    memory=memory,
                    score=1.0,
                    rank=rank,
                    retrieval_mode="direct"
                ))
        return results

    def get_context_for_llm(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        为LLM准备检索上下文

        Args:
            results: 检索结果
            max_tokens: 最大token数

        Returns:
            包含记忆信息的上下文字典
        """
        max_tokens = max_tokens or self.config.max_token_budget

        context = {
            "memories": [],
            "total_tokens": 0,
            "truncated": False
        }

        estimated_tokens = 0
        for result in results:
            # 估算token数（简单估算：1 token ≈ 4 characters）
            memory_tokens = result.memory.tokens.shape[0]
            estimated_tokens += memory_tokens

            if estimated_tokens > max_tokens:
                context["truncated"] = True
                break

            context["memories"].append({
                "id": result.memory_id,
                "score": result.score,
                "created_at": result.memory.meta.created_at.isoformat(),
                "modalities": result.memory.meta.modalities,
                "source": result.memory.meta.source,
                "tokens_shape": list(result.memory.tokens.shape),
                "is_aligned": result.memory.meta.is_aligned
            })

        context["total_tokens"] = estimated_tokens
        context["memory_count"] = len(context["memories"])

        return context
