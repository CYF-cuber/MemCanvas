"""
MemoryRetriever - text

text：
- vectortext
- text
- text
- text
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
    """text"""
    # text
    default_top_k: int = 10
    # text
    similarity_threshold: float = 0.2
    # text
    use_rerank: bool = False
    # text（text）
    rerank_model: Optional[str] = None
    # text（0text）
    time_decay_factor: float = 0.0
    # texttokentext
    max_token_budget: int = 30000


@dataclass
class RetrievalResult:
    """text"""
    memory_id: str
    memory: MemoryToken
    score: float
    rank: int
    # text
    retrieval_mode: str
    # text
    extra: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """
    text

    text：
    1. vector: textvectortext
    2. temporal: text
    3. metadata: text
    4. hybrid: text（vector + text + text）
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
        text

        Args:
            query_vector: textvector（vectortexthybridtext）
            top_k: text
            mode: text
            time_range: text (start_datetime, end_datetime)
            metadata_filter: text
            category: categorytext

        Returns:
            RetrievalResulttext
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
        """vectortext"""
        if query_vector is None:
            raise ValueError("query_vector is required for vector retrieval")

        # text
        search_results = self.index.search(
            query_vector,
            top_k=top_k * 2,  # text，text
            threshold=self.config.similarity_threshold
        )

        results = []
        for sr in search_results:
            # categorytext
            if category:
                ids_in_category = set(self.store.list_ids(category))
                if sr.memory_id not in ids_in_category:
                    continue

            # text
            memory = self.store.load(sr.memory_id)
            if memory is None:
                continue

            # text
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
        """text"""
        start_time, end_time = None, None
        if time_range:
            start_time, end_time = time_range

        results = []
        memory_ids = self.store.list_ids(category)

        # text
        candidates = []
        for mid in memory_ids:
            meta = self.store.get_meta(mid)
            if meta is None:
                continue

            created_at = meta.created_at

            # text
            if start_time and created_at < start_time:
                continue
            if end_time and created_at > end_time:
                continue

            candidates.append((mid, created_at))

        # text（text）
        candidates.sort(key=lambda x: x[1], reverse=True)

        # text
        for rank, (mid, created_at) in enumerate(candidates[:top_k]):
            memory = self.store.load(mid)
            if memory:
                results.append(RetrievalResult(
                    memory_id=mid,
                    memory=memory,
                    score=1.0,  # textsimilarity score
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
        """text"""
        if not metadata_filter:
            metadata_filter = {}

        results = []
        memory_ids = self.store.list_ids(category)

        for mid in memory_ids:
            meta = self.store.get_meta(mid)
            if meta is None:
                continue

            # text
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
        """text"""
        # textvectortext
        if query_vector is not None:
            candidates = self._retrieve_by_vector(
                query_vector, top_k * 3, category
            )
        else:
            # textvector，text
            candidates = self._retrieve_by_time(
                time_range, top_k * 3, category
            )

        results = []
        for candidate in candidates:
            memory = candidate.memory
            meta = memory.meta

            # text
            if time_range:
                start_time, end_time = time_range
                if start_time and meta.created_at < start_time:
                    continue
                if end_time and meta.created_at > end_time:
                    continue

            # text
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
        """text"""
        for key, value in filter_dict.items():
            # textkey，text "extra.key"
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
        """text"""
        now = datetime.now()
        age_days = (now - created_at).days

        # text
        decay = np.exp(-self.config.time_decay_factor * age_days)
        return score * decay

    def retrieve_recent(
        self,
        n: int = 10,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """textNmemories"""
        return self._retrieve_by_time(None, n, category)

    def retrieve_by_ids(self, memory_ids: List[str]) -> List[RetrievalResult]:
        """textIDtext"""
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
        textLLMtext

        Args:
            results: text
            max_tokens: texttokentext

        Returns:
            text
        """
        max_tokens = max_tokens or self.config.max_token_budget

        context = {
            "memories": [],
            "total_tokens": 0,
            "truncated": False
        }

        estimated_tokens = 0
        for result in results:
            # texttokentext（text：1 token ≈ 4 characters）
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
