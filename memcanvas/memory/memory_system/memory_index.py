"""
MemoryIndex - 记忆索引模块

提供基于向量的快速检索索引，支持：
- 内存索引（NanoVectorDB风格）
- FAISS索引（大规模）
- 支持扩展到其他后端
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from pathlib import Path
import numpy as np
import json
from datetime import datetime


@dataclass
class IndexConfig:
    """索引配置"""
    # 索引类型: memory, faiss, hnswlib
    index_type: str = "memory"
    # 向量维度
    vector_dim: int = 1024
    # 相似度度量: cosine, l2, ip
    metric: str = "cosine"
    # 索引文件路径
    index_path: Optional[str] = None
    # HNSW参数
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    # 是否自动保存
    auto_save: bool = True


@dataclass
class SearchResult:
    """搜索结果"""
    memory_id: str
    score: float
    rank: int


class MemoryIndex:
    """
    记忆向量索引

    支持多种索引后端：
    1. memory: 内存中的简单向量索引（适合小规模）
    2. faiss: Facebook的向量索引库（适合大规模）
    3. hnswlib: 高性能近似最近邻索引
    """

    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()

        # 内存存储
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_list: List[str] = []

        # 外部索引（FAISS等）
        self._external_index = None

        self._init_index()

    def _init_index(self):
        """初始化索引"""
        if self.config.index_type == "faiss":
            self._init_faiss()
        elif self.config.index_type == "hnswlib":
            self._init_hnswlib()
        # memory类型不需要特殊初始化

        # 如果有持久化路径，尝试加载
        if self.config.index_path:
            self.load(self.config.index_path)

    def _init_faiss(self):
        """初始化FAISS索引"""
        try:
            import faiss

            if self.config.metric == "cosine":
                # 对于余弦相似度，先normalize再用内积
                self._external_index = faiss.IndexFlatIP(self.config.vector_dim)
            elif self.config.metric == "l2":
                self._external_index = faiss.IndexFlatL2(self.config.vector_dim)
            else:
                self._external_index = faiss.IndexFlatIP(self.config.vector_dim)

        except ImportError:
            print("Warning: faiss not installed, falling back to memory index")
            self.config.index_type = "memory"

    def _init_hnswlib(self):
        """初始化HNSWlib索引"""
        try:
            import hnswlib

            space = 'cosine' if self.config.metric == 'cosine' else 'l2'
            self._external_index = hnswlib.Index(space=space, dim=self.config.vector_dim)
            self._external_index.init_index(
                max_elements=100000,
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_m
            )
            self._external_index.set_ef(self.config.hnsw_ef_search)

        except ImportError:
            print("Warning: hnswlib not installed, falling back to memory index")
            self.config.index_type = "memory"

    def add(self, memory_id: str, vector: np.ndarray):
        """
        添加向量到索引

        Args:
            memory_id: 记忆ID
            vector: 向量 [dim]
        """
        vector = np.asarray(vector, dtype=np.float32)

        if self.config.metric == "cosine":
            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        if self.config.index_type == "memory":
            self._vectors[memory_id] = vector
            if memory_id not in self._id_list:
                self._id_list.append(memory_id)

        elif self.config.index_type == "faiss":
            self._vectors[memory_id] = vector
            if memory_id not in self._id_list:
                self._id_list.append(memory_id)
                self._external_index.add(vector.reshape(1, -1))

        elif self.config.index_type == "hnswlib":
            idx = len(self._id_list)
            self._vectors[memory_id] = vector
            if memory_id not in self._id_list:
                self._id_list.append(memory_id)
                self._external_index.add_items(vector.reshape(1, -1), [idx])

    def remove(self, memory_id: str):
        """
        从索引移除向量

        注意：FAISS不支持删除，需要重建索引
        """
        if memory_id in self._vectors:
            del self._vectors[memory_id]

        if memory_id in self._id_list:
            self._id_list.remove(memory_id)

        # 对于FAISS，标记需要重建
        if self.config.index_type == "faiss":
            self._rebuild_faiss()

    def _rebuild_faiss(self):
        """重建FAISS索引"""
        if self.config.index_type != "faiss":
            return

        self._init_faiss()
        if self._vectors:
            vectors = np.stack([self._vectors[mid] for mid in self._id_list])
            self._external_index.add(vectors)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量 [dim]
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            SearchResult列表
        """
        if len(self._vectors) == 0:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32)

        if self.config.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        if self.config.index_type == "memory":
            return self._search_memory(query_vector, top_k, threshold)
        elif self.config.index_type == "faiss":
            return self._search_faiss(query_vector, top_k, threshold)
        elif self.config.index_type == "hnswlib":
            return self._search_hnswlib(query_vector, top_k, threshold)

        return []

    def _search_memory(
        self,
        query: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """内存索引搜索"""
        scores = []

        for memory_id, vector in self._vectors.items():
            if self.config.metric == "cosine":
                score = float(np.dot(query, vector))
            elif self.config.metric == "l2":
                score = -float(np.linalg.norm(query - vector))
            else:
                score = float(np.dot(query, vector))

            if score >= threshold:
                scores.append((memory_id, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        results = []
        for rank, (memory_id, score) in enumerate(scores[:top_k]):
            results.append(SearchResult(
                memory_id=memory_id,
                score=score,
                rank=rank
            ))

        return results

    def _search_faiss(
        self,
        query: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """FAISS搜索"""
        k = min(top_k, len(self._id_list))
        if k == 0:
            return []

        distances, indices = self._external_index.search(
            query.reshape(1, -1), k
        )

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self._id_list):
                continue

            score = float(dist)
            if self.config.metric == "l2":
                score = -score  # L2距离越小越好

            if score >= threshold:
                results.append(SearchResult(
                    memory_id=self._id_list[idx],
                    score=score,
                    rank=rank
                ))

        return results

    def _search_hnswlib(
        self,
        query: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[SearchResult]:
        """HNSWlib搜索"""
        k = min(top_k, len(self._id_list))
        if k == 0:
            return []

        indices, distances = self._external_index.knn_query(
            query.reshape(1, -1), k=k
        )

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(self._id_list):
                continue

            # HNSWlib返回的是距离
            if self.config.metric == "cosine":
                score = 1 - dist  # cosine distance -> similarity
            else:
                score = -dist

            if score >= threshold:
                results.append(SearchResult(
                    memory_id=self._id_list[idx],
                    score=score,
                    rank=rank
                ))

        return results

    def get_vector(self, memory_id: str) -> Optional[np.ndarray]:
        """获取记忆的向量"""
        return self._vectors.get(memory_id)

    def save(self, path: str):
        """保存索引到文件"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存向量
        vectors_data = {
            mid: vec.tolist() for mid, vec in self._vectors.items()
        }
        with open(path / "vectors.json", 'w') as f:
            json.dump(vectors_data, f)

        # 保存ID列表
        with open(path / "ids.json", 'w') as f:
            json.dump(self._id_list, f)

        # 保存配置
        with open(path / "config.json", 'w') as f:
            json.dump({
                "index_type": self.config.index_type,
                "vector_dim": self.config.vector_dim,
                "metric": self.config.metric,
                "count": len(self._vectors),
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)

        # 保存外部索引
        if self.config.index_type == "faiss" and self._external_index:
            import faiss
            faiss.write_index(self._external_index, str(path / "faiss.index"))

        elif self.config.index_type == "hnswlib" and self._external_index:
            self._external_index.save_index(str(path / "hnsw.index"))

    def load(self, path: str):
        """从文件加载索引"""
        path = Path(path)
        if not path.exists():
            return

        # 加载向量
        vectors_path = path / "vectors.json"
        if vectors_path.exists():
            with open(vectors_path, 'r') as f:
                vectors_data = json.load(f)
                self._vectors = {
                    mid: np.array(vec, dtype=np.float32)
                    for mid, vec in vectors_data.items()
                }

        # 加载ID列表
        ids_path = path / "ids.json"
        if ids_path.exists():
            with open(ids_path, 'r') as f:
                self._id_list = json.load(f)

        # 加载外部索引
        if self.config.index_type == "faiss":
            faiss_path = path / "faiss.index"
            if faiss_path.exists():
                import faiss
                self._external_index = faiss.read_index(str(faiss_path))

        elif self.config.index_type == "hnswlib":
            hnsw_path = path / "hnsw.index"
            if hnsw_path.exists():
                self._external_index.load_index(str(hnsw_path))

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._vectors
