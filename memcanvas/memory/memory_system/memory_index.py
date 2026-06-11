"""
MemoryIndex - memory indextext

provides fast vector-based retrieval indexes，text：
- in-memory index（NanoVectorDBtext）
- FAISStext（large scale）
- supports extension to other backends
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from pathlib import Path
import numpy as np
import json
from datetime import datetime


@dataclass
class IndexConfig:
    """text"""
    # text: memory, faiss, hnswlib
    index_type: str = "memory"
    # vector dimension
    vector_dim: int = 1024
    # similarity metric: cosine, l2, ip
    metric: str = "cosine"
    # index file path
    index_path: Optional[str] = None
    # HNSWtext
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    # whether to auto-save
    auto_save: bool = True


@dataclass
class SearchResult:
    """search result"""
    memory_id: str
    score: float
    rank: int


class MemoryIndex:
    """
    memory vector index

    supports multiple index backends：
    1. memory: simple in-memory vector index（suitable for small scale）
    2. faiss: Facebooktextvectortext（textlarge scale）
    3. hnswlib: high-performance approximate nearest-neighbor index
    """

    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()

        # text
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_list: List[str] = []

        # text（FAISStext）
        self._external_index = None

        self._init_index()

    def _init_index(self):
        """initialize index"""
        if self.config.index_type == "faiss":
            self._init_faiss()
        elif self.config.index_type == "hnswlib":
            self._init_hnswlib()
        # memorytextdoes not need special initialization

        # load if a persistence path is configured
        if self.config.index_path:
            self.load(self.config.index_path)

    def _init_faiss(self):
        """initialize FAISS index"""
        try:
            import faiss

            if self.config.metric == "cosine":
                # for cosine similarity, normalize first and then use inner product
                self._external_index = faiss.IndexFlatIP(self.config.vector_dim)
            elif self.config.metric == "l2":
                self._external_index = faiss.IndexFlatL2(self.config.vector_dim)
            else:
                self._external_index = faiss.IndexFlatIP(self.config.vector_dim)

        except ImportError:
            print("Warning: faiss not installed, falling back to memory index")
            self.config.index_type = "memory"

    def _init_hnswlib(self):
        """initialize HNSWlib index"""
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
        add vector to index

        Args:
            memory_id: memory ID
            vector: vector [dim]
        """
        vector = np.asarray(vector, dtype=np.float32)

        if self.config.metric == "cosine":
            # text
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
        textvector

        text：FAISStext，textrebuild index
        """
        if memory_id in self._vectors:
            del self._vectors[memory_id]

        if memory_id in self._id_list:
            self._id_list.remove(memory_id)

        # textFAISS，text
        if self.config.index_type == "faiss":
            self._rebuild_faiss()

    def _rebuild_faiss(self):
        """textFAISStext"""
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
        textvector

        Args:
            query_vector: textvector [dim]
            top_k: text
            threshold: text

        Returns:
            SearchResulttext
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
        """in-memory indextext"""
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

        # text
        scores.sort(key=lambda x: x[1], reverse=True)

        # texttop_k
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
        """FAISStext"""
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
                score = -score  # L2text

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
        """HNSWlibtext"""
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

            # HNSWlibtext
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
        """textvector"""
        return self._vectors.get(memory_id)

    def save(self, path: str):
        """text"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # textvector
        vectors_data = {
            mid: vec.tolist() for mid, vec in self._vectors.items()
        }
        with open(path / "vectors.json", 'w') as f:
            json.dump(vectors_data, f)

        # textIDtext
        with open(path / "ids.json", 'w') as f:
            json.dump(self._id_list, f)

        # text
        with open(path / "config.json", 'w') as f:
            json.dump({
                "index_type": self.config.index_type,
                "vector_dim": self.config.vector_dim,
                "metric": self.config.metric,
                "count": len(self._vectors),
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)

        # text
        if self.config.index_type == "faiss" and self._external_index:
            import faiss
            faiss.write_index(self._external_index, str(path / "faiss.index"))

        elif self.config.index_type == "hnswlib" and self._external_index:
            self._external_index.save_index(str(path / "hnsw.index"))

    def load(self, path: str):
        """textload index"""
        path = Path(path)
        if not path.exists():
            return

        # textvector
        vectors_path = path / "vectors.json"
        if vectors_path.exists():
            with open(vectors_path, 'r') as f:
                vectors_data = json.load(f)
                self._vectors = {
                    mid: np.array(vec, dtype=np.float32)
                    for mid, vec in vectors_data.items()
                }

        # textIDtext
        ids_path = path / "ids.json"
        if ids_path.exists():
            with open(ids_path, 'r') as f:
                self._id_list = json.load(f)

        # text
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
