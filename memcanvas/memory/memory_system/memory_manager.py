"""
MemoryManager - text

text、text、text、text。
text。
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
    """text"""
    # text
    base_path: str = "./memory_system_data"
    # storage configuration
    store_config: Optional[MemoryStoreConfig] = None
    # text
    index_config: Optional[IndexConfig] = None
    # text
    retrieval_config: Optional[RetrievalConfig] = None
    # update configuration
    update_config: Optional[UpdateConfig] = None
    # textcategory
    default_category: str = "default"
    # text（seconds，0text）
    auto_save_interval: int = 300


class MemoryManager:
    """
    text

    textAPItext：
    1. text（textCanvastext）
    2. text
    3. text
    4. text
    5. text

    text：
    ```python
    manager = MemoryManager()

    # textCanvastext
    memory_id = manager.create_from_canvas(canvas, slice_result)

    # text
    results = manager.retrieve(query_vector, top_k=5)

    # text
    stats = manager.get_statistics()
    ```
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        self._init_components()

    def _init_components(self):
        """text"""
        base_path = Path(self.config.base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # text
        store_config = self.config.store_config or MemoryStoreConfig(
            storage_path=str(base_path / "store")
        )
        self.store = MemoryStore(store_config)

        # text
        index_config = self.config.index_config or IndexConfig(
            index_path=str(base_path / "index")
        )
        self.index = MemoryIndex(index_config)

        # text
        retrieval_config = self.config.retrieval_config or RetrievalConfig()
        self.retriever = MemoryRetriever(self.store, self.index, retrieval_config)

        # text
        update_config = self.config.update_config or UpdateConfig()
        self.updater = MemoryUpdater(self.store, self.index, update_config)

        # text（text）
        self._sync_index()

    def _sync_index(self):
        """synchronize index and storage"""
        for memory_id in self.store.list_ids():
            if memory_id not in self.index:
                memory = self.store.load(memory_id)
                if memory:
                    self.index.add(memory_id, memory.key_embedding)

    # ==================== text ====================

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
        textCanvastext

        Args:
            canvas: CanvasobjecttextPIL.Image
            slice_result: CanvasSlicertext
            vision_tokens: textVisionTokens（text）
            memory_id: memory ID（text，text）
            category: category
            extra_meta: text

        Returns:
            memory_id
        """
        category = category or self.config.default_category
        memory_id = memory_id or self._generate_id()

        # text
        if hasattr(canvas, 'get_image'):
            canvas_image = canvas.get_image()
        elif isinstance(canvas, Image.Image):
            canvas_image = canvas
        else:
            raise ValueError("Invalid canvas type")

        # text，text
        if slice_result is None:
            slicer = CanvasSlicer(SliceConfig(base_size=1024, patch_size=640))
            slice_result = slicer.slice(canvas_image)

        # textMemoryToken
        builder = MemoryTokenBuilder()
        memory_token = builder.build_from_canvas(
            canvas=canvas_image,
            slice_result=slice_result,
            vision_tokens=vision_tokens,
            memory_id=memory_id
        )

        # text
        if extra_meta:
            memory_token.meta.extra.update(extra_meta)

        # text
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
        texttokenstext

        Args:
            tokens: vision tokens [seq_len, hidden_dim]
            key_embedding: key embedding [hidden_dim]
            memory_id: memory ID
            category: category
            source: text
            modalities: text
            extra_meta: text

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

    # ==================== text ====================

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
        text

        Args:
            query_vector: textvector
            query_image: text（text）
            top_k: text
            mode: text (vector, temporal, metadata, hybrid)
            time_range: text
            metadata_filter: text
            category: categorytext

        Returns:
            text
        """
        # text，text
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
        """textNmemories"""
        return self.retriever.retrieve_recent(n, category)

    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryToken]:
        """textIDtextmemories"""
        return self.store.load(memory_id)

    def _extract_query_features(self, image: Image.Image) -> np.ndarray:
        """text"""
        try:
            config = TokenExtractionConfig(device="cuda")
            extractor = VisionTokenExtractor(config)

            # text
            if image.mode != 'RGB':
                image = image.convert('RGB')

            tokens = extractor._extract_single(image)
            # textmean poolingtextvector
            return tokens.mean(axis=0)

        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return np.zeros(768)

    # ==================== text ====================

    def update_memory(
        self,
        memory_id: str,
        new_tokens: Optional[np.ndarray] = None,
        new_key_embedding: Optional[np.ndarray] = None,
        new_meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        textnew memory

        Args:
            memory_id: memory ID
            new_tokens: texttokens（text）
            new_key_embedding: textkey embedding（text）
            new_meta: text（text）

        Returns:
            text
        """
        memory = self.store.load(memory_id)
        if memory is None:
            return False

        # texttokens
        if new_tokens is not None:
            memory.tokens = new_tokens
            memory.meta.total_tokens = new_tokens.shape[0]

        # textkey embedding
        if new_key_embedding is not None:
            memory.key_embedding = new_key_embedding
            self.index.remove(memory_id)
            self.index.add(memory_id, new_key_embedding)

        # text
        if new_meta:
            memory.meta.extra.update(new_meta)

        # text
        # textcategory
        index_info = self.store._index.get(memory_id, "")
        category = index_info.split("/")[0] if "/" in index_info else "default"

        self.store.save(memory, category, overwrite=True)
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """text"""
        return self.updater.forget_memory(memory_id)

    def delete_by_filter(
        self,
        time_range: Optional[tuple] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> int:
        """
        text

        Returns:
            text
        """
        # text
        results = self.retriever.retrieve(
            mode="hybrid" if metadata_filter else "temporal",
            time_range=time_range,
            metadata_filter=metadata_filter,
            category=category,
            top_k=10000  # text
        )

        deleted = 0
        for result in results:
            if self.delete_memory(result.memory_id):
                deleted += 1

        return deleted

    # ==================== text ====================

    def save(self):
        """text"""
        self.store._save_index()
        if self.config.index_config and self.config.index_config.index_path:
            self.index.save(self.config.index_config.index_path)

    def compact(self):
        """text"""
        self.updater.compact()

    def backup(self, backup_path: str):
        """text"""
        self.store.backup(backup_path)

    def clear(self, category: Optional[str] = None):
        """text"""
        self.store.clear(category)
        if category is None:
            # text
            self.index._vectors.clear()
            self.index._id_list.clear()

    # ==================== text ====================

    def get_statistics(self) -> Dict[str, Any]:
        """text"""
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
        """textwith memory"""
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

    # ==================== text ====================

    def _generate_id(self) -> str:
        """textID"""
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
    text

    Args:
        base_path: text
        index_type: text (memory, faiss, hnswlib)
        vector_dim: vector dimension
        duplicate_threshold: duplicate detectiontext
        capacity_limit: capacity limit

    Returns:
        MemoryManagertext
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
