"""Memory update and forgetting utilities."""

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
    """Action selected for an incoming memory."""
    ADD = "add"
    UPDATE = "update"
    REMAIN = "remain"
    MERGE = "merge"


@dataclass
class UpdateConfig:
    """Configuration for memory updates and forgetting."""
    duplicate_threshold: float = 0.85
    merge_threshold: float = 0.7
    auto_duplicate_check: bool = True
    forget_strategy: str = "none"
    capacity_limit: int = 0
    decay_days: int = 365
    importance_threshold: float = 0.3


@dataclass
class UpdateResult:
    """Result returned after processing a memory update."""
    action: UpdateAction
    memory_id: str
    original_id: Optional[str] = None
    similarity: float = 0.0
    message: str = ""


class MemoryUpdater:
    """Apply duplicate detection, merge decisions, and forgetting policies."""

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
        """Process a new memory and decide whether to add, keep, update, or merge it."""
        memory_id = new_memory.meta.memory_id

        if not self.config.auto_duplicate_check or len(self.index) == 0:
            self.store.save(new_memory, category)
            self.index.add(memory_id, new_memory.key_embedding)
            return UpdateResult(
                action=UpdateAction.ADD,
                memory_id=memory_id,
                message="Added new memory"
            )

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
            return self._update_existing(new_memory, similar_id, category)

        elif action == UpdateAction.MERGE:
            return self._merge_memories(new_memory, similar_id, category)

        else:
            self.store.save(new_memory, category)
            self.index.add(memory_id, new_memory.key_embedding)
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
        """Return the update action implied by nearest-neighbor similarity."""
        results = self.index.search(query_embedding, top_k=1, threshold=0.0)

        if not results:
            return UpdateAction.ADD, None, 0.0

        most_similar = results[0]
        similarity = most_similar.score

        if similarity >= self.config.duplicate_threshold:
            return UpdateAction.REMAIN, most_similar.memory_id, similarity

        elif similarity >= self.config.merge_threshold:
            return UpdateAction.MERGE, most_similar.memory_id, similarity

        else:
            return UpdateAction.ADD, most_similar.memory_id, similarity

    def _update_existing(
        self,
        new_memory: MemoryToken,
        existing_id: str,
        category: str
    ) -> UpdateResult:
        """Replace an existing memory with a new memory."""
        self.store.delete(existing_id)
        self.index.remove(existing_id)

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
        """Merge a new memory into a similar existing memory."""
        existing_memory = self.store.load(existing_id)
        if existing_memory is None:
            return self.process_new_memory(new_memory, category)

        merged_tokens = np.concatenate([
            existing_memory.tokens,
            new_memory.tokens
        ], axis=0)

        merged_embedding = (
            existing_memory.key_embedding + new_memory.key_embedding
        ) / 2

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

        self.store.delete(existing_id)
        self.index.remove(existing_id)

        self.store.save(merged_memory, category)
        self.index.add(merged_meta.memory_id, merged_embedding)

        return UpdateResult(
            action=UpdateAction.MERGE,
            memory_id=merged_meta.memory_id,
            original_id=existing_id,
            message=f"Merged {existing_id} + {new_memory.meta.memory_id}"
        )

    def _check_capacity(self):
        """Apply the configured capacity policy if the memory store is over budget."""
        if self.config.capacity_limit <= 0:
            return

        current_count = len(self.store)
        if current_count <= self.config.capacity_limit:
            return

        num_to_forget = current_count - self.config.capacity_limit

        if self.config.forget_strategy == "time_decay":
            self._forget_by_time(num_to_forget)
        elif self.config.forget_strategy == "importance":
            self._forget_by_importance(num_to_forget)
        else:
            self._forget_oldest(num_to_forget)

    def _forget_oldest(self, n: int):
        """Delete the oldest memories."""
        memories_with_time = []
        for mid in self.store.list_ids():
            meta = self.store.get_meta(mid)
            if meta:
                memories_with_time.append((mid, meta.created_at))

        memories_with_time.sort(key=lambda x: x[1])

        for mid, _ in memories_with_time[:n]:
            self.store.delete(mid)
            self.index.remove(mid)

    def _forget_by_time(self, n: int):
        """Delete memories older than the configured decay window."""
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

        if deleted < n:
            self._forget_oldest(n - deleted)

    def _forget_by_importance(self, n: int):
        """Delete memories with the lowest embedding-norm importance proxy."""
        memories_with_importance = []
        for mid in self.store.list_ids():
            vec = self.index.get_vector(mid)
            if vec is not None:
                importance = float(np.linalg.norm(vec))
                memories_with_importance.append((mid, importance))

        memories_with_importance.sort(key=lambda x: x[1])

        for mid, _ in memories_with_importance[:n]:
            self.store.delete(mid)
            self.index.remove(mid)

    def forget_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id not in self.store:
            return False

        self.store.delete(memory_id)
        self.index.remove(memory_id)
        return True

    def compact(self):
        """Synchronize the storage and index after deletes or file cleanup."""
        store_ids = set(self.store.list_ids())
        index_ids = set(self.index._id_list)

        for mid in index_ids - store_ids:
            self.index.remove(mid)

        for mid in store_ids - index_ids:
            memory = self.store.load(mid)
            if memory:
                self.index.add(mid, memory.key_embedding)

    def get_statistics(self) -> Dict[str, Any]:
        """Return memory update statistics."""
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
