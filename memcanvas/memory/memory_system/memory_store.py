"""
MemoryStore - memory storage module

textpersistent storage，text：
- filesystem storage（text）
- supports extension to other backends（Redis、MongoDBtext）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator
from pathlib import Path
from datetime import datetime
import json

from ...encoders.slicer.memory_token import MemoryMeta, MemoryToken


@dataclass
class MemoryStoreConfig:
    """storage configuration"""
    # storage root directory
    storage_path: str = "./memory_store"
    # whether to compress stored data
    compress: bool = True
    # maximum number of memories（0means unlimited）
    max_memories: int = 0
    # automatic backup interval（seconds，0means no backup）
    backup_interval: int = 0
    # textcategory（used for directory-based storage）
    categories: List[str] = field(default_factory=lambda: ["default"])


class MemoryStore:
    """
    memory store

    Responsibilitiestextpersistent storagetext。
    uses the filesystem as the default backend。

    directory structure：
    storage_path/
    ├── index.json          # memory index（ID -> text）
    ├── metadata.json       # text
    ├── default/            # textcategory
    │   ├── mem_001.npz
    │   ├── mem_002.npz
    │   └── ...
    └── category_name/      # textcategory
        └── ...
    """

    def __init__(self, config: Optional[MemoryStoreConfig] = None):
        self.config = config or MemoryStoreConfig()
        self.storage_path = Path(self.config.storage_path)

        # memory index：memory_id -> file_path
        self._index: Dict[str, str] = {}
        # metadata cache：memory_id -> MemoryMeta
        self._meta_cache: Dict[str, MemoryMeta] = {}

        self._init_storage()

    def _init_storage(self):
        """initialize storage directory"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # textcategorytext
        for category in self.config.categories:
            (self.storage_path / category).mkdir(exist_ok=True)

        # load index
        self._load_index()

    def _load_index(self):
        """load indextext"""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                self._index = data.get("index", {})

    def _save_index(self):
        """save index file"""
        index_path = self.storage_path / "index.json"
        with open(index_path, 'w') as f:
            json.dump({
                "index": self._index,
                "updated_at": datetime.now().isoformat(),
                "count": len(self._index)
            }, f, indent=2)

    def save(
        self,
        memory: MemoryToken,
        category: str = "default",
        overwrite: bool = False
    ) -> str:
        """
        save memory

        Args:
            memory: MemoryTokenobject
            category: categorytext
            overwrite: whether to overwrite an existing memory

        Returns:
            memory_id
        """
        memory_id = memory.meta.memory_id

        # check whether it already exists
        if memory_id in self._index and not overwrite:
            raise ValueError(f"Memory {memory_id} already exists. Use overwrite=True to replace.")

        # textcategorytext
        category_path = self.storage_path / category
        category_path.mkdir(exist_ok=True)

        # generate file path
        file_name = f"{memory_id}.npz"
        file_path = category_path / file_name

        # save memory
        memory.save(str(file_path))

        # text
        self._index[memory_id] = str(file_path.relative_to(self.storage_path))
        self._meta_cache[memory_id] = memory.meta
        self._save_index()

        return memory_id

    def load(self, memory_id: str) -> Optional[MemoryToken]:
        """
        text

        Args:
            memory_id: memory ID

        Returns:
            MemoryTokentextNone
        """
        if memory_id not in self._index:
            return None

        file_path = self.storage_path / self._index[memory_id]
        if not file_path.exists():
            # text，text
            del self._index[memory_id]
            self._save_index()
            return None

        return MemoryToken.load(str(file_path))

    def delete(self, memory_id: str) -> bool:
        """
        text

        Args:
            memory_id: memory ID

        Returns:
            text
        """
        if memory_id not in self._index:
            return False

        file_path = self.storage_path / self._index[memory_id]
        if file_path.exists():
            file_path.unlink()

        del self._index[memory_id]
        if memory_id in self._meta_cache:
            del self._meta_cache[memory_id]
        self._save_index()

        return True

    def exists(self, memory_id: str) -> bool:
        """text"""
        return memory_id in self._index

    def list_ids(self, category: Optional[str] = None) -> List[str]:
        """
        textmemory ID

        Args:
            category: text，textcategorytext

        Returns:
            memory IDtext
        """
        if category is None:
            return list(self._index.keys())

        return [
            mid for mid, path in self._index.items()
            if path.startswith(category + "/")
        ]

    def get_meta(self, memory_id: str) -> Optional[MemoryMeta]:
        """text（text）"""
        if memory_id in self._meta_cache:
            return self._meta_cache[memory_id]

        memory = self.load(memory_id)
        if memory:
            self._meta_cache[memory_id] = memory.meta
            return memory.meta
        return None

    def iter_memories(
        self,
        category: Optional[str] = None,
        batch_size: int = 10
    ) -> Iterator[List[MemoryToken]]:
        """
        text

        Args:
            category: textcategorytext
            batch_size: text

        Yields:
            MemoryTokentext
        """
        ids = self.list_ids(category)

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch = []
            for mid in batch_ids:
                memory = self.load(mid)
                if memory:
                    batch.append(memory)
            if batch:
                yield batch

    def get_statistics(self) -> Dict[str, Any]:
        """text"""
        total_size = 0
        category_counts = {}

        for mid, path in self._index.items():
            file_path = self.storage_path / path
            if file_path.exists():
                total_size += file_path.stat().st_size

            category = path.split("/")[0]
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_memories": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "categories": category_counts,
            "storage_path": str(self.storage_path)
        }

    def backup(self, backup_path: str):
        """text"""
        shutil.copytree(self.storage_path, backup_path)

    def clear(self, category: Optional[str] = None):
        """
        text

        Args:
            category: text，textcategory
        """
        if category:
            ids_to_delete = self.list_ids(category)
            for mid in ids_to_delete:
                self.delete(mid)
        else:
            # text
            for mid in list(self._index.keys()):
                self.delete(mid)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, memory_id: str) -> bool:
        return self.exists(memory_id)
