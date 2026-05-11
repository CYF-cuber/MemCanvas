"""
MemoryStore - 记忆存储模块

提供记忆的持久化存储，支持：
- 文件系统存储（默认）
- 支持扩展到其他后端（Redis、MongoDB等）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator
from pathlib import Path
from datetime import datetime
import json

from ...encoders.slicer.memory_token import MemoryMeta, MemoryToken


@dataclass
class MemoryStoreConfig:
    """存储配置"""
    # 存储根目录
    storage_path: str = "./memory_store"
    # 是否压缩存储
    compress: bool = True
    # 最大记忆数量（0表示无限制）
    max_memories: int = 0
    # 自动备份间隔（秒，0表示不备份）
    backup_interval: int = 0
    # 记忆分类（用于分目录存储）
    categories: List[str] = field(default_factory=lambda: ["default"])


class MemoryStore:
    """
    记忆存储器

    负责记忆的持久化存储和加载。
    使用文件系统作为默认后端。

    目录结构：
    storage_path/
    ├── index.json          # 记忆索引（ID -> 文件路径映射）
    ├── metadata.json       # 存储元数据
    ├── default/            # 默认分类
    │   ├── mem_001.npz
    │   ├── mem_002.npz
    │   └── ...
    └── category_name/      # 其他分类
        └── ...
    """

    def __init__(self, config: Optional[MemoryStoreConfig] = None):
        self.config = config or MemoryStoreConfig()
        self.storage_path = Path(self.config.storage_path)

        # 记忆索引：memory_id -> file_path
        self._index: Dict[str, str] = {}
        # 元数据缓存：memory_id -> MemoryMeta
        self._meta_cache: Dict[str, MemoryMeta] = {}

        self._init_storage()

    def _init_storage(self):
        """初始化存储目录"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 创建分类目录
        for category in self.config.categories:
            (self.storage_path / category).mkdir(exist_ok=True)

        # 加载索引
        self._load_index()

    def _load_index(self):
        """加载索引文件"""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                self._index = data.get("index", {})

    def _save_index(self):
        """保存索引文件"""
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
        保存记忆

        Args:
            memory: MemoryToken对象
            category: 分类名称
            overwrite: 是否覆盖已存在的记忆

        Returns:
            memory_id
        """
        memory_id = memory.meta.memory_id

        # 检查是否已存在
        if memory_id in self._index and not overwrite:
            raise ValueError(f"Memory {memory_id} already exists. Use overwrite=True to replace.")

        # 确保分类目录存在
        category_path = self.storage_path / category
        category_path.mkdir(exist_ok=True)

        # 生成文件路径
        file_name = f"{memory_id}.npz"
        file_path = category_path / file_name

        # 保存记忆
        memory.save(str(file_path))

        # 更新索引
        self._index[memory_id] = str(file_path.relative_to(self.storage_path))
        self._meta_cache[memory_id] = memory.meta
        self._save_index()

        return memory_id

    def load(self, memory_id: str) -> Optional[MemoryToken]:
        """
        加载记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryToken或None
        """
        if memory_id not in self._index:
            return None

        file_path = self.storage_path / self._index[memory_id]
        if not file_path.exists():
            # 索引失效，清理
            del self._index[memory_id]
            self._save_index()
            return None

        return MemoryToken.load(str(file_path))

    def delete(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否成功删除
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
        """检查记忆是否存在"""
        return memory_id in self._index

    def list_ids(self, category: Optional[str] = None) -> List[str]:
        """
        列出所有记忆ID

        Args:
            category: 可选，按分类过滤

        Returns:
            记忆ID列表
        """
        if category is None:
            return list(self._index.keys())

        return [
            mid for mid, path in self._index.items()
            if path.startswith(category + "/")
        ]

    def get_meta(self, memory_id: str) -> Optional[MemoryMeta]:
        """获取记忆元数据（不加载完整数据）"""
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
        批量迭代记忆

        Args:
            category: 可选分类过滤
            batch_size: 批次大小

        Yields:
            MemoryToken批次
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
        """获取存储统计信息"""
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
        """备份整个存储"""
        shutil.copytree(self.storage_path, backup_path)

    def clear(self, category: Optional[str] = None):
        """
        清空存储

        Args:
            category: 可选，只清空指定分类
        """
        if category:
            ids_to_delete = self.list_ids(category)
            for mid in ids_to_delete:
                self.delete(mid)
        else:
            # 清空所有
            for mid in list(self._index.keys()):
                self.delete(mid)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, memory_id: str) -> bool:
        return self.exists(memory_id)
