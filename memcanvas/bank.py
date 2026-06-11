"""Persistent memory bank for rendered MemCanvas entries."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class MemoryEntry:
    id: str
    canvas_path: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    image_embedding_path: str | None = None
    text_embedding_path: str | None = None
    access_count: int = 0
    quality_level: int = 0
    deleted: bool = False

    def mark_accessed(self) -> None:
        self.access_count += 1


class MemoryBank:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.canvas_dir = self.root / "canvases"
        self.manifest_path = self.root / "manifest.jsonl"
        self.canvas_dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, MemoryEntry] = {}
        if self.manifest_path.exists():
            self.load()

    def add(
        self,
        image: Image.Image,
        text: str = "",
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> MemoryEntry:
        eid = entry_id or str(uuid.uuid4())
        canvas_path = self.canvas_dir / f"{eid}.png"
        image.save(canvas_path)
        entry = MemoryEntry(
            id=eid,
            canvas_path=str(canvas_path.relative_to(self.root)),
            text=text,
            metadata=metadata or {},
        )
        self.entries[eid] = entry
        return entry

    def get(self, entry_id: str) -> MemoryEntry:
        return self.entries[entry_id]

    def active_entries(self) -> list[MemoryEntry]:
        return [entry for entry in self.entries.values() if not entry.deleted]

    def canvas_file(self, entry: MemoryEntry) -> Path:
        return self.root / entry.canvas_path

    def mark_accessed(self, entry_ids: list[str]) -> None:
        for entry_id in entry_ids:
            if entry_id in self.entries:
                self.entries[entry_id].mark_accessed()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for entry in self.entries.values():
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def load(self) -> None:
        self.entries.clear()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = MemoryEntry(**json.loads(line))
                    self.entries[entry.id] = entry

    def export_json(self, output_path: str | Path) -> None:
        payload = [asdict(entry) for entry in self.entries.values()]
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
