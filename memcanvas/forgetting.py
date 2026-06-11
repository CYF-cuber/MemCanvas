"""Progressive visual forgetting for MemCanvas memory banks."""

from __future__ import annotations

import io
from enum import IntEnum
from pathlib import Path

from PIL import Image

from .bank import MemoryBank, MemoryEntry


class QualityLevel(IntEnum):
    ORIGINAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DELETED = 4


QUALITY_SCALES = {
    QualityLevel.ORIGINAL: 1.0,
    QualityLevel.HIGH: 0.75,
    QualityLevel.MEDIUM: 0.5,
    QualityLevel.LOW: 0.25,
    QualityLevel.DELETED: 0.0,
}


def resize_png_bytes(image_bytes: bytes, scale: float) -> bytes:
    if scale <= 0:
        return b""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if scale < 1:
        width, height = image.size
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def degrade_entry(bank: MemoryBank, entry: MemoryEntry) -> None:
    if entry.deleted:
        return
    next_level = min(int(entry.quality_level) + 1, int(QualityLevel.DELETED))
    entry.quality_level = next_level
    if next_level == int(QualityLevel.DELETED):
        entry.deleted = True
        return
    canvas_path = bank.canvas_file(entry)
    image_bytes = canvas_path.read_bytes()
    scale = QUALITY_SCALES[QualityLevel(next_level)]
    canvas_path.write_bytes(resize_png_bytes(image_bytes, scale))


def review_memory_bank(bank: MemoryBank, threshold: int = 0) -> list[str]:
    degraded: list[str] = []
    for entry in bank.active_entries():
        if entry.access_count <= threshold:
            degrade_entry(bank, entry)
            degraded.append(entry.id)
    return degraded


class ProgressiveForgettingPolicy:
    def __init__(self, review_interval: int = 1000, threshold: int = 0):
        self.review_interval = review_interval
        self.threshold = threshold
        self.seen_queries = 0

    def step(self, bank: MemoryBank, n_queries: int = 1) -> list[str]:
        self.seen_queries += n_queries
        if self.seen_queries < self.review_interval:
            return []
        self.seen_queries = 0
        degraded = review_memory_bank(bank, threshold=self.threshold)
        bank.save()
        return degraded


def estimate_storage_bytes(bank: MemoryBank) -> int:
    total = 0
    for entry in bank.active_entries():
        path = bank.canvas_file(entry)
        if path.exists():
            total += path.stat().st_size
    return total
