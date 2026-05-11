#!/usr/bin/env python3
"""
Canvas Executor — executes agent decisions on the canvas library.

This module bridges the gap between the Agent's abstract decisions
(CREATE, APPEND, MERGE, etc.) and the actual DynamicCanvas operations.
It also manages the canvas library state (metadata, storage, indexing).
"""

import hashlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, "/home/cyf/memory")
from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig

from canvas_agent import CanvasMeta, WriteAction, ReadAction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExecutorConfig:
    patch_size: int = 640
    font_size: int = 18
    max_patches_per_canvas: int = 4
    storage_dir: Optional[str] = None  # if set, persist canvases to disk


# ---------------------------------------------------------------------------
# Canvas Entry — internal representation
# ---------------------------------------------------------------------------
@dataclass
class CanvasEntry:
    """A canvas in the library with its metadata and content."""
    meta: CanvasMeta
    canvas: DynamicCanvas  # the actual canvas object
    items: List[str] = field(default_factory=list)  # stored knowledge items
    image_paths: List[str] = field(default_factory=list)  # associated image paths


# ---------------------------------------------------------------------------
# Canvas Executor
# ---------------------------------------------------------------------------
class CanvasExecutor:
    """Executes canvas management operations and maintains the library."""

    def __init__(self, config: ExecutorConfig = None):
        self.config = config or ExecutorConfig()
        self.library: Dict[int, CanvasEntry] = {}
        self._next_id: int = 0
        self._canvas_config = DynamicCanvasConfig(
            patch_size=self.config.patch_size,
            font_size=self.config.font_size,
            padding=25,
            content_gap=12,
            show_patch_boundary=False,
        )

    @property
    def canvas_metas(self) -> List[CanvasMeta]:
        """Get metadata for all canvases (what the agent sees)."""
        return [entry.meta for entry in self.library.values()]

    def execute_write(self, action: WriteAction, knowledge: str,
                      image: Optional[Image.Image] = None) -> int:
        """Execute a write action. Returns the canvas_id affected."""
        if action.action == "CREATE":
            return self._create(action.params.get("topic", "Untitled"), knowledge, image)
        elif action.action == "APPEND":
            cid = action.params.get("canvas_id", 0)
            if cid in self.library:
                return self._append(cid, knowledge, image)
            else:
                # Canvas doesn't exist, fallback to create
                return self._create(action.params.get("topic", "General"), knowledge, image)
        elif action.action == "NOOP":
            return -1
        else:
            return -1

    def _create(self, topic: str, knowledge: str,
                image: Optional[Image.Image] = None) -> int:
        """Create a new canvas with initial content."""
        canvas_id = self._next_id
        self._next_id += 1

        canvas = DynamicCanvas(self._canvas_config)
        canvas.add_text(f"[{topic}]", font_size=22, bold=True)
        canvas.add_separator()

        if image is not None:
            canvas.add_image(image, max_height=350)

        canvas.add_text(knowledge, font_size=self.config.font_size)

        meta = CanvasMeta(
            canvas_id=canvas_id,
            topic=topic,
            num_items=1,
            num_patches=len(canvas.get_patches()),
            max_patches=self.config.max_patches_per_canvas,
            created_at=time.time(),
            last_accessed=time.time(),
            text_summary=knowledge[:200],
        )

        self.library[canvas_id] = CanvasEntry(
            meta=meta,
            canvas=canvas,
            items=[knowledge],
        )
        return canvas_id

    def _append(self, canvas_id: int, knowledge: str,
                image: Optional[Image.Image] = None) -> int:
        """Append knowledge to an existing canvas."""
        entry = self.library[canvas_id]

        # Check if canvas is full
        if entry.meta.is_full:
            # Create a new canvas with same topic
            return self._create(entry.meta.topic + " (cont.)", knowledge, image)

        canvas = entry.canvas
        canvas.add_separator(style="light")

        if image is not None:
            canvas.add_image(image, max_height=300)

        canvas.add_text(knowledge, font_size=self.config.font_size)

        # Update metadata
        entry.meta.num_items += 1
        entry.meta.num_patches = len(canvas.get_patches())
        entry.meta.last_accessed = time.time()
        entry.items.append(knowledge)

        # Update summary
        all_items = " | ".join(entry.items[-5:])  # last 5 items
        entry.meta.text_summary = all_items[:200]

        return canvas_id

    def get_canvas_image(self, canvas_id: int) -> Optional[Image.Image]:
        """Get the rendered canvas image."""
        if canvas_id not in self.library:
            return None

        entry = self.library[canvas_id]
        entry.meta.access_count += 1
        entry.meta.last_accessed = time.time()

        patches = entry.canvas.get_images()
        if len(patches) == 1:
            return patches[0]

        # Vertically stack patches
        total_h = sum(p.height for p in patches)
        max_w = max(p.width for p in patches)
        combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for p in patches:
            combined.paste(p, (0, y))
            y += p.height
        return combined

    def get_canvas_images(self, canvas_ids: List[int]) -> List[Image.Image]:
        """Get rendered images for multiple canvases."""
        images = []
        for cid in canvas_ids:
            img = self.get_canvas_image(cid)
            if img is not None:
                images.append(img)
        return images

    def get_canvas_bytes(self, canvas_id: int, format: str = "PNG") -> Optional[bytes]:
        """Get canvas image as bytes."""
        img = self.get_canvas_image(canvas_id)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()

    def get_all_canvas_bytes(self) -> List[Tuple[int, bytes]]:
        """Get all canvas images as (canvas_id, bytes) pairs."""
        result = []
        for cid in self.library:
            b = self.get_canvas_bytes(cid)
            if b is not None:
                result.append((cid, b))
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        total_items = sum(e.meta.num_items for e in self.library.values())
        total_patches = sum(e.meta.num_patches for e in self.library.values())
        return {
            "num_canvases": len(self.library),
            "total_items": total_items,
            "total_patches": total_patches,
            "topics": [e.meta.topic for e in self.library.values()],
        }

    def reset(self):
        """Clear all canvases."""
        self.library.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Canvas Executor — Quick Test")
    print("=" * 60)

    executor = CanvasExecutor()

    # Test CREATE
    action = WriteAction(action="CREATE", params={"topic": "Solar System"})
    cid = executor.execute_write(action, "The Sun is a G-type main-sequence star at the center of our solar system.")
    print(f"Created canvas #{cid}: {executor.library[cid].meta.topic}")

    # Test APPEND
    action = WriteAction(action="APPEND", params={"canvas_id": 0})
    executor.execute_write(action, "Earth is the third planet from the Sun and the only known planet to harbor life.")
    print(f"Appended to canvas #0, now {executor.library[0].meta.num_items} items")

    # Test APPEND more
    executor.execute_write(
        WriteAction(action="APPEND", params={"canvas_id": 0}),
        "Mars has two moons: Phobos and Deimos."
    )

    # Test CREATE another
    executor.execute_write(
        WriteAction(action="CREATE", params={"topic": "French Cuisine"}),
        "Croissants originated from the Austrian kipferl pastry."
    )

    # Get statistics
    stats = executor.get_statistics()
    print(f"\nLibrary stats: {stats}")

    # Get canvas image
    img = executor.get_canvas_image(0)
    print(f"\nCanvas #0 image size: {img.size}")

    # List all metas
    for m in executor.canvas_metas:
        print(f"  {m.to_display()}")

    print("\nAll tests passed!")
