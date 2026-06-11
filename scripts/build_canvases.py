#!/usr/bin/env python3
"""Build MemCanvas canvases from JSON/JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from memcanvas.canvas import measure_image, measure_table, measure_text, render_canvas


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("train", "data", "examples"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    raise ValueError("Input must be a list or contain train/data/examples list")


def blocks_from_record(record: dict[str, Any], image_root: Path | None = None):
    blocks = []
    header_parts = [str(record.get(key, "")) for key in ("dataset", "subject", "topic", "category") if record.get(key)]
    if header_parts:
        blocks.append(measure_text(" | ".join(header_parts), font_size=12))
    for key in ("context", "hint", "lecture", "question"):
        if record.get(key):
            label = key.capitalize()
            blocks.append(measure_text(f"{label}: {record[key]}", font_size=14 if key != "question" else 16))
    choices = record.get("choices") or record.get("options")
    if choices:
        answer = record.get("answer")
        choice_text = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        blocks.append(measure_text(f"Choices:\n{choice_text}\nAnswer: {answer}", font_size=14))
    for image_key in ("image", "image_path"):
        if record.get(image_key):
            image_path = Path(record[image_key])
            if image_root and not image_path.is_absolute():
                image_path = image_root / image_path
            if image_path.exists():
                blocks.append(measure_image(Image.open(image_path).convert("RGB"), max_dim=420))
    table = record.get("table")
    if table:
        blocks.append(measure_table(table if isinstance(table, list) else table.get("rows", []), font_size=13))
    if not blocks:
        blocks.append(measure_text(json.dumps(record, ensure_ascii=False)[:1200], font_size=14))
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON/JSONL records")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for PNG canvases")
    parser.add_argument("--image-root", type=Path, help="Optional root for relative image paths")
    parser.add_argument("--limit", type=int, help="Optional maximum number of records")
    args = parser.parse_args()

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, record in enumerate(tqdm(records, desc="build canvases")):
        output_path = args.output_dir / f"{idx:05d}.png"
        render_canvas(blocks_from_record(record, args.image_root), output_path)
        manifest.append({"index": idx, "canvas": output_path.name, "text": record.get("question", "")})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
