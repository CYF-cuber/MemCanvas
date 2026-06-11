#!/usr/bin/env python3
"""Apply MemCanvas topic and modality taxonomies to records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_label_file(path: Path) -> dict[tuple[str, str, str], tuple[str, ...]]:
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not (line.startswith("(") and line.endswith(")")):
            continue
        parts = [part.strip() for part in line[1:-1].split(",")]
        if len(parts) >= 4 and ":" in parts[1]:
            dataset, split_index = parts[0], parts[1]
            split, index = split_index.split(":", 1)
            labels[(dataset, split, index)] = tuple(parts[2:])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="JSONL with dataset/split/index fields")
    parser.add_argument("--topic-labels", required=True, type=Path)
    parser.add_argument("--modality-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    topic_labels = parse_label_file(args.topic_labels)
    modality_labels = parse_label_file(args.modality_labels)
    out_lines = []
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["dataset"]), str(record["split"]), str(record["index"]))
        if key in topic_labels:
            record["major_topic"], record["subtopic"] = topic_labels[key]
        if key in modality_labels:
            record["modality"], record["hop_type"] = modality_labels[key]
        out_lines.append(json.dumps(record, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
