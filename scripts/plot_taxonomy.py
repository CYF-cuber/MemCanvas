#!/usr/bin/env python3
"""Plot taxonomy distributions from MemCanvas classification label files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_labels(path: Path) -> list[tuple[str, str]]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.strip()[1:-1].split(",")] if line.startswith("(") else []
        if len(parts) >= 4:
            labels.append((parts[2], parts[3]))
    return labels


def bar(counter: Counter, title: str, output: Path) -> None:
    names, values = zip(*counter.most_common()) if counter else ([], [])
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="MemCanvas taxonomy distribution")
    args = parser.parse_args()
    labels = load_labels(args.labels)
    bar(Counter(major for major, _ in labels), args.title, args.output)


if __name__ == "__main__":
    main()
