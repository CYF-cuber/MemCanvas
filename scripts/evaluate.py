#!/usr/bin/env python3
"""Evaluate hybrid retrieval maps without running a full VLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memcanvas.retrieval import hybrid_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-embeddings", required=True, type=Path)
    parser.add_argument("--text-embeddings", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    retrieval = hybrid_retrieval(
        np.load(args.image_embeddings),
        np.load(args.text_embeddings),
        np.load(args.query_embeddings),
        alpha=args.alpha,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
