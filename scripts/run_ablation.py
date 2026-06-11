#!/usr/bin/env python3
"""Run alpha/top-k retrieval ablations for MemCanvas."""

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
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    image_embeddings = np.load(args.image_embeddings)
    text_embeddings = np.load(args.text_embeddings)
    query_embeddings = np.load(args.query_embeddings)
    results = []
    for alpha in args.alphas:
        for top_k in args.top_ks:
            retrieval = hybrid_retrieval(image_embeddings, text_embeddings, query_embeddings, alpha, top_k, args.threshold)
            hit_rate = sum(bool(v) for v in retrieval.values()) / max(1, len(retrieval))
            avg_score = np.mean([score for values in retrieval.values() for _, score in values]) if retrieval else 0.0
            results.append({"alpha": alpha, "top_k": top_k, "hit_rate": hit_rate, "avg_score": float(avg_score)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
