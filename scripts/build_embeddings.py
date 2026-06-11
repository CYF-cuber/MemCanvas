#!/usr/bin/env python3
"""Build CLIP embeddings for MemCanvas canvases and text fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memcanvas.retrieval import embed_images, embed_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Manifest JSON with text fields")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--text-key", default="text")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.canvas_dir.glob("*.png"))
    if images:
        np.save(args.output_dir / "clip_img_emb.npy", embed_images(images, model_name=args.model))
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        texts = [str(item.get(args.text_key, "")) for item in manifest]
        np.save(args.output_dir / "clip_txt_emb.npy", embed_texts(texts, model_name=args.model))


if __name__ == "__main__":
    main()
