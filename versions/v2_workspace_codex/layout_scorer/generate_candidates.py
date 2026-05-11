#!/usr/bin/env python3
"""
Phase 0a: Generate 6 candidate layouts for each of 2000 ScienceQA samples.

Outputs:
- layout_scorer_data/candidates/{pid}_{layout_idx}.png  (12,000 images)
- layout_scorer_data/candidates_meta.json               (metadata for each candidate)

Usage:
    python -m layout_scorer.generate_candidates [--num_samples 2000]
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    ContentBlock,
    Layout,
    BlockType,
    layout_single_column,
    layout_two_column,
    layout_side_by_side,
    render_layout,
    sample_to_blocks,
)

SCIQA_DIR = "/home/cyf/memory/ScienceQA/data/scienceqa"
OUTPUT_DIR = "/home/cyf/codex/layout_scorer_data/candidates"
META_PATH = "/home/cyf/codex/layout_scorer_data/candidates_meta.json"


def load_problems() -> Dict:
    with open(os.path.join(SCIQA_DIR, "problems.json")) as f:
        return json.load(f)


def get_image(pid: str, split: str) -> Optional[Image.Image]:
    img_path = os.path.join(SCIQA_DIR, "images", split, pid, "image.png")
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


def generate_all_candidates(blocks: List[ContentBlock]) -> List[Layout]:
    """Generate 6 candidate layouts (same as choose_best_layout but returns all)."""
    total_area = sum(b.area for b in blocks) * 1.3
    target_side = int(math.sqrt(total_area))
    target_side = max(target_side, 350)
    target_side = min(target_side, 1200)

    candidates = [
        layout_single_column(blocks, target_side),
        layout_two_column(blocks, target_side),
        layout_side_by_side(blocks, target_side),
        layout_single_column(blocks, int(target_side * 1.3)),
        layout_two_column(blocks, int(target_side * 1.2)),
        layout_side_by_side(blocks, int(target_side * 1.2)),
    ]
    return candidates


def stratified_sample(problems: Dict, num_samples: int = 2000) -> List[str]:
    """Stratified sampling by subject from training set."""
    # Group by subject
    subject_pids = defaultdict(list)
    for pid, p in problems.items():
        if p.get("split") == "train":
            subj = p.get("subject", "unknown")
            subject_pids[subj].append(pid)

    total_train = sum(len(v) for v in subject_pids.values())
    print(f"Total train samples: {total_train}")
    print(f"Subjects: {list(subject_pids.keys())}")

    # Proportional sampling per subject
    selected = []
    for subj, pids in sorted(subject_pids.items()):
        n = max(1, int(num_samples * len(pids) / total_train))
        n = min(n, len(pids))
        # Take evenly spaced samples
        step = max(1, len(pids) // n)
        sampled = [pids[i] for i in range(0, len(pids), step)][:n]
        selected.extend(sampled)
        print(f"  {subj}: {len(pids)} total → {len(sampled)} sampled")

    # Trim or pad to exact count
    if len(selected) > num_samples:
        selected = selected[:num_samples]
    print(f"Total selected: {len(selected)}")
    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already generated candidates")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Phase 0a: Generate candidate layouts")
    print(f"Target: {args.num_samples} samples × 6 candidates = {args.num_samples * 6} images")
    print("=" * 60)

    problems = load_problems()
    selected_pids = stratified_sample(problems, args.num_samples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing metadata if resuming
    existing_meta = {}
    if args.resume and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            existing_meta = {m["key"]: m for m in json.load(f)}
        print(f"Resuming: {len(existing_meta)} candidates already exist")

    metadata = list(existing_meta.values())
    n_generated = 0
    n_skipped = 0
    t0 = time.time()

    for i, pid in enumerate(selected_pids):
        p = problems[pid]
        split = p.get("split", "train")

        # Check if all 6 candidates exist
        first_key = f"{pid}_0"
        if args.resume and first_key in existing_meta:
            n_skipped += 1
            continue

        # Load image
        img = get_image(pid, split)

        # Convert to blocks
        try:
            blocks = sample_to_blocks(p, img)
        except Exception as e:
            print(f"  [WARN] pid={pid}: block creation failed: {e}")
            continue

        # Generate 6 candidates
        try:
            candidates = generate_all_candidates(blocks)
        except Exception as e:
            print(f"  [WARN] pid={pid}: candidate generation failed: {e}")
            continue

        # Render and save each candidate
        for li, layout in enumerate(candidates):
            key = f"{pid}_{li}"
            img_path = os.path.join(OUTPUT_DIR, f"{key}.png")

            try:
                canvas = render_layout(layout)
                canvas.save(img_path)
            except Exception as e:
                print(f"  [WARN] {key}: render failed: {e}")
                continue

            meta = {
                "key": key,
                "pid": pid,
                "layout_idx": li,
                "layout_name": layout.name,
                "width": layout.width,
                "height": layout.height,
                "squareness": round(layout.squareness, 4),
                "utilization": round(layout.utilization, 4),
                "n_blocks": len(layout.placements),
                "has_image": img is not None,
                "subject": p.get("subject", ""),
                "topic": p.get("topic", ""),
            }
            metadata.append(meta)
            n_generated += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - n_skipped) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(selected_pids)}] Generated: {n_generated}, "
                  f"Skipped: {n_skipped}, Rate: {rate:.1f} samples/s")

    # Save metadata
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {n_generated} candidate images in {elapsed:.1f}s")
    print(f"Metadata saved to {META_PATH}")
    print(f"Images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
