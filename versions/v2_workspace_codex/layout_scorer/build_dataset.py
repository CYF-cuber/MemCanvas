#!/usr/bin/env python3
"""
Phase 1: Build training dataset by fusing VLM scores + heuristic scores.

Extracts CLIP features and handcrafted features, combines with VLM labels,
and saves the final training dataset.

Without QA signal: label = VLM_score * 0.6 + heuristic_score * 0.4
With QA signal:    label = QA_acc * 0.5 + VLM_score * 0.3 + heuristic_score * 0.2

Outputs:
- layout_scorer_data/features.pt      (CLIP + handcrafted features)
- layout_scorer_data/train_data.pt    (features + labels + groups)

Usage:
    python -m layout_scorer.build_dataset [--qa_scores_path ...]
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from PIL import Image

sys.path.insert(0, "/home/cyf/codex")
from layout_scorer.features import (
    load_clip_model,
    batch_extract_clip,
    extract_heuristic_features,
    heuristic_score,
)
from smart_canvas_layout import Layout, PlacedBlock, ContentBlock, BlockType

CANDIDATES_DIR = "/home/cyf/codex/layout_scorer_data/candidates"
META_PATH = "/home/cyf/codex/layout_scorer_data/candidates_meta.json"
VLM_SCORES_PATH = "/home/cyf/codex/layout_scorer_data/vlm_scores.json"
OUTPUT_DIR = "/home/cyf/codex/layout_scorer_data"


def reconstruct_layout(meta: Dict) -> Layout:
    """Reconstruct a minimal Layout from metadata for heuristic feature extraction."""
    # We need the full Layout with placements for typographic features,
    # but metadata only has summary info. Create a minimal Layout for geometric features.
    layout = Layout(
        name=meta["layout_name"],
        width=meta["width"],
        height=meta["height"],
        placements=[],
    )
    # For typographic features, we create placeholder placements from metadata
    # Since we don't have the full placement info, we'll use what we have
    # and fill in reasonable defaults based on layout dimensions
    return layout


def extract_features_from_image_and_meta(
    meta_list: List[Dict],
    clip_model,
    clip_processor,
    device: str = "cuda",
    clip_batch_size: int = 64,
) -> Dict:
    """Extract all features for candidates."""
    print("Extracting CLIP features...")
    t0 = time.time()

    # Load all images
    images = []
    valid_indices = []
    for i, meta in enumerate(meta_list):
        img_path = os.path.join(CANDIDATES_DIR, f"{meta['key']}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        else:
            print(f"  [WARN] Missing: {img_path}")

    # CLIP features
    clip_features = batch_extract_clip(
        images, clip_model, clip_processor,
        device=device, batch_size=clip_batch_size
    )
    print(f"  CLIP features: {clip_features.shape}, took {time.time()-t0:.1f}s")

    # Heuristic features (from metadata — geometric only since we don't have full placements)
    print("Extracting heuristic features...")
    heuristic_features = []
    for idx in valid_indices:
        meta = meta_list[idx]
        layout = reconstruct_layout(meta)
        # For geometric features, we can compute from metadata
        geo_feats = [
            meta.get("squareness", 0.5),
            meta.get("utilization", 0.5),
            meta["width"] / 1000.0,
            meta["height"] / 1000.0,
            meta["width"] / max(meta["height"], 1),
            (meta["width"] // 28) * (meta["height"] // 28) / 2000.0,
            meta.get("n_blocks", 3) / 10.0,
            1.0 if 400 <= meta["width"] <= 900 and
                  400 <= meta["height"] <= 900 else 0.0,
        ]
        # For typographic features, use reasonable defaults from metadata
        has_img = 1.0 if meta.get("has_image", False) else 0.0
        typo_feats = [
            0.5,      # min_text_w / 600 (default)
            0.8,      # text readability (default)
            0.5,      # text area ratio (default)
            0.3 * has_img,  # image area ratio
            has_img,  # has image
            0.0,      # has table (unknown from meta, default 0)
            meta.get("n_blocks", 3) / 6.0,  # text block count
        ]
        heuristic_features.append(geo_feats + typo_feats)

    heuristic_features = torch.tensor(heuristic_features, dtype=torch.float32)
    print(f"  Heuristic features: {heuristic_features.shape}")

    return {
        "clip_features": clip_features,           # [N, 768]
        "heuristic_features": heuristic_features,  # [N, 15]
        "valid_indices": valid_indices,
    }


def compute_labels(
    meta_list: List[Dict],
    valid_indices: List[int],
    vlm_scores: Dict,
    qa_scores: Optional[Dict] = None,
) -> torch.Tensor:
    """Compute fused training labels."""
    labels = []
    for idx in valid_indices:
        meta = meta_list[idx]
        key = meta["key"]

        # VLM score (normalize 1-5 → 0-1)
        vlm = vlm_scores.get(key, {})
        vlm_total = vlm.get("total", 3.0)
        vlm_norm = (vlm_total - 1.0) / 4.0  # [0, 1]

        # Heuristic score
        h_score = (
            meta.get("squareness", 0.5) * 0.6 +
            meta.get("utilization", 0.5) * 0.3 +
            (1.0 if 300 <= meta["width"] <= 1200 and 300 <= meta["height"] <= 1200 else 0.7) * 0.1
        )

        # QA signal (if available)
        if qa_scores and key in qa_scores:
            qa_acc = qa_scores[key]
            label = qa_acc * 0.50 + vlm_norm * 0.30 + h_score * 0.20
        else:
            label = vlm_norm * 0.60 + h_score * 0.40

        labels.append(label)

    return torch.tensor(labels, dtype=torch.float32)


def build_groups(meta_list: List[Dict], valid_indices: List[int]) -> List[List[int]]:
    """Build groups of candidates from the same sample (for ranking loss)."""
    pid_to_positions = defaultdict(list)
    for pos, idx in enumerate(valid_indices):
        pid = meta_list[idx]["pid"]
        pid_to_positions[pid].append(pos)

    groups = [indices for indices in pid_to_positions.values() if len(indices) > 1]
    return groups


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_scores_path", type=str, default=None,
                        help="Path to QA scores JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_batch_size", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Build Training Dataset")
    print("=" * 60)

    # Load metadata
    with open(META_PATH) as f:
        meta_list = json.load(f)
    print(f"Total candidates: {len(meta_list)}")

    # Load VLM scores
    with open(VLM_SCORES_PATH) as f:
        vlm_scores = json.load(f)
    print(f"VLM scores: {len(vlm_scores)}")

    # Load QA scores if available
    qa_scores = None
    if args.qa_scores_path and os.path.exists(args.qa_scores_path):
        with open(args.qa_scores_path) as f:
            qa_scores = json.load(f)
        print(f"QA scores: {len(qa_scores)}")

    # Load CLIP model
    clip_model, clip_processor = load_clip_model(args.device)

    # Extract features
    features = extract_features_from_image_and_meta(
        meta_list, clip_model, clip_processor,
        device=args.device, clip_batch_size=args.clip_batch_size
    )

    # Compute labels
    labels = compute_labels(
        meta_list, features["valid_indices"],
        vlm_scores, qa_scores
    )
    print(f"Labels: {labels.shape}, mean={labels.mean():.3f}, std={labels.std():.3f}")

    # Build groups
    groups = build_groups(meta_list, features["valid_indices"])
    print(f"Groups: {len(groups)} (avg size: {sum(len(g) for g in groups)/len(groups):.1f})")

    # Save features
    features_path = os.path.join(OUTPUT_DIR, "features.pt")
    torch.save({
        "clip_features": features["clip_features"],
        "heuristic_features": features["heuristic_features"],
        "valid_indices": features["valid_indices"],
    }, features_path)
    print(f"Features saved to {features_path}")

    # Save training data
    train_data_path = os.path.join(OUTPUT_DIR, "train_data.pt")
    torch.save({
        "clip_features": features["clip_features"],
        "heuristic_features": features["heuristic_features"],
        "labels": labels,
        "groups": groups,
        "valid_indices": features["valid_indices"],
        "meta_keys": [meta_list[i]["key"] for i in features["valid_indices"]],
    }, train_data_path)
    print(f"Training data saved to {train_data_path}")

    # Print label distribution per layout type
    print("\nLabel distribution by layout type:")
    layout_labels = defaultdict(list)
    for pos, idx in enumerate(features["valid_indices"]):
        ltype = meta_list[idx]["layout_name"]
        layout_labels[ltype].append(labels[pos].item())

    for ltype, lvals in sorted(layout_labels.items()):
        import numpy as np
        arr = np.array(lvals)
        print(f"  {ltype}: n={len(arr)}, mean={arr.mean():.3f}, std={arr.std():.3f}")


if __name__ == "__main__":
    main()
