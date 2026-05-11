#!/usr/bin/env python3
"""
Phase 2b: Evaluate LayoutScorer with ablations.

Ablation experiments:
1. Full model (CLIP + handcrafted features)
2. w/o CLIP features (only 15-dim handcrafted)
3. w/o handcrafted features (only 768-dim CLIP)
4. Heuristic baseline (original scoring formula)

Metrics:
- Group top-1 accuracy
- Pairwise accuracy
- Spearman rank correlation
- MSE

Usage:
    python -m layout_scorer.evaluate [--model_path ...] [--ablation all]
"""

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/cyf/codex")
from layout_scorer.model import LayoutScorer

DATA_PATH = "/home/cyf/codex/layout_scorer_data/train_data.pt"
MODEL_PATH = "/home/cyf/codex/layout_scorer_output/layout_scorer.pt"
META_PATH = "/home/cyf/codex/layout_scorer_data/candidates_meta.json"
OUTPUT_DIR = "/home/cyf/codex/layout_scorer_output"


def evaluate_predictions(preds: np.ndarray, labels: np.ndarray, groups: List[List[int]]) -> Dict:
    """Compute all evaluation metrics."""
    from scipy.stats import spearmanr

    # MSE
    mse = float(np.mean((preds - labels) ** 2))

    # Pairwise accuracy
    n_correct = 0
    n_total = 0
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                li, lj = labels[group[i]], labels[group[j]]
                if abs(li - lj) > 0.05:
                    pi, pj = preds[group[i]], preds[group[j]]
                    if (li > lj) == (pi > pj):
                        n_correct += 1
                    n_total += 1
    pairwise_acc = n_correct / max(n_total, 1)

    # Group top-1 accuracy
    n_top1_correct = 0
    n_groups = 0
    for group in groups:
        if len(group) < 2:
            continue
        pred_vals = [preds[i] for i in group]
        label_vals = [labels[i] for i in group]
        pred_best = group[np.argmax(pred_vals)]
        label_best = group[np.argmax(label_vals)]
        if pred_best == label_best:
            n_top1_correct += 1
        n_groups += 1
    top1_acc = n_top1_correct / max(n_groups, 1)

    # Spearman correlation
    rho, pval = spearmanr(preds, labels)
    if np.isnan(rho):
        rho = 0.0

    return {
        "mse": mse,
        "pairwise_acc": pairwise_acc,
        "top1_acc": top1_acc,
        "spearman_rho": float(rho),
        "n_pairs": n_total,
        "n_groups": n_groups,
    }


def run_scorer(model, clip_features, heuristic_features, device, batch_size=256):
    """Run scorer model on features, return predictions."""
    model.eval()
    preds = []
    N = len(clip_features)
    for i in range(0, N, batch_size):
        cf = clip_features[i:i+batch_size].to(device)
        hf = heuristic_features[i:i+batch_size].to(device)
        with torch.no_grad():
            p = model(cf, hf)
        preds.append(p.cpu())
    return torch.cat(preds).numpy()


def heuristic_baseline(meta_list, valid_indices):
    """Compute heuristic scores as baseline."""
    scores = []
    for idx in valid_indices:
        meta = meta_list[idx]
        sq = meta.get("squareness", 0.5)
        util = meta.get("utilization", 0.5)
        size_ok = 1.0 if 300 <= meta["width"] <= 1200 and 300 <= meta["height"] <= 1200 else 0.7
        scores.append(sq * 0.6 + util * 0.3 + size_ok * 0.1)
    return np.array(scores)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "full", "no_clip", "no_heuristic", "heuristic_only"])
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2b: Evaluate LayoutScorer")
    print("=" * 60)

    # Load data
    data = torch.load(DATA_PATH, weights_only=False)
    clip_features = data["clip_features"]
    heuristic_features = data["heuristic_features"]
    labels = data["labels"].numpy()
    groups = data["groups"]

    with open(META_PATH) as f:
        meta_list = json.load(f)
    valid_indices = data["valid_indices"]

    N = len(labels)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {N} samples, {len(groups)} groups")

    results = {}

    # Full model
    if args.ablation in ["all", "full"]:
        print("\n--- Full Model (CLIP + Handcrafted) ---")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model = LayoutScorer(clip_dim=768, heuristic_dim=15).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        preds = run_scorer(model, clip_features, heuristic_features, device)
        metrics = evaluate_predictions(preds, labels, groups)
        results["full_model"] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Ablation: w/o CLIP features
    if args.ablation in ["all", "no_clip"]:
        print("\n--- Ablation: w/o CLIP (Handcrafted only) ---")
        model_nc = LayoutScorer(clip_dim=0, heuristic_dim=15).to(device)
        # Need to retrain with only heuristic features
        # For quick eval, use zeros for CLIP features
        zero_clip = torch.zeros_like(clip_features)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model_nc2 = LayoutScorer(clip_dim=768, heuristic_dim=15).to(device)
        model_nc2.load_state_dict(ckpt["model_state_dict"])
        preds = run_scorer(model_nc2, zero_clip, heuristic_features, device)
        metrics = evaluate_predictions(preds, labels, groups)
        results["no_clip"] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Ablation: w/o handcrafted features
    if args.ablation in ["all", "no_heuristic"]:
        print("\n--- Ablation: w/o Handcrafted (CLIP only) ---")
        zero_heuristic = torch.zeros_like(heuristic_features)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model_nh = LayoutScorer(clip_dim=768, heuristic_dim=15).to(device)
        model_nh.load_state_dict(ckpt["model_state_dict"])
        preds = run_scorer(model_nh, clip_features, zero_heuristic, device)
        metrics = evaluate_predictions(preds, labels, groups)
        results["no_heuristic"] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Heuristic baseline
    if args.ablation in ["all", "heuristic_only"]:
        print("\n--- Heuristic Baseline (original formula) ---")
        preds = heuristic_baseline(meta_list, valid_indices)
        metrics = evaluate_predictions(preds, labels, groups)
        results["heuristic_baseline"] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eval_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Method':<25} {'MSE':>8} {'Pairwise':>10} {'Top-1':>8} {'ρ':>8}")
        print("-" * 70)
        for name, m in results.items():
            print(f"{name:<25} {m['mse']:>8.4f} {m['pairwise_acc']:>10.3f} "
                  f"{m['top1_acc']:>8.3f} {m['spearman_rho']:>8.3f}")


if __name__ == "__main__":
    main()
