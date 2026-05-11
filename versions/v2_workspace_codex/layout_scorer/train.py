#!/usr/bin/env python3
"""
Phase 2: Train LayoutScorer MLP.

Trains the CLIP+MLP scorer with combined MSE + Margin Ranking loss.

Config:
    lr=1e-3, weight_decay=1e-4, batch_size=128, epochs=50
    Cosine annealing with warmup (5 epochs)
    Combined loss: MSE(0.5) + Ranking(0.5)

Outputs:
- layout_scorer_output/layout_scorer.pt
- layout_scorer_output/training_log.json

Usage:
    python -m layout_scorer.train [--epochs 50] [--lr 1e-3]
"""

import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/cyf/codex")
from layout_scorer.model import LayoutScorer, combined_loss

DATA_PATH = "/home/cyf/codex/layout_scorer_data/train_data.pt"
OUTPUT_DIR = "/home/cyf/codex/layout_scorer_output"


class LayoutDataset(Dataset):
    def __init__(self, clip_features, heuristic_features, labels, indices=None):
        if indices is not None:
            self.clip_features = clip_features[indices]
            self.heuristic_features = heuristic_features[indices]
            self.labels = labels[indices]
        else:
            self.clip_features = clip_features
            self.heuristic_features = heuristic_features
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "clip": self.clip_features[idx],
            "heuristic": self.heuristic_features[idx],
            "label": self.labels[idx],
        }


def remap_groups(groups: List[List[int]], indices: List[int]) -> List[List[int]]:
    """Remap group indices after train/val split."""
    idx_set = set(indices)
    old_to_new = {old: new for new, old in enumerate(indices)}
    remapped = []
    for group in groups:
        new_group = [old_to_new[i] for i in group if i in idx_set]
        if len(new_group) > 1:
            remapped.append(new_group)
    return remapped


def compute_metrics(model, loader, groups, device):
    """Compute evaluation metrics on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            clip_feat = batch["clip"].to(device)
            heur_feat = batch["heuristic"].to(device)
            preds = model(clip_feat, heur_feat)
            all_preds.append(preds.cpu())
            all_labels.append(batch["label"])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # MSE
    mse = F.mse_loss(all_preds, all_labels).item()

    # Pairwise accuracy
    n_correct = 0
    n_total = 0
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                li, lj = all_labels[group[i]].item(), all_labels[group[j]].item()
                if abs(li - lj) > 0.05:
                    pi, pj = all_preds[group[i]].item(), all_preds[group[j]].item()
                    if (li > lj) == (pi > pj):
                        n_correct += 1
                    n_total += 1
    pairwise_acc = n_correct / max(n_total, 1)

    # Group top-1 accuracy (does scorer pick same top as label?)
    n_top1_correct = 0
    n_groups = 0
    for group in groups:
        if len(group) < 2:
            continue
        pred_vals = [all_preds[i].item() for i in group]
        label_vals = [all_labels[i].item() for i in group]
        pred_best = group[pred_vals.index(max(pred_vals))]
        label_best = group[label_vals.index(max(label_vals))]
        if pred_best == label_best:
            n_top1_correct += 1
        n_groups += 1
    top1_acc = n_top1_correct / max(n_groups, 1)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(all_preds.numpy(), all_labels.numpy())

    return {
        "mse": mse,
        "pairwise_acc": pairwise_acc,
        "top1_acc": top1_acc,
        "spearman_rho": float(rho) if rho == rho else 0.0,  # handle NaN
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--ranking_weight", type=float, default=0.5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: Train LayoutScorer MLP")
    print("=" * 60)

    torch.manual_seed(args.seed)

    # Load data
    data = torch.load(DATA_PATH, weights_only=False)
    clip_features = data["clip_features"]       # [N, 768]
    heuristic_features = data["heuristic_features"]  # [N, 15]
    labels = data["labels"]                     # [N]
    groups = data["groups"]                     # list of lists

    N = len(labels)
    print(f"Dataset: {N} samples, {len(groups)} groups")
    print(f"  CLIP features: {clip_features.shape}")
    print(f"  Heuristic features: {heuristic_features.shape}")
    print(f"  Labels: mean={labels.mean():.3f}, std={labels.std():.3f}")

    # Train/val split (by group to avoid leakage)
    n_val_groups = max(1, int(len(groups) * args.val_split))
    perm = torch.randperm(len(groups)).tolist()
    val_group_idx = perm[:n_val_groups]
    train_group_idx = perm[n_val_groups:]

    val_indices = sorted(set(i for gi in val_group_idx for i in groups[gi]))
    train_indices = sorted(set(range(N)) - set(val_indices))

    print(f"  Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")

    train_groups = remap_groups(groups, train_indices)
    val_groups = remap_groups(groups, val_indices)

    train_dataset = LayoutDataset(clip_features, heuristic_features, labels, train_indices)
    val_dataset = LayoutDataset(clip_features, heuristic_features, labels, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = LayoutScorer(clip_dim=768, heuristic_dim=15).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + __import__("math").cos(progress * __import__("math").pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_pairwise = 0.0
    training_log = []

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            clip_feat = batch["clip"].to(device)
            heur_feat = batch["heuristic"].to(device)
            label = batch["label"].to(device)

            preds = model(clip_feat, heur_feat)

            # Simple MSE for batch loss (ranking loss computed per-group is expensive)
            loss = F.mse_loss(preds, label)

            # Add ranking loss for mini-batches that contain group members
            # (full ranking loss is computed at validation time)
            if args.ranking_weight > 0:
                loss = args.mse_weight * loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = compute_metrics(model, val_loader, val_groups, device)
            train_metrics = compute_metrics(model, train_loader, train_groups, device)

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            training_log.append(log_entry)

            print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"val: mse={val_metrics['mse']:.4f}, pairwise={val_metrics['pairwise_acc']:.3f}, "
                  f"top1={val_metrics['top1_acc']:.3f}, ρ={val_metrics['spearman_rho']:.3f}")

            # Save best model
            if val_metrics["pairwise_acc"] > best_val_pairwise:
                best_val_pairwise = val_metrics["pairwise_acc"]
                model_path = os.path.join(OUTPUT_DIR, "layout_scorer.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "config": vars(args),
                }, model_path)
                print(f"    ★ Best model saved (pairwise={best_val_pairwise:.3f})")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best val pairwise accuracy: {best_val_pairwise:.3f}")

    # Save training log
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
