"""
LayoutScorer: CLIP visual features + handcrafted features → layout quality score.

Architecture:
    CLIP ViT-L/14 [768-dim] + geometric [8-dim] + typographic [7-dim] = [783-dim]
    → Linear(783, 256) → ReLU → Dropout(0.1)
    → Linear(256, 64)  → ReLU → Dropout(0.1)
    → Linear(64, 1)    → Sigmoid
    → score ∈ [0, 1]

Total params: ~217K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutScorer(nn.Module):
    """CLIP visual features + handcrafted features → layout quality score."""

    def __init__(self, clip_dim: int = 768, heuristic_dim: int = 15):
        super().__init__()
        input_dim = clip_dim + heuristic_dim  # 783

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, clip_features: torch.Tensor, heuristic_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_features: [B, 768] CLIP image embeddings (L2-normalized)
            heuristic_features: [B, 15] geometric + typographic features
        Returns:
            scores: [B] quality scores in [0, 1]
        """
        x = torch.cat([clip_features, heuristic_features], dim=-1)
        return self.mlp(x).squeeze(-1)


def combined_loss(
    pred_scores: torch.Tensor,
    labels: torch.Tensor,
    group_indices: list,
    mse_weight: float = 0.5,
    ranking_weight: float = 0.5,
    ranking_margin: float = 0.05,
) -> torch.Tensor:
    """
    Combined MSE regression + pairwise margin ranking loss.

    Args:
        pred_scores: [B] predicted scores
        labels: [B] target labels
        group_indices: list of lists, each inner list contains indices of
                       candidates from the same sample
        mse_weight: weight for MSE loss
        ranking_weight: weight for ranking loss
        ranking_margin: margin for margin ranking loss
    """
    # MSE regression
    mse = F.mse_loss(pred_scores, labels)

    # Pairwise margin ranking within each group
    ranking_loss = torch.tensor(0.0, device=pred_scores.device)
    n_pairs = 0

    for group in group_indices:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                idx_i, idx_j = group[i], group[j]
                li, lj = labels[idx_i].item(), labels[idx_j].item()

                if abs(li - lj) > 0.05:  # only meaningful differences
                    si = pred_scores[idx_i].unsqueeze(0)
                    sj = pred_scores[idx_j].unsqueeze(0)
                    target = torch.tensor([1.0 if li > lj else -1.0],
                                          device=pred_scores.device)
                    ranking_loss = ranking_loss + F.margin_ranking_loss(
                        si, sj, target, margin=ranking_margin
                    )
                    n_pairs += 1

    if n_pairs > 0:
        ranking_loss = ranking_loss / n_pairs

    return mse_weight * mse + ranking_weight * ranking_loss
