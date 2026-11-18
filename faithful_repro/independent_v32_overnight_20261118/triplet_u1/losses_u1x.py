# -*- coding: utf-8 -*-
"""Existence focal loss — port of upstream
SimplifiedMultiTypeAssociationLoss._compute_existence_loss
(upstream_main_experiments_hetero1.py L115-133).

  pos: -(1-p)^gamma * log(p)          summed over positive cells
  neg: alpha * (-p^gamma * log(1-p))  summed over negative cells
  loss = (pos_sum + neg_sum) / (n_pos + n_neg)

Pinned constants (protocol REV2 §5): gamma=2.0, alpha=0.5
(upstream param.py default), neg = one-time pinned subsample of
mask-zero cells.
"""
import torch

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5


def existence_focal(exist_scores, n_pos, gamma=FOCAL_GAMMA,
                    alpha=FOCAL_ALPHA):
    """exist_scores: (n_pos + n_neg,) sigmoid outputs — first n_pos rows
    are positives, rest are negatives. Returns (loss, dict)."""
    if exist_scores.ndim != 1 or len(exist_scores) <= n_pos:
        raise ValueError('exist_scores shape')
    if not torch.isfinite(exist_scores).all():
        raise ValueError('nonfinite existence scores')
    pos = exist_scores[:n_pos].clamp(1e-7, 1 - 1e-7)
    neg = exist_scores[n_pos:].clamp(1e-7, 1 - 1e-7)
    pos_loss = (-(1 - pos) ** gamma * torch.log(pos)).sum()
    neg_loss = (-neg ** gamma * torch.log(1 - neg)).sum() * alpha
    count = len(pos) + len(neg)
    return ((pos_loss + neg_loss) / count,
            {'exist_pos_n': int(len(pos)), 'exist_neg_n': int(len(neg)),
             'exist_pos_mean': float(pos.mean().detach()),
             'exist_neg_mean': float(neg.mean().detach()),
             'L_exist': float(((pos_loss + neg_loss) / count).detach())})
