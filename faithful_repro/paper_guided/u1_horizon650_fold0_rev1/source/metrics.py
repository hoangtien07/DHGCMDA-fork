# -*- coding: utf-8 -*-
"""Supervised type objective + metric conventions (declared, frozen).

Loss (design §Task,loss and metrics): stable log-softmax on logits,
normalized by total weighted positive mass over the evaluated pairs:

    L = - sum_{i,k} w_k * y_ik * logsoftmax(logits)_ik
          / sum_{i,k} w_k * y_ik

This is a scalar normalization of the previously used weighted-sum
objective when weights and data are fixed; it does not change the
optimum structure. Class weights come from contracts.class_weights_train
(beta=0.99999 effective counts, sum normalized to C) computed on TRAIN
labels only. No existence objective exists at this stage.

Metrics: primary = argmax-onehot vs multi-hot macro F1 (per-class F1
averaged, zero-division -> 0). Also reported: micro F1, per-class
precision/recall/support, any-hit rate, and the two LEGACY hybrid
conventions kept only for continuity with archived reports — they are
labeled legacy and never mixed into the primary number.
"""
import numpy as np


def type_loss(logits, multi_hot, class_weights):
    """Normalized weighted multi-label softmax CE on logits.

    logits:        torch tensor (n_pairs, C) — keeps device/grad graph
    multi_hot:     (n_pairs, C) 0/1 labels (torch or numpy; detached to
                   the logits' device)
    class_weights: (C,) nonnegative weights (torch or numpy; detached)
    Returns scalar loss tensor. Gradients flow to logits only.
    """
    import torch
    import torch.nn.functional as F
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(np.asarray(logits))
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('logits must be a finite 2-D tensor')
    def _detached(v):
        if torch.is_tensor(v):
            return v.detach().to(dtype=logits.dtype,
                                 device=logits.device)
        return torch.as_tensor(np.asarray(v), dtype=logits.dtype,
                               device=logits.device)
    y = _detached(multi_hot)
    w = _detached(class_weights)
    if y.shape != logits.shape:
        raise ValueError('multi_hot must match logits shape')
    if not ((y == 0) | (y == 1)).all():
        raise ValueError('multi_hot must be binary 0/1')
    if w.ndim != 1 or w.shape[0] != logits.shape[1] \
            or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError('class_weights must be finite, nonnegative, '
                         'length C')
    logp = F.log_softmax(logits, dim=-1)
    num = -(w.view(1, -1) * y * logp).sum()
    den = (w.view(1, -1) * y).sum()
    if float(den) <= 0:
        raise ValueError('zero weighted positive mass')
    return num / den


def _harmonic(p, r):
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def type_metrics(truth, scores):
    """Metric conventions on argmax-onehot predictions vs multi-hot truth.

    truth:  (n,C) multi-hot (nonzero = positive)
    scores: (n,C) class scores/probabilities — argmax only is used; no
            additional softmax is applied.
    """
    yr = np.asarray(truth)
    if yr.ndim != 2 or yr.size == 0:
        raise ValueError('truth must be a non-empty 2-D array')
    if not np.isfinite(yr.astype(float)).all() \
            or not np.isin(yr, (0, 1, 0.0, 1.0)).all():
        raise ValueError('truth must be binary multi-hot')
    if not yr.any():
        raise ValueError('all-zero truth is meaningless for type ranking')
    y = (yr != 0).astype(np.int64)
    s = np.asarray(scores)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError('scores must be finite and match truth')
    pred = s.argmax(axis=1)
    n, C = y.shape
    onehot = np.zeros_like(y)
    onehot[np.arange(n), pred] = 1
    tp = (onehot & y.astype(bool)).sum(axis=0).astype(np.float64)
    fp = (onehot.astype(bool) & ~y.astype(bool)).sum(axis=0)
    fn = (~onehot.astype(bool) & y.astype(bool)).sum(axis=0)
    prec = np.divide(tp, tp + fp, out=np.zeros(C), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(C), where=(tp + fn) > 0)
    f1c = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros(C), where=(prec + rec) > 0)
    micro_p = tp.sum() / (tp.sum() + fp.sum()) if (tp + fp).sum() else 0.0
    micro_r = tp.sum() / (tp.sum() + fn.sum()) if (tp + fn).sum() else 0.0
    hit = y[np.arange(n), pred] > 0
    per_class_recall = [float(np.mean(pred[y[:, k] > 0] == k))
                        for k in range(C) if y[:, k].any()]
    first = y.argmax(1)
    fp_ = float(np.mean(first == pred))
    fr_ = float(np.mean([np.mean(pred[first == k] == k)
                         for k in np.unique(first)]))
    return {
        'primary': {
            'macro_f1': float(f1c.mean()),
            'definition': 'argmax-onehot vs multi-hot, per-class F1 '
                          'averaged over all C channels, zero_division=0'},
        'micro_f1': _harmonic(micro_p, micro_r),
        'per_class': {
            'precision': prec.tolist(), 'recall': rec.tolist(),
            'f1': f1c.tolist(), 'support': y.sum(axis=0).tolist()},
        'any_hit_rate': float(hit.mean()),
        'legacy_any_hit_hybrid': {
            'precision': float(hit.mean()),
            'recall': float(np.mean(per_class_recall)),
            'f1': _harmonic(float(hit.mean()),
                            float(np.mean(per_class_recall))),
            'label': 'LEGACY diagnostic — not the primary metric'},
        'legacy_first_label_hybrid': {
            'precision': fp_, 'recall': fr_,
            'f1': _harmonic(fp_, fr_), 'label': 'LEGACY diagnostic'},
    }
