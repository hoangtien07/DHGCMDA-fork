# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 evaluation (pure scoring + metrics).

Joint triplet score s(i,j,k) = sigma(exist_logit) * softmax(logits/T)[k]
(F01; justification in ../DESIGN_REVIEW.md §3). Pair score = bare channel
(Task P). binary_metrics is fixture-verified (perfect/inverted/hand cases).
"""
import numpy as np
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)


def binary_metrics(y_true, scores):
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(s).all() and set(np.unique(y)) <= {0, 1}
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, (s >= 0.5).astype(np.int64), average='binary',
        zero_division=0)
    return {'auroc': float(roc_auc_score(y, s)),
            'average_precision': float(average_precision_score(y, s)),
            'precision_at_0_5': float(prec),
            'recall_at_0_5': float(rec),
            'f1_at_0_5': float(f1)}


def stratified_scores(y, s, strata_codes, names):
    out = {}
    for code, name in enumerate(names):
        sel = np.asarray(strata_codes) == code
        if not sel.any():
            out[name] = {'n': 0}
            continue
        yy = np.asarray(y)[sel]
        out[name] = {'n': int(sel.sum())}
        out[name]['score_mean'] = float(np.asarray(s)[sel].mean())
        if len(np.unique(yy)) > 1:
            out[name].update(binary_metrics(yy, np.asarray(s)[sel]))
    return out


def joint_score(exist_prob, type_probs, triplets, pair_row):
    """s(i,j,k)=exist_prob[pair] * type_probs[pair,k]; pair_row maps each
    triplet to its row in the prediction arrays."""
    k = np.asarray(triplets)[:, 2]
    return (np.asarray(exist_prob)[pair_row]
            * np.asarray(type_probs)[pair_row, k])


def pair_score(exist_prob, pair_row):
    return np.asarray(exist_prob)[pair_row]
