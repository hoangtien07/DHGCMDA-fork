# -*- coding: utf-8 -*-
"""Track N baselines B0–B2 (REV-N1.1 repair).

B0: delegates to the VERIFIED per-type prior from
    independent_v32_outer_refit_20261117/prior.py (per-channel
    frequency, fallback-to-global, product_laplace1) — no
    type-agnostic broadcast.
B1: multi-label logistic on class-aware train-only features.
    Count features for a TRAIN pair subtract that pair's own
    labels (leave-one-pair-out); test features use outer-train
    counts only. Similarity features index correctly:
    S_mi[miRNA, miRNA], S_dis[disease, disease].
B2: per-type label propagation on train-masked multi-hot labels.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_PR = Path(__file__).resolve().parent.parent / \
    'independent_v32_outer_refit_20261117'
sys.path.insert(0, str(_PR))
import prior as _prior  # noqa: E402


def prior_scores(train_pairs, y_train, test_pairs, n_mi, n_dis):
    """B0: verified 5-channel prior. Returns dict of (n_test,5)."""
    counts = _prior.fit_counts(train_pairs, y_train, n_mi, n_dis)
    return _prior.predict_counts(counts, test_pairs)


def pair_features(c_m, c_d, pairs, S_mi, S_dis, y_own=None):
    """Class-aware features for `pairs` from fit counts c_m (M,5),
    c_d (D,5) built ONLY from fit pairs (inner-val excluded).
    y_own: (n,5) own labels of these pairs — required when the
    pairs themselves are IN the fit set: subtracts each pair's own
    labels from its entity counts (per-pair LOPO) for BOTH the
    direct count block AND the similarity-weighted block (the
    latter via S[i,i]*y_own and S[j,j]*y_own corrections).
    Features: direct counts (5+5) + sim-weighted counts (5+5)."""
    i, j = pairs[:, 0], pairs[:, 1]
    if y_own is not None:
        cm_i = c_m[i] - y_own
        cd_j = c_d[j] - y_own
        cm_i = np.clip(cm_i, 0, None)
        cd_j = np.clip(cd_j, 0, None)
        sw_m = np.abs(S_mi[i] @ c_m) - np.abs(
            S_mi[i, i])[:, None] * y_own
        sw_d = np.abs(S_dis[j] @ c_d) - np.abs(
            S_dis[j, j])[:, None] * y_own
    else:
        cm_i, cd_j = c_m[i], c_d[j]
        sw_m = np.abs(S_mi[i] @ c_m)
        sw_d = np.abs(S_dis[j] @ c_d)
    X = [np.log1p(cm_i), np.log1p(cd_j),
         np.log1p(np.maximum(sw_m, 0)),
         np.log1p(np.maximum(sw_d, 0))]
    return np.concatenate(X, axis=1)


def fit_counts(Y_fit):
    """Entity×type counts from fit-masked label tensor."""
    M, D, T = Y_fit.shape
    tr = np.argwhere(Y_fit.sum(-1) > 0)
    c_m = np.zeros((M, T)); c_d = np.zeros((D, T))
    np.add.at(c_m, tr[:, 0], Y_fit[tr[:, 0], tr[:, 1]])
    np.add.at(c_d, tr[:, 1], Y_fit[tr[:, 0], tr[:, 1]])
    return c_m, c_d


def logistic_scores(y_tr, X_tr, X_te, T=5, seed=0):
    """B1: per-type logistic on train-only (features, labels)."""
    out = np.zeros((X_te.shape[0], T))
    for k in range(T):
        col = y_tr[:, k]
        if np.ptp(col) == 0:
            out[:, k] = float(col.mean())
            continue
        m = LogisticRegression(max_iter=500, C=1.0,
                               random_state=seed)
        m.fit(X_tr, col)
        out[:, k] = m.predict_proba(X_te)[:, 1]
    return out


def label_prop_scores(Y_train, test_pairs, S_mi, S_dis, alpha=0.5):
    """B2: per-type propagation on train-masked multi-hot labels.
    score_k(i,j) = sum_m S_mi[i,m]Y[m,j,k] + sum_d Y[i,d,k]S_dis[d,j]
    Normalization: divide by train-only row mass (no test labels).
    """
    i, j = test_pairs[:, 0], test_pairs[:, 1]
    T = Y_train.shape[-1]
    # train-only normalizer: mass of similarity to entities that
    # carry ANY train label
    has_tr = (Y_train.sum(-1) > 0)
    mi_carriers = has_tr.any(1)   # (M,)
    dis_carriers = has_tr.any(0)  # (D,)
    nm = np.abs(S_mi[:, mi_carriers]).sum(1)
    nd = np.abs(S_dis[:, dis_carriers]).sum(1)
    out = np.zeros((len(test_pairs), T))
    for k in range(T):
        Ak = Y_train[..., k].astype(float)
        out[:, k] = (S_mi @ Ak + Ak @ S_dis)[i, j]
    out = out / np.maximum(nm[i] + nd[j], 1e-12)[:, None]
    return out * alpha
