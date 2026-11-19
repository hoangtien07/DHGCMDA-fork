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


def pair_features(Y_train, pairs, S_mi, S_dis, c_m=None, c_d=None,
                  lopo_pairs=None):
    """Class-aware features for `pairs` given train-masked Y_train
    (M,D,5). Counts c_m/c_d may be precomputed; if lopo_pairs is
    given, each row's OWN labels are subtracted (train features).
    Features: per-type mi count (5), per-type dis count (5),
    sim-weighted per-type count (5+5) -> (n,20)."""
    i, j = pairs[:, 0], pairs[:, 1]
    if c_m is None:
        c_m = np.zeros((Y_train.shape[0], 5))
        c_d = np.zeros((Y_train.shape[1], 5))
        tr = np.argwhere(Y_train.sum(-1) > 0)
        np.add.at(c_m, tr[:, 0], Y_train[tr[:, 0], tr[:, 1]])
        np.add.at(c_d, tr[:, 1], Y_train[tr[:, 0], tr[:, 1]])
    if lopo_pairs is not None:
        cm = c_m.copy()
        cd = c_d.copy()
        own = Y_train[lopo_pairs[:, 0], lopo_pairs[:, 1]]
        np.subtract.at(cm, lopo_pairs[:, 0], own)
        np.subtract.at(cd, lopo_pairs[:, 1], own)
        cm = np.clip(cm, 0, None)
        cd = np.clip(cd, 0, None)
    else:
        cm, cd = c_m, c_d
    X = [np.log1p(cm[i]), np.log1p(cd[j])]
    # similarity-weighted type counts: how much do entities similar
    # to me carry each type? uses S_mi[i,m]·c_m[m,k] etc.
    X.append(np.log1p(np.abs(S_mi[i] @ c_m)))
    X.append(np.log1p(np.abs(S_dis[j] @ c_d)))
    return np.concatenate(X, axis=1)


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
