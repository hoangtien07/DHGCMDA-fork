# -*- coding: utf-8 -*-
"""Track N baselines (B0–B2) — pure, deterministic-verifiable code.
Each returns score matrix (n_test_pairs, 5). Train-only by contract:
inputs are train_pairs/train label tensor only.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def prior_scores(Y_train_bin, test_pairs, kind='product_laplace1'):
    mi = Y_train_bin.sum(1)
    dis = Y_train_bin.sum(0)
    gl = Y_train_bin.sum()
    i, j = test_pairs.T
    if kind == 'global':
        s = np.full(len(test_pairs), gl)
    elif kind == 'mirna':
        s = mi[i]
    elif kind == 'disease':
        s = dis[j]
    else:
        s = np.log1p(mi[i]) + np.log1p(dis[j]) - np.log1p(gl)
    # broadcast same scalar to all 5 type channels (priors are
    # type-agnostic — honest floor)
    return np.repeat(np.asarray(s)[:, None], 5, axis=1)


def pair_features(Y_train_bin, test_pairs, sims):
    i, j = test_pairs.T
    mi, dis = Y_train_bin.sum(1), Y_train_bin.sum(0)
    X = [np.log1p(mi[i]), np.log1p(dis[j]),
         np.log1p(Y_train_bin.sum())]
    for S in sims:
        X.append(S[i, j] if S.shape[0] == Y_train_bin.shape[0]
                 else S[i, j])
    return np.stack(X, axis=1)


def logistic_scores(yt, X_tr, X_te, T=5):
    """Per-type logistic on train-only features.
    yt: multi-hot labels of the train pairs (n_train, T)."""
    n = X_te.shape[0]
    out = np.zeros((n, T))
    for k in range(T):
        if yt[:, k].ptp() == 0:
            out[:, k] = float(yt[:, k].mean())
            continue
        m = LogisticRegression(max_iter=500, C=1.0)
        m.fit(X_tr, yt[:, k])
        out[:, k] = m.predict_proba(X_te)[:, 1]
    return out


def label_prop_scores(Y_train, test_pairs, S_mi, S_dis, alpha=0.5):
    """Per-type two-sided label propagation:
    score_k(i,j) = sum_m S_mi[i,m]*Y[m,j,k] + sum_d Y[i,d,k]*S_dis[d,j].
    Y_train: (M, D, T) multi-hot, train-pair cells only."""
    i, j = test_pairs.T
    T = Y_train.shape[-1]
    out = np.zeros((len(test_pairs), T))
    for k in range(T):
        Ak = Y_train[..., k].astype(float)
        out[:, k] = (S_mi @ Ak + Ak @ S_dis)[i, j]
    return out * alpha
