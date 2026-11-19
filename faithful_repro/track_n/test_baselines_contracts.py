# -*- coding: utf-8 -*-
"""Track N baseline contract fixtures (G2 preflight; no training)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (prior_scores, pair_features, logistic_scores,
                       label_prop_scores)


def t_prior_formula():
    Y = np.array([[1, 0], [1, 1], [0, 0]])
    tp = np.array([[0, 1], [2, 0]])
    s = prior_scores(Y, tp, 'product_laplace1')
    # pair(0,1): log1p(1)+log1p(1)-log1p(3); pair(2,0): log1p(0)+log1p(2)-log1p(3)
    e = np.array([np.log1p(1)+np.log1p(1)-np.log1p(3),
                  np.log1p(0)+np.log1p(2)-np.log1p(3)])
    assert np.allclose(s[:, 0], e) and (s[:, 0] == s[:, 4]).all()


def t_logistic_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.zeros((200, 5)); y[X[:, 0] > 0, 2] = 1
    tr, te = np.arange(150), np.arange(150, 200)
    s = logistic_scores(y[tr], X[tr], X[te])
    # channel 2 should separate
    pos = s[y[te, 2] == 1, 2].mean() if (y[te, 2] == 1).any() else 0
    neg = s[y[te, 2] == 0, 2].mean()
    assert pos > neg


def t_labelprop_uses_train():
    A = np.zeros((2, 2, 5)); A[0, 0, 1] = 1; A[1, 1, 3] = 1
    Sm = np.eye(2) * 0.9 + 0.05
    Sd = np.eye(2) * 0.9 + 0.05
    tp = np.array([[0, 0], [1, 1]])
    s = label_prop_scores(A, tp, Sm, Sd)
    assert s[0, 1] > s[0, 0]   # type 1 present on pair (0,0)
    assert s[1, 3] > s[1, 0]   # type 3 present on pair (1,1)
    assert s[0, 3] < s[1, 3]   # cross-pair weaker


if __name__ == '__main__':
    t_prior_formula(); t_logistic_learns(); t_labelprop_uses_train()
    print('ALL 3 BASELINE CONTRACTS PASS')
