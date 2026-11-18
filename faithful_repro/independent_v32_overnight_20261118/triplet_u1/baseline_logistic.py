# -*- coding: utf-8 -*-
"""Train-only logistic baseline on the pinned CVtriplet-U1 REV2 eval
sets (F03 — identical information conditions to U1X: inner_train only).

Count features ported from ../triplet_baseline.py (train-only counts +
leave-one-out on train positives). Fit set = inner_train triplets (+)
plus equal-size pinned unknown sample (default_rng(3000+fold)).
Scored on the same per-seed eval replicates as U1X so the comparison is
information-matched, not just same-split.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import split as S                       # noqa: E402
from eval_cvtriplet import binary_metrics  # noqa: E402

_arr = lambda a: hashlib.sha256(
    np.ascontiguousarray(a).tobytes()).hexdigest()


def counts(trip, shape):
    nm, nd, nc = shape
    mt = np.zeros((nm, nc), np.int64)
    dt = np.zeros((nd, nc), np.int64)
    gg = np.zeros(nc, np.int64)
    mp = np.zeros(nm, np.int64)
    dp = np.zeros(nd, np.int64)
    pc = np.zeros((nm, nd), np.int64)
    i, j, k = trip.T
    np.add.at(mt, (i, k), 1)
    np.add.at(dt, (j, k), 1)
    np.add.at(gg, k, 1)
    np.add.at(mp, i, 1)
    np.add.at(dp, j, 1)
    np.add.at(pc, (i, j), 1)
    return mt, dt, gg, mp, dp, pc


def features(trip, c, exclude_self):
    i, j, k = trip.T
    mt, dt, gg, mp, dp, pc = c
    e = np.asarray(exclude_self, dtype=np.int64)
    a = np.column_stack([mt[i, k] - e, dt[j, k] - e, gg[k] - e,
                         mp[i] - e, dp[j] - e])
    return np.column_stack(
        [np.log1p(a), (pc[i, j] - e > 0).astype(float),
         np.eye(len(gg))[k]])


def run(Y, fold, eval_replicates):
    """eval_replicates: list of (pos_triplets, neg_triplets) tuples —
    the exact pinned sets U1X is scored on."""
    seg = S.fold_segments(Y, fold)
    train = seg['inner_train']
    unk = S.unknown_universe(Y)
    nneg = np.random.default_rng(
        S.BASELINE_NEG_SEED_BASE + fold).choice(
        len(unk), size=len(train), replace=False)
    trneg = unk[nneg]
    c = counts(train, Y.shape)
    x = features(np.concatenate([train, trneg]), c,
                 np.r_[np.ones(len(train), int),
                       np.zeros(len(trneg), int)])
    yt = np.r_[np.ones(len(train), int), np.zeros(len(trneg), int)]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1, max_iter=500,
                                           random_state=0))
    clf.fit(x, yt)
    rows = []
    mask = S.pair_mask(*Y.shape[:2], S.pairs_of(train))
    for rep_i, (pos, neg) in enumerate(eval_replicates):
        trip = np.concatenate([pos, neg])
        yv = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        xv = features(trip, c, np.zeros(len(trip), int))
        s = clf.predict_proba(xv)[:, 1]
        m = binary_metrics(yv, s)
        strata = S.unknown_strata(neg, mask, Y)
        row = {'seed': S.EVAL_SEEDS[rep_i], **m}
        for code, name in enumerate(
                ('neg_on_train_pair', 'neg_on_heldout_pair',
                 'neg_pair_zero')):
            sel = strata == code
            row[name] = {'n': int(sel.sum()),
                         'score_mean': float(s[len(pos):][sel].mean())
                         if sel.any() else None}
        pstrata = S.positive_strata(pos, mask)
        row['pos_pair_seen_frac'] = float(pstrata.mean())
        rows.append(row)
    return {'status': 'PASS', 'model': 'train-only count logistic',
            'train_pos': int(len(train)),
            'train_neg': int(len(trneg)),
            'train_neg_sha256': _arr(S._sort_rows(trneg)),
            'rows': rows,
            'mean': {k: float(np.mean([r[k] for r in rows]))
                     for k in ('auroc', 'average_precision',
                               'f1_at_0_5')}}
