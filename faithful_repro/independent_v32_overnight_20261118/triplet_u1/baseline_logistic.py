# -*- coding: utf-8 -*-
"""Train-only logistic baselines on the pinned CVtriplet-U1 REV2 eval
replicates (AMENDMENT B2 — two explicit negative pools).

`mask_zero_pairs` (INFO-MATCHED): train negatives = triplet cells on
pairs absent from train_mask — the same negative pool the U1X
existence head sees (PU semantics: held-out-positive pairs are
mask-zero, same as upstream zero_index).

`all_unknown` (ORACLE-ONLY): negatives = cells of the whole ~Y,
including held-out positives and eval negatives — kept only as a
labeled transductive reference, never the info-matched comparison.

train/eval negative overlap is counted and reported per replicate.

Count features ported from ../triplet_baseline.py (train-only counts +
leave-one-out on train positives).
"""
import hashlib
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

POOLS = ('mask_zero_pairs', 'all_unknown')


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


def negative_pool(Y, mask, pool):
    """Return (N,3) triplet cells forming the baseline negative pool."""
    if pool == 'mask_zero_pairs':
        i, j = np.where(~mask.astype(bool))
        C = Y.shape[2]
        I = np.repeat(i, C)
        J = np.repeat(j, C)
        K = np.tile(np.arange(C, dtype=np.int64), len(i))
        return np.stack([I, J, K], axis=1)
    if pool == 'all_unknown':
        return S.unknown_universe(Y)
    raise ValueError(pool)


def run(Y, fold, eval_replicates, pool='mask_zero_pairs'):
    """eval_replicates: list of (pos_triplets, neg_triplets) — the
    exact pinned sets U1X is scored on. Returns metrics + the fitted
    scores per replicate (for artifact export) + overlap counts."""
    seg = S.fold_segments(Y, fold)
    train = seg['inner_train']
    mask = S.pair_mask(*Y.shape[:2], S.pairs_of(train))
    pool_cells = negative_pool(Y, mask, pool)
    nneg = np.random.default_rng(
        S.BASELINE_NEG_SEED_BASE + fold).choice(
        len(pool_cells), size=len(train), replace=False)
    trneg = S._sort_rows(pool_cells[nneg])
    c = counts(train, Y.shape)
    x = features(np.concatenate([train, trneg]), c,
                 np.r_[np.ones(len(train), int),
                       np.zeros(len(trneg), int)])
    yt = np.r_[np.ones(len(train), int), np.zeros(len(trneg), int)]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1, max_iter=500,
                                           random_state=0))
    clf.fit(x, yt)
    trneg_set = S.triplet_set(trneg)
    rows, scores = [], []
    for rep_i, (pos, neg) in enumerate(eval_replicates):
        trip = np.concatenate([pos, neg])
        yv = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        xv = features(trip, c, np.zeros(len(trip), int))
        s = clf.predict_proba(xv)[:, 1]
        scores.append(s)
        m = binary_metrics(yv, s)
        strata = S.unknown_strata(neg, mask, Y)
        overlap = sum(1 for t in neg.tolist()
                      if tuple(t) in trneg_set)
        row = {'seed': S.EVAL_SEEDS[rep_i], **m,
               'train_eval_neg_overlap': int(overlap)}
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
    return {'status': 'PASS',
            'model': 'train-only count logistic',
            'negative_pool': pool,
            'info_matched': pool == 'mask_zero_pairs',
            'train_pos': int(len(train)),
            'train_neg': int(len(trneg)),
            'train_neg_sha256': _arr(trneg),
            'rows': rows,
            'scores': scores,
            'mean': {k: float(np.mean([r[k] for r in rows]))
                     for k in ('auroc', 'average_precision',
                               'f1_at_0_5')}}
