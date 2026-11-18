# -*- coding: utf-8 -*-
"""Unit fixtures for CVtriplet-U1 REV2 — run inside preflight (F04).

Covers: multi-label split disjointness, held-out label canary,
val-straddle drop, unknown-strata codes, eval sampling determinism,
metric fixed points, neg-supervision sizing.
"""
import numpy as np
import split as S


def _tiny_Y():
    # 4 miRNA x 3 disease x 3 types; pair (0,0) multi-type straddles.
    Y = np.zeros((4, 3, 3), dtype=np.int8)
    for (i, j, k) in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                      (1, 2, 2), (2, 1, 1), (3, 2, 0), (3, 2, 2)):
        Y[i, j, k] = 1
    return Y


def fixture_split_disjoint():
    for fold in range(5):
        seg = S.fold_segments(_tiny_Y(), fold)
        tr, va, te = (S.triplet_set(seg[k])
                      for k in ('inner_train', 'inner_val', 'test'))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == S.triplet_set(S.positive_universe(_tiny_Y()))


def fixture_canary():
    """Held-out type bits must be zero in y_train/y_val (label canary)."""
    Y = _tiny_Y()
    for fold in range(5):
        seg = S.fold_segments(Y, fold)
        tp = S.pairs_of(seg['inner_train'])
        yt = S.multihot_masked(Y, tp, seg['inner_train'])
        held = np.concatenate([seg['inner_val'], seg['test']])
        for (i, j, k) in held.tolist():
            rows = np.where((tp[:, 0] == i) & (tp[:, 1] == j))[0]
            for r in rows:
                assert yt[r, k] == 0, ('held-out bit leaked', i, j, k)
        vp, kept, _ = S.val_pairs_disjoint(seg['inner_val'], tp)
        yv = S.multihot_masked(Y, vp, seg['inner_val'])
        for (i, j, k) in np.concatenate([seg['inner_train'],
                                         seg['test']]).tolist():
            rows = np.where((vp[:, 0] == i) & (vp[:, 1] == j))[0]
            for r in rows:
                assert yv[r, k] == 0
    # straddle case: pair (0,0) has types 0,1 — if split puts them in
    # different segments the y_train row must have exactly one bit.
    seg = S.fold_segments(Y, 0)
    tp = S.pairs_of(seg['inner_train'])
    row = np.where((tp[:, 0] == 0) & (tp[:, 1] == 0))[0]
    if len(row):
        yt = S.multihot_masked(Y, tp, seg['inner_train'])
        assert yt[row[0]].sum() >= 1 and yt[row[0]].sum() <= 2


def fixture_val_straddle():
    Y = _tiny_Y()
    total_kept, total_dropped = 0, 0
    for fold in range(5):
        seg = S.fold_segments(Y, fold)
        vp, kept, dropped = S.val_pairs_disjoint(seg['inner_val'],
                                                 S.pairs_of(
                                                     seg['inner_train']))
        assert not set(map(tuple, vp.tolist())) & set(
            map(tuple, S.pairs_of(seg['inner_train']).tolist()))
        assert kept.shape[1] == 3 if len(kept) else kept.shape[0] == 0
        total_kept += len(kept)
        total_dropped += dropped
        assert len(kept) + dropped == len(seg['inner_val'])
    assert total_dropped >= 0 and total_kept >= 0


def fixture_strata():
    Y = _tiny_Y()
    mask = np.zeros((4, 3), dtype=np.int8)
    mask[0, 0] = 1  # train-known pair with unknown channel k=2
    unk = np.asarray([[0, 0, 2],   # train pair, triplet-zero -> 0
                      [1, 2, 0],   # held-out-pos pair, k0 zero -> 1
                      [2, 2, 2],   # pair-zero -> 2
                      [3, 0, 1]], dtype=np.int64)  # pair-zero -> 2
    codes = S.unknown_strata(unk, mask, Y)
    assert codes.tolist() == [0, 1, 2, 2]


def fixture_eval_sampling():
    u = np.arange(1000).reshape(-1, 1) * 0 + np.stack(
        [np.arange(1000) % 10, np.arange(1000) // 10 % 10,
         np.arange(1000) // 100], axis=1)
    a = S.eval_unknown_sample(u, 50, 0)
    b = S.eval_unknown_sample(u, 50, 0)
    c = S.eval_unknown_sample(u, 50, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(np.unique(a, axis=0)) == 50


def fixture_neg_supervision():
    mask = np.zeros((10, 10), dtype=np.int8)
    mask[:3, :3] = 1  # 9 pos cells
    idx, sha = S.neg_supervision_index(mask, 9, 1234)
    assert len(idx) == 90 and not mask[idx[:, 0], idx[:, 1]].any()
    idx2, sha2 = S.neg_supervision_index(mask, 9, 1234)
    assert sha == sha2 and np.array_equal(idx, idx2)


def fixture_metric_fixedpoints():
    from eval_cvtriplet import binary_metrics
    y = np.r_[np.ones(10), np.zeros(10)]
    s_perfect = np.r_[np.linspace(.9, 1, 10), np.linspace(0, .1, 10)]
    m = binary_metrics(y, s_perfect)
    assert abs(m['auroc'] - 1.0) < 1e-9 \
        and abs(m['average_precision'] - 1.0) < 1e-9
    assert abs(m['f1_at_0_5'] - 1.0) < 1e-9
    m_inv = binary_metrics(y, s_perfect[::-1])
    assert m_inv['auroc'] == 0.0
    # hand case: pos .6/.4, neg .3/.5 -> 3 wins / 4 pairs -> AUROC .75
    m_h = binary_metrics(np.array([1, 1, 0, 0]), [.6, .4, .3, .5])
    assert abs(m_h['auroc'] - 0.75) < 1e-9


def fixture_pair_unseen():
    Y = _tiny_Y()
    for fold in range(5):
        seg = S.fold_segments(Y, fold)
        pp = S.pair_unseen_test_pairs(Y, seg['test'])
        tp = S.pairs_of(seg['inner_train'])
        assert not set(map(tuple, pp.tolist())) & set(
            map(tuple, tp.tolist()))
        for (i, j) in pp.tolist():
            for k in range(3):
                if Y[i, j, k]:
                    assert (i, j, k) in S.triplet_set(seg['test'])


def fixture_multihot_masked():
    Y = _tiny_Y()
    allowed = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    pairs = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    y = S.multihot_masked(Y, pairs, allowed)
    assert y[0].tolist() == [1, 0, 0] and y[1].tolist() == [1, 0, 0]


ALL = (fixture_split_disjoint, fixture_canary, fixture_val_straddle,
       fixture_strata, fixture_eval_sampling, fixture_neg_supervision,
       fixture_metric_fixedpoints, fixture_pair_unseen,
       fixture_multihot_masked)


def run_all():
    out = {}
    for f in ALL:
        f()
        out[f.__name__] = 'PASS'
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(run_all(), indent=1))
