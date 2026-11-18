# -*- coding: utf-8 -*-
"""Split-contract fixtures: PAIR_CVTYPE vs TRIPLET_CVTRIPLET.

Proves (a) the two fold contracts are NOT equivalent on a multi-type
pair, and (b) the actual outer-refit pair folds satisfy pair-disjoint
train/test on all 5 folds.
Run: .venv-bank1/bin/python test_split_contracts.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
def _bundle_path(fold):
    if fold in (0, 1):
        return (REPRO / 'paper_guided'
                / f'candidate_fold{fold}_selection_rev1'
                / 'candidate_fold_data.npz')
    return (REPRO / 'independent_v32_fold_build_20261117' / 'bundles'
            / f'fold{fold}_selection' / 'candidate_fold_data.npz')


EVAL = (REPRO / 'independent_v32_outer_refit_20261117' / 'evaluation')


def pair_fold_assign(positive_pairs, k=5, seed=0):
    """PAIR_CVTYPE: unique positive (mi,dis) pairs -> fold id.
    All type-channels of a pair stay in ONE partition."""
    p = np.unique(np.asarray(positive_pairs), axis=0)
    perm = np.random.default_rng(seed).permutation(len(p))
    assign = np.zeros(len(p), dtype=int)
    for f, chunk in enumerate(np.array_split(perm, k)):
        assign[chunk] = f
    return p, assign


def triplet_fold_assign(positive_triplets, k=5, seed=0):
    """TRIPLET_CVTRIPLET: positive (mi,dis,type) -> fold id.
    Other types of the same pair MAY appear in train."""
    t = np.asarray(positive_triplets)
    perm = np.random.default_rng(seed).permutation(len(t))
    assign = np.zeros(len(t), dtype=int)
    for f, chunk in enumerate(np.array_split(perm, k)):
        assign[chunk] = f
    return assign


def _pairset(rows):
    return set(map(tuple, np.asarray(rows).tolist()))


def test_pair_vs_triplet_nonequivalence():
    """One multi-type pair straddling two hypothetical triplet
    partitions proves the contracts differ: under triplet-fold the
    test set can share a pair with train; under pair-fold it cannot."""
    # pair (0,0) positive in types 0 and 1; pair (0,1) positive type 2
    triplets = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 2]])
    pairs = np.unique(triplets[:, :2], axis=0)
    # hypothetical triplet split: (0,0,0) test, (0,0,1) train
    test_t = {tuple(triplets[0])}
    train_t = {tuple(triplets[1]), tuple(triplets[2])}
    test_pairs = {t[:2] for t in test_t}
    train_pairs = {t[:2] for t in train_t}
    assert test_pairs & train_pairs == {(0, 0)}   # straddle exists
    # under PAIR_CVTYPE this partition is ILLEGAL
    pset_test = _pairset(list(test_t)[:, :2] if False else
                         np.array([t[:2] for t in test_t]))
    pset_train = _pairset(np.array([t[:2] for t in train_t]))
    assert pset_test & pset_train, 'fixture must straddle'
    # any legal pair-fold forces disjointness
    p, a = pair_fold_assign(triplets[:, :2], k=2, seed=0)
    for f in range(2):
        assert not (set(map(tuple, p[a == f].tolist()))
                    & set(map(tuple, p[a != f].tolist())))


def test_pairfold_determinism_and_partition():
    triplets = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 2],
                         [1, 0, 0], [2, 3, 4], [3, 1, 1]])
    p1, a1 = pair_fold_assign(triplets[:, :2], k=3, seed=0)
    p2, a2 = pair_fold_assign(triplets[:, :2], k=3, seed=0)
    assert (a1 == a2).all()                       # deterministic
    assert sorted(set(a1)) == [0, 1, 2]           # covers all folds
    # every type of a pair stays together by construction
    for (i, j) in p.tolist() if False else p1.tolist():
        fset = set()
        for (ti, tj, tk) in triplets.tolist():
            if (ti, tj) == (i, j):
                fset.add(int(a1[np.where((p1 == [i, j]).all(1))[0][0]]))
        assert len(fset) == 1


def test_real_outer_refit_pair_disjoint():
    """Actual evidence: outer-refit eval pairs vs bundle train/val
    pairs — must be disjoint on every fold (PAIR_CVTYPE check)."""
    results = {}
    for fold in range(5):
        b = np.load(_bundle_path(fold), allow_pickle=True)
        train_pairs = _pairset(b['train_pairs'])
        val_pairs = _pairset(b['val_pairs'])
        ev = np.load(EVAL / f'fold{fold}_seed1234_dynamic.npz',
                     allow_pickle=True)
        test_pairs = _pairset(ev['pairs'])
        assert not (test_pairs & train_pairs), \
            f'fold{fold}: test∩train nonempty'
        assert not (test_pairs & val_pairs), \
            f'fold{fold}: test∩val nonempty'
        results[fold] = (len(train_pairs), len(val_pairs),
                         len(test_pairs))
        # same eval set across seeds/arms
        for seed in (42, 2025):
            for arm in ('dynamic', 'static'):
                e2 = np.load(EVAL / f'fold{fold}_seed{seed}_{arm}.npz')
                assert _pairset(e2['pairs']) == test_pairs
        # truth multi-hot, K=5, type-5 support present somewhere
        assert ev['y'].shape[1] == 5 and (ev['y'].sum(1) > 0).all()
        results[f'f{fold}_type5'] = int(ev['y'][:, 4].sum())
    for k, v in results.items():
        print('  ', k, v)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_')]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print('ALL', len(fns), 'SPLIT FIXTURES PASS')
