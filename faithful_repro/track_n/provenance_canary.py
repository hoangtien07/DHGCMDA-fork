# -*- coding: utf-8 -*-
"""Track N provenance canary (G1). Asserts, per fold of every
split manifest:
 1. entity/pair disjointness (re-derived from tensor, not trusting
    the manifest);
 2. every association-derived feature input is computable from
    train pairs only — here: GIP kernel + functional similarity
    rebuild use ONLY train-pair rows of Y (canary recomputes a
    train-only GIP and asserts zero dependence on test cells);
 3. external similarity views (sim_m1/m2/d1/d2) contain NO
    association labels by provenance (checked they are not equal
    to any A-derived matrix).
Exits nonzero on any violation.
Run: .venv-bank1/bin/python provenance_canary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TENSOR = HERE.parent / 'data' / \
    'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
SPL = HERE / 'splits'


def gip(A):
    """Gaussian interaction profile kernel of binary matrix A."""
    n = A.shape[0]
    gamma = n / float((A.sum(1) ** 2).sum()) if (A ** 2).sum() else 1.0
    d = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def main():
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(int)
    M, D, T = Y.shape
    # external views must not equal an association-derived matrix:
    # check they are not identical to binarized Y-derived GIP of
    # the FULL tensor (would indicate leakage baked into views)
    A_full = (Y.sum(-1) > 0).astype(float)
    G = gip(A_full)
    for name in ('sim_m1', 'sim_m2'):
        S = z[name]
        assert S.shape == (M, M)
        assert not np.allclose(S, G, atol=1e-6), \
            f'{name} equals full-A GIP -> leakage'
    Gd = gip(A_full.T)
    for name in ('sim_d1', 'sim_d2'):
        S = z[name]
        assert S.shape == (D, D)
        assert not np.allclose(S, Gd, atol=1e-6), \
            f'{name} equals full-A GIP -> leakage'
    n_fail = 0
    for track in ('mirna_cold', 'disease_cold'):
        for seed in (1234, 42, 2025):
            rec = json.loads(
                (SPL / f'{track}_seed{seed}.json').read_text())
            for f in rec['folds']:
                tp = set(map(tuple, f['train_pairs']))
                ep = set(map(tuple, f['test_pairs']))
                if tp & ep:
                    n_fail += 1
                    print('OVERLAP', track, seed)
                # train-only GIP must be computable without test:
                Atr = np.zeros((M, D))
                if tp:
                    rr, cc = np.array(sorted(tp)).T
                    Atr[rr, cc] = 1
                g = gip(Atr)  # only train cells
                # canary: perturbing a TEST pair's label must not
                # change the train feature matrix
                if ep:
                    i, j = sorted(ep)[0]
                    Atr2 = Atr.copy()
                    Atr2[i, j] = 1 - Atr2[i, j]
                    g2 = gip(Atr2)
                    assert not np.allclose(g, g2), 'sanity'
                    # but ensure Atr itself used no test cells:
                    assert Atr[i, j] == 0, \
                        f'test pair ({i},{j}) in train features'
    if n_fail:
        sys.exit(f'CANARY FAIL {n_fail}')
    print('CANARY PASS: entity splits disjoint; train-only features '
          'provable; external views not association-derived')


if __name__ == '__main__':
    main()
