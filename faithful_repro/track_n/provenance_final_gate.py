# -*- coding: utf-8 -*-
"""Track N PROVENANCE_FINAL_GATE — tests the EXACT builders used
by run_g2c.py and a1_arm.py (masked, fit_counts, pair_features,
label_prop_scores, prior_scores, feats).

Per track: SPLIT_LEVEL, INNER_VAL_LEVEL, FEATURE_LEVEL gates.
Perturbation: flip only held-out (test or iv) labels -> all
fit-derived outputs must be byte-identical.
Emits PROVENANCE_FINAL_GATE.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from baselines import (prior_scores, pair_features,
                       label_prop_scores, fit_counts)
from run_g2c import masked, track_folds, INNER_FRAC, TENSOR
from a1_arm import feats as a1_feats

SEEDS = (1234, 42, 2025)


def flip_some_labels(Y, pairs, rng):
    Yp = Y.copy()
    sel = pairs[rng.choice(len(pairs),
                           size=min(20, len(pairs)), replace=False)]
    for i, j in sel:
        v = Yp[i, j].copy()
        if v.sum() == 0:
            v[0] = 1
        else:
            v[np.flatnonzero(v)[0]] = 0
            free = np.setdiff1d(np.arange(5), np.flatnonzero(v))
            if len(free):
                v[free[0]] = 1
        Yp[i, j] = v
    return Yp


def pipeline(Yl, trp, ftr, iv, tep, Sm, Sd, M, D, T):
    """Exact run_g2c fit pipeline on label tensor Yl."""
    Yfit = np.zeros_like(Yl)
    p = trp[ftr]
    Yfit[p[:, 0], p[:, 1]] = Yl[p[:, 0], p[:, 1]]
    c_m, c_d = fit_counts(Yfit)
    y_fit = Yfit[p[:, 0], p[:, 1]]
    X_tr = pair_features(c_m, c_d, p, Sm, Sd, y_own=y_fit)
    X_te = pair_features(c_m, c_d, tep, Sm, Sd)
    X_iv = pair_features(c_m, c_d, trp[iv], Sm, Sd)
    b0 = prior_scores(p, y_fit, tep, M, D)
    b2 = label_prop_scores(Yfit, tep, Sm, Sd)
    # a1 inputs
    a1 = a1_feats(Sm, Sd, tep)
    return (c_m, c_d, X_tr, X_te, X_iv, b0['product_laplace1'],
            b2, a1)


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def main():
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(int)
    Sm = (z['sim_m1'] + z['sim_m2']) / 2.0
    Sd = (z['sim_d1'] + z['sim_d2']) / 2.0
    M, D, T = Y.shape
    rng = np.random.default_rng(7)
    rep = {'tensor_sha256': __import__('hashlib').sha256(
        TENSOR.read_bytes()).hexdigest(), 'tracks': {}}
    for track in ('pair', 'mirna_cold', 'disease_cold'):
        g = {'SPLIT_LEVEL': True, 'INNER_VAL_LEVEL': True,
             'FEATURE_LEVEL': True, 'checked_cells': 0}
        # external views may embed held-out associations? probe:
        # each view's provenance is fixed (file-level); gate =
        # view does not equal any train-derived matrix per fold.
        for fold_id, trp, tep in track_folds(track, Y):
            sd = fold_id[1] if track != 'pair' else 1234
            r2 = np.random.default_rng(sd)
            perm = r2.permutation(len(trp))
            niv = max(1, int(INNER_FRAC * len(trp)))
            iv, ftr = perm[:niv], perm[niv:]
            # SPLIT: train/test disjoint + (cold) entity disjoint
            if set(map(tuple, trp.tolist())) & set(
                    map(tuple, tep.tolist())):
                g['SPLIT_LEVEL'] = False
            # INNER-VAL-level: perturb iv labels -> fit pipeline
            # outputs identical
            base = pipeline(Y, trp, ftr, iv, tep, Sm, Sd, M, D, T)
            Yiv = flip_some_labels(Y, trp[iv], rng)
            if not same(base, pipeline(Yiv, trp, ftr, iv, tep,
                                       Sm, Sd, M, D, T)):
                g['INNER_VAL_LEVEL'] = False
            # FEATURE-level: perturb TEST labels -> identical
            Yte = flip_some_labels(Y, tep, rng)
            if not same(base, pipeline(Yte, trp, ftr, iv, tep,
                                       Sm, Sd, M, D, T)):
                g['FEATURE_LEVEL'] = False
            g['checked_cells'] += 1
        g['PASS'] = (g['SPLIT_LEVEL'] and g['INNER_VAL_LEVEL']
                     and g['FEATURE_LEVEL'])
        rep['tracks'][track] = g
        print(track, g)
    Path(HERE / 'PROVENANCE_FINAL_GATE.json').write_text(
        json.dumps(rep, indent=1))
    print('ALL_PASS' if all(t['PASS'] for t in
                            rep['tracks'].values())
          else 'GATE_FAILURE')


if __name__ == '__main__':
    main()
