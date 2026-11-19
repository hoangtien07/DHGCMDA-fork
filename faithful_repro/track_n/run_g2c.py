# -*- coding: utf-8 -*-
"""Track N G2 runner CORRECTED (REV-N1.2 / WP1-WP3 repairs).

Fixes vs run_g2.py (whose runs_g2/ output is SUPERSEDED_UNVERIFIED):
- Y_fit is masked to OUTER-train MINUS inner-val pairs only;
  inner-val labels never enter counts, features, propagation or
  normalization.
- pair_features LOPO is per-pair (direct counts AND sim-weighted
  counts corrected via S[i,i]*y_own, S[j,j]*y_own).
- Inner-val rows scored with Y_fit features (no own labels present
  by construction).
- tau selection happens ONLY on inner-val scores vs inner-val
  labels.

New namespace: runs_g2c/. Freezes scores/labels/indices/taus/hashes.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from baselines import (prior_scores, pair_features, logistic_scores,
                       label_prop_scores, fit_counts)

TENSOR = ROOT / 'data' / \
    'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
EVALDIR = ROOT / 'independent_v32_outer_refit_20261117' / 'evaluation'
OUT = HERE / 'runs_g2c'
SEEDS = (1234, 42, 2025)
TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
INNER_FRAC = 0.10


def track_folds(track, Y):
    if track == 'pair':
        for f in range(5):
            z = np.load(EVALDIR /
                        f'fold{f}_seed1234_static.npz',
                        allow_pickle=True)
            te = z['pairs']
            teset = set(map(tuple, te.tolist()))
            tr = np.array([p for p in np.argwhere(Y.sum(-1) > 0)
                           if tuple(p) not in teset])
            yield f, tr, te
    else:
        for seed in SEEDS:
            rec = json.loads(
                (HERE / 'splits' /
                 f'{track}_seed{seed}.json').read_text())
            for f, fold in enumerate(rec['folds']):
                yield (f, seed), np.asarray(
                    fold['train_pairs']), np.asarray(
                    fold['test_pairs'])


def masked(Y, trp):
    Yt = np.zeros_like(Y)
    Yt[trp[:, 0], trp[:, 1]] = Y[trp[:, 0], trp[:, 1]]
    return Yt


def norm_fit(sc):
    lo = sc.min(0)
    hi = np.maximum(sc.max(0) - lo, 1e-12)
    return lo, hi


def select_tau(pn, y):
    taus = np.zeros(y.shape[1])
    for k in range(y.shape[1]):
        best, tau = -1, 0.5
        for t in TAU_GRID:
            pr = (pn[:, k] >= t).astype(int)
            tp = int((pr & (y[:, k] == 1)).sum())
            fp = int((pr & (y[:, k] == 0)).sum())
            fn = int(((1 - pr) & (y[:, k] == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best, tau = f1, t
        taus[k] = tau
    return taus


def run_all():
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(int)
    Sm = (z['sim_m1'] + z['sim_m2']) / 2.0
    Sd = (z['sim_d1'] + z['sim_d2']) / 2.0
    M, D, T = Y.shape
    man = {'supersedes': 'runs_g2 (SUPERSEDED_UNVERIFIED)',
           'cells': []}
    for track in ('pair', 'mirna_cold', 'disease_cold'):
        for fold_id, trp, tep in track_folds(track, Y):
            seeds = SEEDS if track == 'pair' else (fold_id[1],)
            fnum = fold_id if track == 'pair' else fold_id[0]
            for sd in seeds:
                rng = np.random.default_rng(sd)
                perm = rng.permutation(len(trp))
                niv = max(1, int(INNER_FRAC * len(trp)))
                iv, ftr = perm[:niv], perm[niv:]
                # Y_fit EXCLUDES inner-val pairs (repair)
                Yfit = masked(Y, trp[ftr])
                c_m, c_d = fit_counts(Yfit)
                y_fit = Yfit[trp[ftr, 0], trp[ftr, 1]]
                y_iv = Y[trp[iv, 0], trp[iv, 1]]
                y_te = Y[tep[:, 0], tep[:, 1]]
                # ---- B0 (fit counts only)
                b0 = prior_scores(trp[ftr], y_fit, tep, M, D)
                b0_iv = prior_scores(trp[ftr], y_fit, trp[iv], M, D)
                # ---- B1: LOPO per-pair on fit rows
                X_tr = pair_features(c_m, c_d, trp[ftr], Sm, Sd,
                                     y_own=y_fit)
                X_iv = pair_features(c_m, c_d, trp[iv], Sm, Sd)
                X_te = pair_features(c_m, c_d, tep, Sm, Sd)
                b1_te = logistic_scores(y_fit, X_tr, X_te, T, sd)
                b1_iv = logistic_scores(y_fit, X_tr, X_iv, T, sd)
                # ---- B2 on Y_fit only
                b2_te = label_prop_scores(Yfit, tep, Sm, Sd)
                b2_iv = label_prop_scores(Yfit, trp[iv], Sm, Sd)
                scores = dict(b0)
                scores['B1_logistic'] = b1_te
                scores['B2_labelprop'] = b2_te
                iv_scores = dict(b0_iv)
                iv_scores['B1_logistic'] = b1_iv
                iv_scores['B2_labelprop'] = b2_iv
                taus, norms = {}, {}
                for name, sc in scores.items():
                    lo, hi = norm_fit(iv_scores[name])
                    norms[name] = [lo.tolist(), hi.tolist()]
                    taus[name] = select_tau(
                        np.clip((iv_scores[name] - lo) / hi, 0, 1),
                        y_iv).tolist()
                    scores[name] = np.clip((sc - lo) / hi, 0, 1)
                cdir = OUT / track / f'fold{fnum}_seed{sd}'
                cdir.mkdir(parents=True, exist_ok=True)
                np.savez(cdir / 'scores.npz',
                         test_pairs=tep, y_test=y_te,
                         train_pairs=trp[ftr], iv_pairs=trp[iv],
                         y_iv=y_iv,
                         **scores)
                np.savez(cdir / 'iv_scores.npz',
                         iv_pairs=trp[iv], y_iv=y_iv, **iv_scores)
                (cdir / 'taus.json').write_text(
                    json.dumps({'taus': taus, 'norm': norms}))
                man['cells'].append({
                    'track': track, 'fold': fnum, 'seed': sd,
                    'n_fit': len(ftr), 'n_iv': len(iv),
                    'n_test': len(tep),
                    'scores_sha256': hashlib.sha256(
                        (cdir / 'scores.npz').read_bytes()
                    ).hexdigest()})
        print(track, 'done')
    (OUT / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    run_all()
