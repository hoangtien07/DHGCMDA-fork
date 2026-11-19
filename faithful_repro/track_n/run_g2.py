# -*- coding: utf-8 -*-
"""Track N G2 runner: B0–B2 real-data baselines on the 3 locked
tracks. Freezes scores/labels/indices/taus + manifests per cell.
Metric computation lives in metrics_g2.py (independent recompute).

Usage: .venv-bank1/bin/python run_g2.py [track] [fold] [seed]
       (no args = run all cells)
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
                       label_prop_scores)

TENSOR = ROOT / 'data' / \
    'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
EVALDIR = ROOT / 'independent_v32_outer_refit_20261117' / 'evaluation'
OUT = HERE / 'runs_g2'
SEEDS = (1234, 42, 2025)
TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
INNER_FRAC = 0.10


def load():
    z = np.load(TENSOR, allow_pickle=True)
    return (z['Y'].astype(int), z['sim_m1'], z['sim_m2'],
            z['sim_d1'], z['sim_d2'])


def track_folds(track, Y):
    """Yield (fold_id, train_pairs, test_pairs). Pair track folds
    recovered from the archived eval npz partition."""
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


def norm_fit(sc_tr):
    lo = sc_tr.min(0)
    hi = sc_tr.max(0)
    return lo, np.maximum(hi - lo, 1e-12)


def norm_apply(sc, lo, hi):
    return np.clip((sc - lo) / hi, 0, 1)


def select_tau(p_tr_norm, y_tr):
    taus = np.zeros(y_tr.shape[1])
    for k in range(y_tr.shape[1]):
        best, tau = -1, 0.5
        for t in TAU_GRID:
            pr = (p_tr_norm[:, k] >= t).astype(int)
            tp = (pr & (y_tr[:, k] == 1)).sum()
            fp = (pr & (y_tr[:, k] == 0)).sum()
            fn = ((1 - pr) & (y_tr[:, k] == 1)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best, tau = f1, t
        taus[k] = tau
    return taus


def masked(Y, trp):
    Yt = np.zeros_like(Y)
    Yt[trp[:, 0], trp[:, 1]] = Y[trp[:, 0], trp[:, 1]]
    return Yt


def run_all():
    Y, Sm1, Sm2, Sd1, Sd2 = load()
    M, D, T = Y.shape
    man = {'cells': []}
    for track in ('pair', 'mirna_cold', 'disease_cold'):
        for fold_id, trp, tep in track_folds(track, Y):
            if track == 'pair':
                f, seed = fold_id if isinstance(
                    fold_id, tuple) else (fold_id, None)
                seeds = SEEDS
                fnum = fold_id
            else:
                fnum, seed = fold_id
                seeds = (seed,)
            for sd in seeds:
                cdir = OUT / track / f'fold{fnum}_seed{sd}'
                cdir.mkdir(parents=True, exist_ok=True)
                Ytr = masked(Y, trp)
                y_tr = Ytr[trp[:, 0], trp[:, 1]]
                y_te = Y[tep[:, 0], tep[:, 1]]
                # inner-val split for tau (train-only)
                rng = np.random.default_rng(sd)
                perm = rng.permutation(len(trp))
                niv = max(1, int(INNER_FRAC * len(trp)))
                iv, ftr = perm[:niv], perm[niv:]
                scores = {}
                # B0: 4 prior variants (train-only counts)
                b0 = prior_scores(trp[ftr], y_tr[ftr], tep, M, D)
                for name, sc in b0.items():
                    scores[f'B0_{name}'] = sc
                b0_iv = prior_scores(trp[ftr], y_tr[ftr],
                                     trp[iv], M, D)
                # B1 logistic (LOPO train features)
                X_tr = pair_features(Ytr, trp[ftr], Sm1, Sd1,
                                     lopo_pairs=trp[ftr])
                X_iv = pair_features(Ytr, trp[iv], Sm1, Sd1,
                                     lopo_pairs=trp[iv])
                X_te = pair_features(Ytr, tep, Sm1, Sd1)
                scores['B1_logistic'] = logistic_scores(
                    y_tr[ftr], X_tr, X_te, T, seed=sd)
                b1_iv = logistic_scores(y_tr[ftr], X_tr, X_iv, T,
                                        seed=sd)
                # B2 label prop (train-masked)
                Smm = (Sm1 + Sm2) / 2.0
                Sdd = (Sd1 + Sd2) / 2.0
                scores['B2_labelprop'] = label_prop_scores(
                    Ytr, tep, Smm, Sdd)
                b2_iv = label_prop_scores(Ytr, trp[iv], Smm, Sdd)
                # taus: fit normalization on inner-val scores
                taus = {}
                for name, sc in scores.items():
                    iv_sc = {'B0_' + k: v for k, v in
                             b0_iv.items()}.get(name)
                    if iv_sc is None:
                        iv_sc = b1_iv if name == 'B1_logistic' \
                            else b2_iv
                    lo, hi = norm_fit(iv_sc)
                    taus[name] = select_tau(
                        norm_apply(iv_sc, lo, hi),
                        y_tr[iv]).tolist()
                    scores[name] = norm_apply(sc, lo, hi)
                np.savez(cdir / 'scores.npz',
                         test_pairs=tep, y_test=y_te,
                         **{k: v for k, v in scores.items()})
                (cdir / 'taus.json').write_text(json.dumps(taus))
                man['cells'].append({
                    'track': track, 'fold': fnum, 'seed': sd,
                    'n_train': len(trp), 'n_test': len(tep),
                    'scores_sha256': hashlib.sha256(
                        (cdir / 'scores.npz').read_bytes()
                    ).hexdigest()})
                print(track, fnum, sd, 'done', len(tep))
    (OUT / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    run_all()
