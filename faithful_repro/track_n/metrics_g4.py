# -*- coding: utf-8 -*-
"""Independent metric recompute for G4 A1 fleet + paired diffs vs
corrected baselines (runs_g2c) and U1 static frozen logits."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics_g2c import metrics_for  # noqa

OUT = HERE / 'runs_g4'
G2C = HERE / 'runs_g2c'
EVALDIR = (HERE.parent / 'independent_v32_outer_refit_20261117'
           / 'evaluation')
SEEDS = (1234, 42, 2025)
KEYS = ('macroF1', 'top1_firstpos_harmonic', 'minority_recall',
        'multitype_macroF1')


def cell_metrics(track_dir):
    z = np.load(track_dir / 'scores.npz')
    y = z['y_test']
    taus = json.loads((track_dir / 'taus.json').read_text())['taus']
    out = {}
    for k in z.files:
        if k in ('test_pairs', 'y_test', 'train_pairs', 'iv_pairs',
                 'y_iv'):
            continue
        # independent recompute: re-derive normalized scores the same
        # way metrics_g2c does — scores.npz stores raw sigmoid probs
        out[k] = metrics_for(y, z[k], taus[k])
    return out, y, z['test_pairs']


def main():
    rows = []
    for f in range(5):
        for s in SEEDS:
            m, y, tep = cell_metrics(OUT / f'fold{f}_seed{s}')
            a1 = m['A1_bce']
            # corrected baseline on same cell (pair track)
            c = G2C / 'pair' / f'fold{f}_seed{s}'
            bm, by, _ = cell_metrics(c) if c.exists() else ({}, y, tep)
            best_b = max((bm or {}).items(),
                         key=lambda kv: kv[1]['macroF1'])
            # U1 static frozen logits (same test pairs)
            ez = np.load(EVALDIR / f'fold{f}_seed1234_static.npz',
                         allow_pickle=True)
            p_u1 = 1 / (1 + np.exp(-ez['logits']))
            u1_taus = np.full(5, 0.5)  # no inner-val for U1: fixed 0.5
            u1 = metrics_for(y, p_u1, u1_taus)
            rows.append({
                'fold': f, 'seed': s,
                'a1': {k: a1[k] for k in KEYS},
                'best_baseline': best_b[0],
                'b': {k: best_b[1][k] for k in KEYS},
                'd_a1_vs_bestb': {k: a1[k] - best_b[1][k] for k in KEYS},
                'u1_static': {k: u1[k] for k in KEYS},
                'd_a1_vs_u1': {k: a1[k] - u1[k] for k in KEYS},
            })
    agg = {}
    for k in KEYS:
        a1v = np.array([r['a1'][k] for r in rows])
        db = np.array([r['d_a1_vs_bestb'][k] for r in rows])
        du = np.array([r['d_a1_vs_u1'][k] for r in rows])
        agg[k] = {'a1_mean': float(a1v.mean()),
                  'a1_std': float(a1v.std(ddof=1)),
                  'paired_diff_vs_bestb_mean': float(db.mean()),
                  'paired_diff_vs_u1_mean': float(du.mean()),
                  'per_fold_diff_bestb': db.tolist()}
    summ = {'arm': 'A1_bce (similarity-row MLP, weighted BCE, PU OFF)',
            'cells': 15, 'rows': rows, 'agg': agg,
            'h1': {'metric': 'multihot_macroF1 (key macroF1)',
                   'delta_required': 0.03}}
    (OUT / 'AGGREGATE_G4.json').write_text(json.dumps(summ, indent=1))
    for k in KEYS:
        a = agg[k]
        print(f"{k:26s} a1={a['a1_mean']:.4f}±{a['a1_std']:.4f}"
              f"  dB={a['paired_diff_vs_bestb_mean']:+.4f}"
              f"  dU1={a['paired_diff_vs_u1_mean']:+.4f}")


if __name__ == '__main__':
    main()
