# -*- coding: utf-8 -*-
"""WP-A2 metric bridge: rescore frozen row-level outputs under
SEPARATELY NAMED truth/metric conventions. No primary-metric
replacement — every column carries its convention name.
Output: METRIC_BRIDGE.json (+ console table for the report).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metric_contracts import (generalized_top1_harmonic,
                              per_class_macro_f1, micro_f1,
                              any_hit_rate, confusion_and_support)

EVAL = HERE.parent / 'independent_v32_outer_refit_20261117' / 'evaluation'
PRIO = HERE.parent / 'independent_v32_outer_refit_20261117' / 'priors'
sys.path.insert(0, str(HERE.parent /
                       'independent_v32_outer_refit_20261117'))
from prior import predict_counts  # noqa: E402


def conv_metrics(y, s):
    """All named conventions on the same (y multi-hot, s scores)."""
    pred = np.asarray(s).argmax(-1)
    truth_first = np.argmax(y > 0, -1)          # first positive
    truth_last = 4 - np.argmax((y > 0)[:, ::-1], -1)  # last positive
    m = {}
    for name, conv in (('first_positive', 'first_positive'),
                       ('last_positive', 'any_hit')):
        pass
    def harm(real, pred_):
        P = float((real == pred_).mean())
        R = float(np.mean([np.mean(pred_[real == k] == k)
                           if (real == k).any() else 0.0
                           for k in range(5)]))
        F = 2 * P * R / (P + R) if P + R else 0.0
        return P, R, F
    m['K5_firstpos_F1'] = \
        generalized_top1_harmonic(y, s, 'first_positive')['f1_harmonic']
    P, R, F = harm(truth_last, pred)
    m['K5_lastpos_P'], m['K5_lastpos_R'], m['K5_lastpos_F1'] = P, R, F
    P, R, F = harm(truth_first, pred)
    m['K5_firstpos_P'], m['K5_firstpos_R'] = P, R
    m['anyhit_F1'] = generalized_top1_harmonic(
        y, s, 'any_hit')['f1_harmonic']
    m['anyhit_rate'] = any_hit_rate(y, s)
    m['multihot_macroF1'] = per_class_macro_f1(y, s)['macro_f1']
    m['multihot_microF1'] = micro_f1(y, s)
    # 4-type filtered convention (release-style: drop type-5 rows
    # and use only channels 0-3)
    keep = ~((y > 0)[:, 4])  # pairs without type-5 label only
    y4, s4 = y[keep][:, :4], s[keep][:, :4]
    if len(y4):
        tf = np.argmax(y4 > 0, -1)
        m['K4filtered_F1'] = harm(tf, s4.argmax(-1))[2]
        m['K4filtered_n'] = int(len(y4))
    # multi-type pair strata
    multi = (y > 0).sum(-1) >= 2
    m['n_multitype_pairs'] = int(multi.sum())
    if multi.any():
        m['multitype_anyhit'] = float(
            ((pred[multi, None] == np.where(y[multi] > 0, 1, 0))
             .any(-1)).mean())
    m['singletype_acc'] = float((pred[~multi] == truth_first[~multi]).mean())
    return m


def main():
    out = {'conventions': {
        'K5_firstpos_*': 'truth=first positive index, pooled P/R/harmonic',
        'K5_lastpos_*': 'truth=last positive index',
        'anyhit_*': 'pair correct if argmax(pred) is any true type',
        'multihot_*': 'full 5-type multi-hot truth',
        'K4filtered_*': 'drop all pairs carrying type-5 label, 4 channels',
    }, 'rows': [], 'oracle': {}}
    # oracle ceilings (same as feasibility check)
    Y = None
    for fold in range(5):
        ev = np.load(EVAL / f'fold{fold}_seed1234_dynamic.npz')
        pairs, y = ev['pairs'], ev['y'].astype(int)
        Nk = y.sum(0).astype(float)
        hits = np.zeros(5)
        for r in y:
            ks = np.where(r > 0)[0]
            hits[ks[np.argmin(Nk[ks])]] += 1
        out['oracle'][f'fold{fold}'] = {
            'macro_recall_bound': float((hits / Nk).mean()),
            'micro_recall_bound': float(len(y) / y.sum()),
            'anyhit_bound': 1.0}
    for fold in range(5):
        counts = {k: np.load(PRIO / f'fold{fold}_counts.npz')[k]
                  for k in ('mi', 'dis', 'global')}
        ev0 = np.load(EVAL / f'fold{fold}_seed1234_dynamic.npz')
        for name, s in predict_counts(counts, ev0['pairs']).items():
            row = conv_metrics(ev0['y'], s)
            row.update(fold=fold, seed='shared', arm='prior_' + name)
            out['rows'].append(row)
        for seed in (1234, 42, 2025):
            for arm in ('dynamic', 'static'):
                z = np.load(EVAL / f'fold{fold}_seed{seed}_{arm}.npz')
                row = conv_metrics(z['y'], z['logits'])
                row.update(fold=fold, seed=seed, arm=arm)
                out['rows'].append(row)
    (HERE / 'METRIC_BRIDGE.json').write_text(json.dumps(out, indent=1))
    import statistics as st
    for arm in ('dynamic', 'static', 'prior_product_laplace1'):
        sub = [r for r in out['rows'] if r['arm'] == arm]
        print(arm, {k: round(st.mean([r[k] for r in sub]), 4)
                    for k in ('K5_firstpos_F1', 'K5_lastpos_F1',
                              'anyhit_F1', 'multihot_macroF1',
                              'multihot_microF1', 'K4filtered_F1')})
    print('rows', len(out['rows']))


if __name__ == '__main__':
    main()
