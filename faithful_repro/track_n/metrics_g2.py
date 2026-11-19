# -*- coding: utf-8 -*-
"""Track N metric recompute — INDEPENDENT path: reads only
frozen scores.npz + taus.json per cell and computes the locked
metrics from row-level arrays. Writes metrics.json per cell and
AGGREGATE.json. This IS the independent recompute (no reliance
on any training-time numbers).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs_g2'


def metrics_for(y, p_norm, taus):
    yhat = (p_norm >= np.asarray(taus)).astype(int)
    T = y.shape[1]
    per = []
    for k in range(T):
        tp = int(((yhat[:, k] == 1) & (y[:, k] == 1)).sum())
        fp = int(((yhat[:, k] == 1) & (y[:, k] == 0)).sum())
        fn = int(((yhat[:, k] == 0) & (y[:, k] == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per.append({'P': prec, 'R': rec, 'F1': f1,
                    'support': int((y[:, k] == 1).sum())})
    macro = float(np.mean([p['F1'] for p in per]))
    # micro
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    mp = tp / max(tp + fp, 1)
    mr = tp / max(tp + fn, 1)
    micro = 2 * mp * mr / max(mp + mr, 1e-12)
    # multitype-restricted macro-F1
    multi = y.sum(1) >= 2
    mm = None
    if multi.sum() >= 5:
        ym, pm = y[multi], yhat[multi]
        fs = []
        for k in range(T):
            tp = int(((pm[:, k] == 1) & (ym[:, k] == 1)).sum())
            fp = int(((pm[:, k] == 1) & (ym[:, k] == 0)).sum())
            fn = int(((pm[:, k] == 0) & (ym[:, k] == 1)).sum())
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            fs.append(2 * pr * rc / max(pr + rc, 1e-12))
        mm = float(np.mean(fs))
    # top1 firstpos harmonic
    hits, total = 0.0, 0
    pred1 = p_norm.argmax(1)
    for i in range(len(y)):
        ks = np.flatnonzero(y[i])
        if len(ks):
            total += 1
            if y[i, pred1[i]]:
                hits += 1.0 / len(ks)
    top1 = hits / max(total, 1)
    return {'macroF1': macro, 'microF1': micro,
            'multitype_macroF1': mm,
            'top1_firstpos_harmonic': top1,
            'minority_recall': per[1]['R'],
            'per_type': per,
            'pred_cardinality_mean': float(yhat.sum(1).mean())}


def main():
    agg = {}
    for scores in sorted(OUT.rglob('scores.npz')):
        cdir = scores.parent
        z = np.load(scores)
        y = z['y_test']
        taus = json.loads((cdir / 'taus.json').read_text())
        res = {}
        for k in z.files:
            if k in ('test_pairs', 'y_test'):
                continue
            res[k] = metrics_for(y, z[k], taus[k])
        (cdir / 'metrics.json').write_text(json.dumps(res, indent=1))
        track = cdir.parent.name
        cell = cdir.name
        for name, m in res.items():
            agg.setdefault(track, {}).setdefault(name, []).append(
                {'cell': cell, 'macroF1': m['macroF1'],
                 'microF1': m['microF1'],
                 'multitype_macroF1': m['multitype_macroF1'],
                 'top1': m['top1_firstpos_harmonic'],
                 'minority_recall': m['minority_recall'],
                 'cardinality': m['pred_cardinality_mean']})
    summ = {}
    for track, arms in agg.items():
        summ[track] = {}
        for name, cells in arms.items():
            summ[track][name] = {
                met: {'mean': float(np.nanmean([c[met] for c in cells])),
                      'n': len(cells)}
                for met in ('macroF1', 'microF1',
                            'multitype_macroF1', 'top1',
                            'minority_recall', 'cardinality')}
    (OUT / 'AGGREGATE.json').write_text(json.dumps(summ, indent=1))
    for t, arms in summ.items():
        print('==', t)
        for a, s in sorted(arms.items()):
            print(f"  {a:22s} macroF1={s['macroF1']['mean']:.4f} "
                  f"micro={s['microF1']['mean']:.4f} "
                  f"multi={s['multitype_macroF1']['mean']:.4f} "
                  f"minR={s['minority_recall']['mean']:.4f} "
                  f"card={s['cardinality']['mean']:.2f}")


if __name__ == '__main__':
    main()
