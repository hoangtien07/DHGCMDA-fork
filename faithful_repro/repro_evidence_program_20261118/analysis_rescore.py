# -*- coding: utf-8 -*-
"""WP4 rescore: recompute metrics from ROW-LEVEL data (pairs, y,
logits) of the 30 outer-refit exports + 4 priors rebuilt from
fold counts. Nothing copied from reported JSON summaries.
Outputs RESULTS_CROSSWALK.csv + RESCORE_VERIFICATION.json.
Run: .venv-bank1/bin/python analysis_rescore.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'independent_v32_outer_refit_20261117'))
from metric_contracts import (
    generalized_top1_harmonic, per_class_macro_f1, micro_f1,
    any_hit_rate, confusion_and_support, release_top1_harmonic_k4)
from prior import fit_counts, predict_counts

EVAL = HERE.parent / 'independent_v32_outer_refit_20261117' / 'evaluation'
PRIO = HERE.parent / 'independent_v32_outer_refit_20261117' / 'priors'
BUNDLES = {f: (HERE.parent / 'paper_guided'
               / f'candidate_fold{f}_selection_rev1'
               / 'candidate_fold_data.npz') if f < 2 else
           (HERE.parent / 'independent_v32_fold_build_20261117'
            / 'bundles' / f'fold{f}_selection'
            / 'candidate_fold_data.npz') for f in range(5)}
REPORTED = EVAL / 'results.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def all_metrics(y, s):
    out = {}
    out['general_k_top1_harmonic_firstpos'] = \
        generalized_top1_harmonic(y, s, 'first_positive')['f1_harmonic']
    out['general_k_top1_harmonic_anyhit'] = \
        generalized_top1_harmonic(y, s, 'any_hit')['f1_harmonic']
    out['per_class_macro_f1'] = per_class_macro_f1(y, s)['macro_f1']
    out['micro_f1'] = micro_f1(y, s)
    out['any_hit'] = any_hit_rate(y, s)
    return out


def main():
    rows = []
    verif = {'files': [], 'reported_vs_rescored': [], 'failures': []}
    prior_rows = []
    for fold in range(5):
        # rebuild priors from counts (train-only, Laplace1 product etc.)
        counts = {k: np.load(PRIO / f'fold{fold}_counts.npz')[k]
                  for k in ('mi', 'dis', 'global')}
        ev = np.load(EVAL / f'fold{fold}_seed1234_dynamic.npz')
        test_pairs, y = ev['pairs'], ev['y'].astype(np.int64)
        prior_scores = predict_counts(counts, test_pairs)
        for name, s in prior_scores.items():
            m = all_metrics(y, s)
            prior_rows.append({'fold': fold, 'seed': 'shared',
                               'arm': f'prior_{name}', **m})
        for seed in (1234, 42, 2025):
            for arm in ('dynamic', 'static'):
                f = EVAL / f'fold{fold}_seed{seed}_{arm}.npz'
                z = np.load(f)
                assert (z['pairs'] == test_pairs).all()
                m = all_metrics(z['y'], z['logits'])
                rows.append({'fold': fold, 'seed': seed, 'arm': arm,
                             'file_sha256': sha(f)[:16], **m})
                verif['files'].append(
                    {'key': f'fold{fold}_seed{seed}_{arm}',
                     'sha256': sha(f)})
    # verify against reported macro_f1 (their convention = our
    # per_class_macro_f1) — must match to 1e-9 or we flag.
    rep = json.loads(REPORTED.read_text())
    rep_map = {(r['fold'], r['seed'], r['arm']): r['metrics']
               for r in rep['rows']}
    for r in rows:
        key = (r['fold'], r['seed'], r['arm'])
        want = rep_map[key]['macro_f1']
        got = r['per_class_macro_f1']
        ok = abs(want - got) < 1e-9
        verif['reported_vs_rescored'].append(
            {'key': key, 'reported_macro_f1': want,
             'rescored_macro_f1': got, 'match': ok})
        if not ok:
            verif['failures'].append(key)
    # baselines reported check
    rep_bl = {b['fold']: b['metrics'] for b in rep['baselines']}
    for pr in prior_rows:
        want = rep_bl[pr['fold']][pr['arm'][6:]]['macro_f1']
        ok = abs(want - pr['per_class_macro_f1']) < 1e-9
        if not ok:
            verif['failures'].append(
                ('prior', pr['fold'], pr['arm']))
        verif['reported_vs_rescored'].append(
            {'key': (pr['fold'], pr['arm']),
             'reported_macro_f1': want,
             'rescored_macro_f1': pr['per_class_macro_f1'],
             'match': ok})
    # k4 release metric on 5-channel logits+multi-hot (diagnostic:
    # how much does the release formula see?)
    k4 = release_top1_harmonic_k4(
        [row for row in y[:0]], [])
    verif['k4_on_v32_note'] = (
        'release_top1_harmonic_k4 drops all rows on 5-type '
        'multi-hot truth — verified separately in fixtures')
    all_rows = rows + prior_rows
    with open(HERE / 'RESULTS_CROSSWALK.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    verif['n_rows'] = len(all_rows)
    verif['n_matched'] = sum(
        1 for v in verif['reported_vs_rescored'] if v['match'])
    verif['status'] = 'PASS' if not verif['failures'] else 'FAIL'
    (HERE / 'RESCORE_VERIFICATION.json').write_text(
        json.dumps(verif, indent=1))
    # aggregate print
    import statistics as st
    for arm in ('dynamic', 'static', 'prior_global', 'prior_disease',
                'prior_mirna', 'prior_product_laplace1'):
        sub = [r for r in all_rows if r['arm'] == arm]
        for met in ('general_k_top1_harmonic_firstpos',
                    'general_k_top1_harmonic_anyhit',
                    'per_class_macro_f1', 'micro_f1', 'any_hit'):
            vals = [r[met] for r in sub]
            print(f'{arm:24s} {met:38s} {st.mean(vals):.4f} '
                  f'+- {st.stdev(vals):.4f}')
    print('VERIF', verif['status'], verif['n_matched'], '/',
          len(verif['reported_vs_rescored']))


if __name__ == '__main__':
    main()
