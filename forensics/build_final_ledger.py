"""build_final_ledger.py — merge all experiment JSONs → FROZEN leaderboard.

Sources:
  m2_baseline_results.json + _seed{1,2}      (tournament + learned seeds)
  golden_spld_eval.json                      (SPLD static-MISIM legacy)
  golden_spld_eval_fold.json                 (SPLD fold-MISIM)
  golden_spld_eval_none.json                 (SPLD MGSM only)
  m3_rerun_results.json                      (DHGCMDA encoder honest)
  final_suite_results.json                   (vote ablations + rep-CV seeds)
Writes forensics/FROZEN_LEADERBOARD.json + FROZEN_LEADERBOARD.md.
"""
import json, os
import numpy as np

F = '/home/ubuntu/repos/DHGCMDA-fork/forensics'
load = lambda n: json.load(open(os.path.join(F, n)))

KEYS = ['legacy_top1_precision', 'legacy_micro_recall', 'legacy_macro_recall',
        'legacy_f1', 'macro_AUPR', 'micro_AUPR', 'macro_AUC', 'micro_AUC',
        'Hit@1', 'Recall@3', 'micro_F1@0.5']


def agg_from_folds(folds):
    return {k: float(np.nanmean([f[k] for f in folds])) for k in KEYS
            if k in folds[0]}


def std_f1(folds):
    v = [f['legacy_f1'] for f in folds]
    return float(np.std(v))


rows = {}
# tournament (seed0)
m2 = load('m2_baseline_results.json')
for name, res in m2.items():
    rows[name] = {'track': 'LEAKY' if 'misim' in name else 'honest',
                  'seed0': res['aggregate'], 'f1_per_fold': [f['legacy_f1'] for f in res['folds']]}
# learned-model seeds 1,2
for s in (1, 2):
    fn = f'm2_baseline_results_seed{s}.json'
    p = os.path.join(F, fn)
    if os.path.exists(p):
        d = load(fn)
        for name, res in d.items():
            rows.setdefault(name, {'track': 'honest'})[f'seed{s}'] = res['aggregate']
# SPLD golden variants
spl = load('golden_spld_eval.json')
rows['SPLD_staticMISIM'] = {'track': 'LEGACY(leaky-lite)', 'seed0': agg_from_folds(spl['folds']),
                            'f1_per_fold': [f['legacy_f1'] for f in spl['folds']]}
for tag, fn in (('SPLD_foldMISIM', 'golden_spld_eval_fold.json'),
                ('SPLD_noMISIM', 'golden_spld_eval_none.json')):
    p = os.path.join(F, fn)
    if os.path.exists(p):
        d = load(fn)
        rows[tag] = {'track': 'LEGACY(corrected)', 'seed0': agg_from_folds(d['folds']),
                     'f1_per_fold': [f['legacy_f1'] for f in d['folds']]}
# encoder (aggregate block only; file truncated mid-dump but aggregate intact)
m3p = os.path.join(F, 'v32_spldeval_dssm_gip.json')
if os.path.exists(m3p):
    import re
    txt = open(m3p).read()
    m = re.search(r'"aggregate":\s*\{([^}]*)\}', txt)
    if m:
        agg = {k: float(v) for k, v in
               re.findall(r'"([^"]+)":\s*([0-9.eE+-]+)', m.group(1))}
        rows['DHGCMDA_encoder'] = {'track': 'honest', 'seed0': agg}
# final suite ablations
su = load('final_suite_results.json')
for name, v in su.items():
    rname = f'vote_{name}'
    rows.setdefault(rname, {'track': 'LEAKY' if name == 'misim_dssm' else 'honest'})
    for seed, res in v.items():
        rows[rname][seed] = res['aggregate']
        if seed == 'seed0':
            rows[rname]['f1_per_fold'] = [f['legacy_f1'] for f in res['folds']]

json.dump(rows, open(os.path.join(F, 'FROZEN_LEADERBOARD.json'), 'w'), indent=2)

lines = ['# FROZEN LEADERBOARD — MDAv3.2-3, golden SPLD pair-fold protocol\n',
         'Frozen at final-validation-suite completion. Seeds 0/1/2 for learned',
         'models; deterministic methods report seed0 only (rep-CV seeds for',
         'vote_gip_dssm shown as secondary robustness).\n',
         '| Model | Track | P | micro-R | macro-R | **legacy-F1** | mAUPR | seeds |',
         '|---|---|---:|---:|---:|---:|---:|---|']
order = sorted(rows, key=lambda r: -rows[r]['seed0']['legacy_f1'])
for r in order:
    a = rows[r]['seed0']
    seeds = '/'.join(f"{s}:{rows[r][s]['legacy_f1']:.3f}" for s in
                     sorted(k for k in rows[r] if k.startswith('seed')))
    lines.append(f"| {r} | {rows[r]['track']} | {a['legacy_top1_precision']:.4f} | "
                 f"{a['legacy_micro_recall']:.4f} | {a['legacy_macro_recall']:.4f} | "
                 f"**{a['legacy_f1']:.4f}** | {a.get('macro_AUPR', float('nan')):.4f} | {seeds} |")
open(os.path.join(F, 'FROZEN_LEADERBOARD.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
