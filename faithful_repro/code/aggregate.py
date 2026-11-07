import json, glob, os, numpy as np
R = os.path.expanduser(os.environ.get('RES', '~/dhg_runs/faithful/results'))
PAPER = {('v2.0', 'type'): dict(top1_precision=0.5842, top1_recall=0.6341, top1_f1=0.5970),
         ('v2.0', 'triplet'): dict(aupr=0.9738, auc=0.9669, f1=0.9278),
         ('v3.2', 'type'): dict(top1_precision=0.7915, top1_recall=0.9421, top1_f1=0.8600),
         ('v3.2', 'triplet'): dict(aupr=0.9271, auc=0.9181, f1=0.8674)}
rows = {}
for f in sorted(glob.glob(R + '/*.json')):
    j = json.load(open(f)); a = j['args']
    if a.get('only_fold', -1) >= 0 or len(j['folds']) < a['folds']:
        continue
    key = (a['data'], a['protocol'], a['ablation'], a.get('recon_reduction', 'sum'), a.get('diag_leak', 0))
    rows.setdefault(key, []).append(j)
out = []
for key in sorted(rows):
    data, prot, abl, red, leak = key
    ks = ['top1_precision', 'top1_recall', 'top1_f1'] if prot == 'type' else ['aupr', 'auc', 'f1']
    ks += [k for k in (['top1_recall_micro', 'top1_f1_micro'] if prot == 'type' else ['aupr_draw0', 'auc_draw0', 'f1_draw0', 'f1_valthr', 'f1_valthr_draw0']) if all(k in r for j in rows[key] for r in j['folds'])]
    seeds = [j['args']['seed'] for j in rows[key]]
    per_seed = {k: [np.mean([r[k] for r in j['folds']]) for j in rows[key]] for k in ks}
    allf = {k: [r[k] for j in rows[key] for r in j['folds']] for k in ks}
    pap = PAPER[('v2.0' if data == 'v2.0' else 'v3.2', prot)]
    cells = [f"{k}={np.mean(per_seed[k]):.4f}±{np.std(per_seed[k]):.4f}" + (f" (paper {pap[k]:.4f}, Δ {np.mean(per_seed[k]) - pap[k]:+.4f})" if k in pap else "") for k in ks]
    out.append(f"| {data} | {prot} | {abl} | {red} | {'LEAK' if leak else '-'} | {seeds} | " + ' | '.join(cells) + ' |')
    for j in rows[key]:
        out.append(f"|  |  |  |  |  | seed {j['args']['seed']} folds | " + ' | '.join(' / '.join(f"{r[k]:.4f}" for r in j['folds']) for k in ks) + ' |')
print('| data | protocol | ablation | recon | leak | seeds | metric1 | metric2 | metric3 |\n|---|---|---|---|---|---|---|---|---|')
print('\n'.join(out))
