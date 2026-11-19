# -*- coding: utf-8 -*-
"""WP-A0 metric feasibility: oracle bounds on the strict tensor.
One prediction per pair vs full 5-type multi-hot truth.
Run: .venv-bank1/bin/python metric_feasibility_check.py
"""
import numpy as np

z = np.load('../data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',
            allow_pickle=True)
Y = z['Y'].astype(int)
pairs = np.argwhere(Y.sum(-1) > 0)
Nk = Y.sum((0, 1)).astype(float)
P = len(pairs)
L = int(Y.sum())

# Oracle assignment: each pair independently picks its rarest true
# type -> maximizes sum_k hits_k/N_k (no capacity binds since
# hits_k <= pairs_k = Nk always).
hits = np.zeros(5)
for i, j in pairs:
    ks = np.where(Y[i, j] > 0)[0]
    hits[ks[np.argmin(Nk[ks])]] += 1
oracle_macro_recall = float((hits / Nk).mean())

# Council's rough bound for comparison
rough = 1 - (L - P) / (5 * Nk.max())

res = {
    'pairs': P, 'labels': L, 'Nk': Nk.astype(int).tolist(),
    'oracle_macro_recall_bound': oracle_macro_recall,
    'council_rough_bound': rough,
    'micro_recall_bound': P / L,
    'paper_R': 0.9421,
    'paper_R_exceeds_exact_bound': oracle_macro_recall < 0.9421,
    'paper_R_exceeds_micro_bound': P / L < 0.9421,
}
import json
print(json.dumps(res, indent=1))
assert oracle_macro_recall < 0.9421
assert P / L < 0.9421
print('FEASIBILITY FIXTURE PASS: paper R=0.9421 is impossible under '
      'pooled 5-type multi-hot macro/micro recall on this tensor')
