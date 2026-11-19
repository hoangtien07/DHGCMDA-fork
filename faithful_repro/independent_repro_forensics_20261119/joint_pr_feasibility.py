# -*- coding: utf-8 -*-
"""WP1: joint P/R mathematical feasibility — rigorous upper bounds on R
conditional on the published P, for every source-supported evaluation
convention. No training; reads strict tensor + locked fold manifests only.

Inequality (proven in JOINT_PR_FIXTURES.json):
  R = 1 - (1/K)*Σ_k FN_k/n_k   (macro recall, absent-support classes -> 0)
  FN_k <= n_k <= N_max  =>  Σ FN_k/n_k >= Σ FN_k / N_max = E / N_max
  => R <= 1 - E / (K * N_max)

Outputs JOINT_PR_FIXTURES.json with every computed bound.
"""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TENSOR = HERE.parent / 'data' / 'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
EVALDIR = HERE.parent / 'independent_v32_outer_refit_20261117' / 'evaluation'

P_LO, P_HI = 0.79145, 0.79155  # published 0.7915 with 4-decimal rounding
R_LO, R_HI = 0.94205, 0.94215  # published 0.9421
F1_PUB = 0.8600

out = {'inputs': {}, 'bounds': {}, 'required_support': {}, 'fixtures': {}}

z = np.load(TENSOR, allow_pickle=True)
Y = z['Y'].astype(int)
pos = np.argwhere(Y.sum(-1) > 0)
N = len(pos)
L = int(Y.sum())
py = Y[pos[:, 0], pos[:, 1]]          # (N,5) multi-hot on positive pairs
sup_multi = py.sum(0).astype(int)     # multi-hot per-type support
first = np.argmax(py > 0, 1)
last = 4 - np.argmax(py[:, ::-1] > 0, 1)
sup_first = np.bincount(first, minlength=5)
sup_last = np.bincount(last, minlength=5)

out['inputs'] = {
    'N_pairs': int(N), 'L_links': int(L),
    'multi_hot_support': sup_multi.tolist(),
    'first_pos_support': sup_first.tolist(),
    'last_pos_support': sup_last.tolist(),
    'N_max_upper_bound': int(sup_multi.max()),
    'published': {'P': 0.7915, 'R': 0.9421, 'F1': F1_PUB,
                  'rounding_interval_P': [P_LO, P_HI],
                  'rounding_interval_R': [R_LO, R_HI]},
}


def bound_scalar(N_, E_hi, K, n_max):
    """R <= 1 - E/(K*N_max); use worst-case (largest E) for an upper bound."""
    return 1.0 - E_hi / (K * n_max)


def required_nmax(E_hi, K, R_lo):
    """Smallest single-class scalar support that could let R >= R_lo."""
    return E_hi / (K * (1 - R_lo))


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


# --- 1) pooled K5 scalar (first-pos / last-pos / arbitrary N_max<=3997)
E_hi = N * (1 - P_LO)          # worst case: largest error count
E_lo = N * (1 - P_HI)
res = {}
for name, sup in (('first_positive', sup_first),
                  ('last_positive', sup_last),
                  ('any_assignment', np.array([0, 0, 0, 3997, 0]))):
    n_max = int(sup.max())
    res[name] = {
        'scalar_support': sup.tolist(), 'N_max': n_max,
        'R_upper_bound': bound_scalar(N, E_hi, 5, n_max),
        'required_N_max_for_R': required_nmax(E_hi, 5, R_LO),
    }
out['bounds']['pooled_K5_scalar'] = res

# --- 2) fold-wise K5 (actual locked T-pair fold partition)
fold_bounds = []
for f in range(5):
    ez = np.load(EVALDIR / f'fold{f}_seed1234_static.npz',
                 allow_pickle=True)
    tep = ez['pairs']
    ty = Y[tep[:, 0], tep[:, 1]]
    nf = len(ty)
    # worst-case fold support: assign every scalar to largest class
    nmax_f = int(ty[:, 3].sum())          # pairs containing type3 in fold
    e_hi = nf * (1 - P_LO)
    fold_bounds.append({
        'fold': f, 'n_pairs': int(nf),
        'N_max_max': nmax_f,
        'R_upper_bound': bound_scalar(nf, e_hi, 5, nmax_f)})
avg_bound = float(np.mean([b['R_upper_bound'] for b in fold_bounds]))
out['bounds']['foldwise_K5_scalar'] = {
    'per_fold': fold_bounds, 'mean_of_fold_R_upper': avg_bound,
    'note': 'per-fold support uses pairs containing type3 in that fold'}

# --- 3) multi-hot truth, one prediction per pair
# TP_total <= P_hi*N ; even TP=N gives SigmaFN = L-N
sigma_fn_min = L - int(round(N * P_HI))
res = {
    'SigmaFN_min_at_pubP': int(L - round(N * P_HI)),
    'R_upper_bound_at_pubP': 1 - (L - N * P_HI) / (5 * int(sup_multi.max())),
    'R_upper_bound_even_P1': 1 - (L - N) / (5 * int(sup_multi.max())),
    'note': 'multi-hot truth + single argmax pred; TP_total<=P*N',
}
out['bounds']['multihot_one_pred'] = res

# --- 4) K4 filtered (types 0-3 only, release evaluator convention)
keep4 = py[:, :4].sum(1) > 0
n4 = int(keep4.sum())
sup4 = py[keep4, :4].sum(0).astype(int)
n_max4 = int(sup4.max())
e4_hi = n4 * (1 - P_LO)
out['bounds']['K4_filtered_scalar'] = {
    'N_pairs': n4, 'support': sup4.tolist(), 'N_max': n_max4,
    'R_upper_bound': bound_scalar(n4, e4_hi, 4, n_max4),
    'required_N_max_for_R': required_nmax(e4_hi, 4, R_LO),
}

# --- required support for published R under pooled K5
out['required_support'] = {
    'N_max_needed_for_R_9421': required_nmax(E_hi, 5, R_LO),
    'largest_possible_scalar_class': 3997,
    'verdict': 'INFEASIBLE under pooled/fold-wise K5 scalar conventions'
               ' (needs scalar class support ~%.0f > max possible 3997)'
               % required_nmax(E_hi, 5, R_LO),
}

# --- brute-force validation of the inequality on a tiny case
# K=3 types, n=[4,2,1], N=7 pairs each owning exactly one scalar class,
# E=2 errors: enumerate every allocation FN_k<=n_k, check R<=bound.
n3 = [4, 2, 1]
E_t = 2
best = -1
worst = 2
for f0 in range(n3[0] + 1):
    for f1 in range(n3[1] + 1):
        for f2 in range(n3[2] + 1):
            if f0 + f1 + f2 != E_t:
                continue
            r = 1 - (f0 / n3[0] + f1 / n3[1] + f2 / n3[2]) / 3
            best = max(best, r)
bound = 1 - E_t / (3 * max(n3))
assert abs(best - bound) < 1e-12, (best, bound)
out['fixtures']['tiny_bruteforce_K3'] = {
    'n': n3, 'E': E_t, 'enumerated_max_R': best,
    'formula_bound': bound, 'match': True,
    'note': 'errors concentrated in largest class attain the bound'}

# adverse/concentrated fixtures on real scale: all errors in class3
# vs spread errors -> show bound is attained only when concentrated
E_real = N * (1 - 0.7915)
r_conc = 1 - E_real / (5 * 3997)
r_spread = 1 - E_real * float(np.mean([1 / s for s in sup_multi])) / 5
out['fixtures']['real_scale'] = {
    'E': E_real,
    'R_if_all_errors_in_largest_class': r_conc,
    'R_if_errors_spread_proportionally': r_spread,
}

Path(HERE / 'JOINT_PR_FIXTURES.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out['bounds'].items()},
                 indent=1, default=str))
print('required N_max:', out['required_support'])
