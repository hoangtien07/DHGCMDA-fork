# -*- coding: utf-8 -*-
"""REV F1.1 — corrected joint P/R feasibility.

Corrections vs v1:
1. Max-achievable-R uses the MINIMUM error count: integer-feasible
   correct predictions C with C/N rounding to 0.7915. For N=8735 the
   only feasible counts are found explicitly (C=6914 -> 0.79153).
2. Fold-wise bound allows ARBITRARY per-fold precision: total error
   budget E may be allocated to any fold (respecting E_f <= n_f), and
   errors are optimally placed in folds with the largest support class.
   A greedy fill (sorted by n_max desc) yields the rigorous upper bound.
3. Per-fold N_max audited as max over ALL classes (fold2 max=797 from
   type4, not type3).
4. Counterexample tests that FAIL under the old equal-P-per-fold code.
5. Aggregation-order analysis: harmonic(0.7915,0.9421)=0.86026 which
   rounds to 0.8603, NOT 0.8600; pooled harmonic over the full
   rounding rectangle never reaches 0.8600.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TENSOR = HERE.parent / 'data' / 'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
EVALDIR = HERE.parent / 'independent_v32_outer_refit_20261117' / 'evaluation'

out = {'integer_feasibility': {}, 'bounds': {}, 'aggregation': {},
       'counterexamples': {}, 'verdict': {}}

z = np.load(TENSOR, allow_pickle=True)
Y = z['Y'].astype(int)
pos = np.argwhere(Y.sum(-1) > 0)
N = len(pos)
L = int(Y.sum())
py = Y[pos[:, 0], pos[:, 1]]
sup_multi = py.sum(0).astype(int)
first = np.argmax(py > 0, 1)
last = 4 - np.argmax(py[:, ::-1] > 0, 1)
sup_first = np.bincount(first, minlength=5)
sup_last = np.bincount(last, minlength=5)

# ---------- 1) integer feasibility of published P ----------
feasible = [c for c in range(N + 1)
            if round(c / N, 4) == 0.7915]
out['integer_feasibility'] = {
    'N': int(N),
    'correct_counts_rounding_to_0_7915': feasible,
    'n_correct': feasible[0] if feasible else None,
    'n_errors_min': int(N - (feasible[-1] if feasible else N)),
    'check_6914': {'value': 6914 / N,
                   'rounds_to': round(6914 / N, 4)},
}
C_feas = feasible[-1]            # most correct => fewest errors
E_min = N - C_feas               # = 1821
E_max = N - feasible[0]          # = 1822 (same set {6914})

# ---------- rigorous scalar bound ----------
def r_bound(K, N_, E, n_max):
    return 1 - E / (K * n_max)

def req_nmax(E, K, r_lo):
    return E / (K * (1 - r_lo))

res = {}
for name, sup in (('first_positive', sup_first),
                  ('last_positive', sup_last),
                  ('any_assignment', sup_multi)):
    res[name] = {'scalar_support': sup.tolist(),
                 'N_max': int(sup.max()),
                 'R_upper_bound_Emin': r_bound(5, N, E_min, sup.max()),
                 'required_N_max': req_nmax(E_min, 5, 0.94205)}
out['bounds']['pooled_K5_scalar'] = res

# ---------- fold-wise: ARBITRARY per-fold precision ----------
fold_info = []
for f in range(5):
    ez = np.load(EVALDIR / f'fold{f}_seed1234_static.npz',
                 allow_pickle=True)
    ty = Y[ez['pairs'][:, 0], ez['pairs'][:, 1]]
    fold_info.append({'fold': f, 'n': int(len(ty)),
                      'support': ty.sum(0).astype(int).tolist(),
                      'n_max': int(ty.sum(0).max())})
# allocate the total error budget to folds with largest n_max,
# respecting E_f <= n_f, to maximize mean-R (rigorous bound)
budget = E_min
alloc = []
for fi in sorted(fold_info, key=lambda x: -x['n_max']):
    e = min(budget, fi['n'])
    alloc.append((fi['fold'], e, fi['n_max']))
    budget -= e
    if budget <= 0:
        break
sigma = sum(e / nm for _, e, nm in alloc)
mean_R_bound = 1 - sigma / 25
out['bounds']['foldwise_K5_arbitrary_P'] = {
    'per_fold_support': fold_info,
    'error_allocation': [{'fold': f_, 'errors': e, 'n_max': nm}
                         for f_, e, nm in alloc],
    'mean_R_upper_bound': mean_R_bound,
    'note': 'errors concentrated into largest-support folds; '
            'respects E_f <= n_f; arbitrary per-fold precision'}

# old (invalid) equal-P assumption for comparison
old_bound = float(np.mean([1 - (fi['n'] * (1 - 0.7915)) /
                           (5 * fi['n_max']) for fi in fold_info]))
out['bounds']['foldwise_old_equalP_WRONG'] = {
    'value': old_bound,
    'why_wrong': 'assumed identical per-fold precision; arbitrary P_f '
                 'lets errors concentrate in the largest-support fold '
                 'and legitimately raise mean R above this value'}

# ---------- multi-hot one-pred ----------
out['bounds']['multihot_one_pred'] = {
    'R_upper_at_pub_P': 1 - (L - C_feas) / (5 * sup_multi.max()),
    'R_upper_even_P1': 1 - (L - N) / (5 * sup_multi.max())}

# ---------- K4 filtered ----------
keep4 = py[:, :4].sum(1) > 0
n4 = int(keep4.sum())
sup4 = py[keep4, :4].sum(0).astype(int)
c4 = round(0.7915 * n4)              # generous: same precision
e4_min = n4 - c4
out['bounds']['K4_filtered_scalar'] = {
    'N_pairs': n4, 'support': sup4.tolist(),
    'R_upper_bound': r_bound(4, n4, e4_min, sup4.max()),
    'required_N_max': req_nmax(e4_min, 4, 0.94205)}

# ---------- WP2: aggregation-order ----------
def harm(p, r):
    return np.asarray(2 * np.asarray(p) * np.asarray(r) /
                      (np.asarray(p) + np.asarray(r)))

P, R = 0.7915, 0.9421
agg = {
    'harmonic_published': float(harm(P, R)),
    'rounds_to': round(float(harm(P, R)), 4),
    'claim': 'harmonic(0.7915,0.9421)=0.86026 -> rounds 0.8603, '
             'NOT the reported 0.8600',
    'harmonic_range_over_rounding_rectangle': [
        float(harm(0.79145, 0.94205)), float(harm(0.79155, 0.94215))],
    'implication': ('pooled harmonic cannot equal 0.8600 for ANY '
                    'unrounded P,R inside the published 4-decimal '
                    'rounding rectangle; F1 must come from a different '
                    'aggregation (e.g. mean-of-fold-F1) or different '
                    'underlying values'),
}
# mean-of-fold-F1 feasibility: 5 equal-size folds; one fold gets
# (P1,R1), the other four share (P2,R2) so that means match published.
# Solve for the single parameter s = spread.
best = None
for s in np.linspace(0.001, 0.6, 60000):
    p1, r1 = P - 4 * s, R - 4 * s
    p2, r2 = P + s, R + s
    if r2 > 1 or p1 <= 0:
        break
    f1 = (harm(p1, r1) + 4 * harm(p2, r2)) / 5
    if best is None or abs(f1 - 0.86) < abs(best[0] - 0.86):
        best = (f1, s, p1, r1, p2, r2)
f1v, s, p1, r1, p2, r2 = best
agg['mean_of_fold_F1_feasible'] = {
    'exists': bool(abs(f1v - 0.86) < 5e-4),
    'solution': {'foldA_P': p1, 'foldA_R': r1,
                 'foldsB_P': p2, 'foldsB_R': r2,
                 'mean_P': (p1 + 4 * p2) / 5,
                 'mean_R': (r1 + 4 * r2) / 5,
                 'mean_F1': f1v},
    'note': 'existence construction ONLY — no claim that author folds '
            'had these values; proves 0.8600 is consistent with '
            'mean-of-fold-F1 aggregation'}
out['aggregation'] = agg

# ---------- counterexamples (fail vs old implementation) ----------
ce = {}
# CE1: two folds, unequal support; errors to largest fold raises R
# old code bound (equal P per fold) vs true max with arbitrary P_f
n1, n2 = 100, 100
nm1, nm2 = 10, 50
E_t = 20
old = (1 - (E_t / 2) / (5 * nm1) + 1 - (E_t / 2) / (5 * nm2)) / 2
new = 1 - (E_t / nm2) / (5 * 2)      # all errors into fold2 (feasible)
ce['foldwise_concentration'] = {
    'old_equal_P_bound': old, 'true_max_arbitrary_P': new,
    'old_was_not_a_valid_upper_bound': bool(new > old)}
# CE2: rounding — maximum achievable R must use C=6914 (feasible),
# not fractional E from P_LO
ce['rounding_direction'] = {
    'E_min_integer': E_min,
    'bound_with_E_min': r_bound(5, N, E_min, 3997),
    'bound_with_E_max': r_bound(5, N, E_max, 3997),
    'note': 'reporting the maximum achievable R requires E_min'}
out['counterexamples'] = ce

req = req_nmax(E_min, 5, 0.94205)
out['verdict'] = {
    'required_single_class_support': req,
    'largest_possible': 3997,
    'infeasible_under_source_supported_conventions': bool(req > 3997),
    'conditional': 'UNRESOLVED_EVALUATION_CONVENTION'}

Path(HERE / 'JOINT_PR_COUNTEREXAMPLES.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out['bounds'], indent=1, default=str)[:2600])
print(json.dumps(out['aggregation'], indent=1, default=str)[:1200])
print('verdict:', out['verdict'])
