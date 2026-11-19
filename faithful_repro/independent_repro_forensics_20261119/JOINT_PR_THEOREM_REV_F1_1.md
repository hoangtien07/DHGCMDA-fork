# JOINT_PR_THEOREM_REV_F1_1 — corrected proof (council REV-F1.1)

Corrections applied vs v1:
- Max achievable R now uses the **minimum integer error count** (feasible correct count C=6914 → E=1821; 6914/8735=0.79153 rounds to 0.7915 — the ONLY feasible integer, verified by enumeration of all C∈[0,8735]).
- Fold-wise bound allows **arbitrary per-fold precision** (previous version wrongly assumed equal P per fold — see counterexample: old bound 0.88 vs true max 0.96 ⇒ old bound was NOT a valid upper bound).
- Per-fold N_max audited as max over **all classes** (fold2: 797 via type4, not type3=784).
- Aggregation-order analyzed separately (AGGREGATION_ORDER_ANALYSIS.md).

## Corrected rigorous bounds (E_min=1821)

| convention | K | N | N_max | R upper bound | feasible for R≥0.94205? |
|---|---:|---:|---:|---:|---|
| pooled K5 scalar (any truth assignment) | 5 | 8735 | 3997 | 0.9089 | NO — needs class ≥6285 |
| fold-wise mean, K5 scalar, **arbitrary per-fold P** | 5 | 1747/fold | 832 (fold3) | **0.9123** | NO |
| first-positive scalar | 5 | 8735 | 2903 | 0.8745 | NO |
| last-positive scalar | 5 | 8735 | 3900 | 0.9066 | NO |
| multi-hot truth, 1 pred/pair | 5 | 8735 | 3997 | 0.7581 (cap 0.849 at P=1) | NO |
| K4 filtered scalar (N4=6342) | 4 | 6342 | 3997 | 0.9173 | NO — needs class ≥5703 |

Fold-wise bound derivation: mean R = 1 − (1/25)Σ_f E_f/n_max,f. To maximize,
minimize Σ E_f/n_max,f given ΣE_f=1821, E_f≤n_f → greedy fill into largest-
support folds: fold3 gets all 1747 (n_max 832), remainder 74 into fold1 (800)
→ Σ=2.1916 → mean R ≤ 1 − 2.1916/25 = 0.9123. Per-fold support audited on the
actual locked fold manifests (fold2 n_max=797 via type4 — corrects v1's 784).

**Tightness caveat (documentation errata):** the 0.91231 bound is computed on
the *audited, locked fold partition* of this tensor under the greedy optimal
error allocation. It is a **valid relaxed upper bound** — i.e., no allocation
of 1821 errors over these folds yields mean R above it — but it is **not a
proven tight maximum**: it does not certify that 0.9123 is itself achievable,
only that nothing higher is. Conclusion (R≥0.94205 infeasible) is unaffected
because the bound sits far below the target.

## What is PROVEN vs what remains a convention hypothesis

**Proven conditional results:** under EVERY convention listed above — the only
ones with any source support (released evaluator semantics, fold CV, multi-hot,
K4 filtering) — the published joint (P=0.7915, R=0.9421) is mathematically
unreachable on this tensor. The required single-class scalar support (~6285)
exceeds the largest possible (3997).

**Untested conventions (HYPOTHESIS only, no source support):** different label
population/subset, per-(pair,type) prediction unit, micro recall, mean-of-fold
aggregation with non-uniform denominators, or a scalar-truth convention with
class supports different from this tensor's.

## Aggregation (WP2 summary)
harmonic(0.7915, 0.9421) = 0.86026 → **0.8603**, not the reported 0.8600.
No unrounded P,R inside the published rounding rectangle yields pooled
harmonic 0.8600 (range [0.86021, 0.86031]). F1=0.8600 is therefore consistent
only with a different aggregation order (e.g., mean-of-fold-F1 — corrected
existence witness in AGGREGATION_ORDER_ANALYSIS.md, mean F1 = 0.8600 exactly,
NOT an author claim) or different underlying values.

## Verdict
`UNRESOLVED_EVALUATION_CONVENTION` stands, now on a corrected, rounding- and
integer-aware proof. Not an accusation: the missing artifact could be a
different label population or prediction unit.
