# G2_CORRECTION_AUDIT — inner-val leakage repair (WP1/WP2)

## Defect found (old runs_g2/)
`Y_train` used for baseline fitting included the **inner-validation pairs** (10% of train). Consequences: iv-scores/features saw their own labels; B1 logistic trained on leaked labels; B2 propagation/normalization saw iv labels; sim-weighted count features used counts that already included iv pairs.

## Corrections applied
1. `Y_fit = masked(Y, outer_train_pairs MINUS inner_val_pairs)` — strictly excludes iv pairs (run_g2c.py).
2. B1 features/iv-scoring/B2 propagation/score normalization all use `Y_fit` only.
3. **Per-pair LOPO**: for each training pair, its own label vector is subtracted from entity counts — `c_m[i] - y_own` for direct features and `S[i,i]·y_own`, `S[j,j]·y_own` corrections for similarity-weighted features (baselines.py `pair_features`, `y_own` argument). Not a global per-entity subtraction.
4. Perturbation fixtures prove changing any iv label does not change fit outputs (`t_inner_val_perturbation_no_leak`, 6/6 contracts).
5. Metric `top1_firstpos_harmonic` reverted to the locked general-K first-positive precision / mean-per-class-recall harmonic (counterexample fixtures, 5/5 pass).
6. Provenance gates test the **exact** builders of run_g2c.py and a1_arm.py at split/inner-val/feature levels — ALL_PASS on all 3 tracks (`PROVENANCE_FINAL_GATE.json`).

## Namespace handling
- Old `runs_g2/` kept immutable, marked `SUPERSEDED_UNVERIFIED.txt` — not to be cited.
- Corrected results live in `runs_g2c/` (45 cells frozen with scores, labels, train/iv/test indices, τ selection data, taus.json, source hashes via MANIFEST).

## Residual notes
- sklearn ConvergenceWarnings on some B1 cells (logistic max_iter reached); flagged, not blocking — thresholds picked on inner-val regardless.
- Metric `macroF1` key = locked `multihot_macroF1` primary metric.
