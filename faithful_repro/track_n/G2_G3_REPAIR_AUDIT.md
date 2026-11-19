# G2_G3_REPAIR_AUDIT — WP1/WP2 repairs + verification

Base: `679239b` → this commit. Council verdict: CHANGES_REQUIRED_BEFORE_G2.

## WP1 repairs

| Defect | Fix | Fixture proving it |
|---|---|---|
| B0 was type-agnostic broadcast | B0 delegates to verified `_prior.fit_counts/predict_counts` (5-channel, fallback-to-global) | `t_prior_equivalence_archived` — asserts per-type channels differ & archived npz shape/keys |
| `pair_features` indexed S[miRNA, disease] on square matrices | per-type count features (5+5) + sim-weighted type counts using S_mi[i,:]·c_m and S_dis[j,:]·c_d | `t_sim_indexing` |
| No LOPO — train features leaked own labels | `lopo_pairs` subtracts own label vector | `t_lopo_removes_own_label` |
| `arr.ptp()` NumPy 2.x | `np.ptp(col)` | `t_logistic_ptp_and_learns` |
| B2 unchecked normalization | per-type propagation, normalizer = train-carrier similarity mass only | `t_labelprop_pertype_trainonly` |

5/5 fixtures PASS (all fail under the old code).

## WP2 provenance gate

Separated SPLIT_PASS / FEATURE_PROVENANCE_PASS /
INDUCTIVE_CHECK_PASS / EXTERNAL_VIEWS_PASS in
`PROVENANCE_CANARY_REPORT.json`:

- SPLIT_PASS: entity + pair disjointness re-derived from tensor
  (not trusting manifests) — PASS both cold-start tracks.
- FEATURE_PROVENANCE_PASS: production builders (masked-Y → A →
  GIP_mi/GIP_dis) rerun after perturbing ONLY held-out labels →
  all train-derived matrices byte-identical, all 30 cold-start
  fold-cells — PASS.
- INDUCTIVE_CHECK_PASS: held-out entities carry zero train
  associations — PASS.
- EXTERNAL_VIEWS_PASS: sim_m1/sim_m2/sim_d1/sim_d2 ≠ any
  A-derived GIP — PASS (independent check per view).
- pair_track FEATURE_PROVENANCE: same mechanism, deferred flag
  (folds are archived-verified).

No NO-GO track.

## Known honest caveats (recorded, not fixed)

- `B0_global` / `B0_disease` τ-selection degenerates on some
  tracks (card 0 or 5) — constant/near-constant scores; kept as
  honest baseline behavior, not tuned away.
- A1 deviation: similarity-row inputs instead of U1 embeddings —
  producing embeddings is itself a training run; flagged for
  fleet decision.
- Cold-start minority support 23–251/fold → per-fold estimates
  noisy; descriptive only (preregistered).
