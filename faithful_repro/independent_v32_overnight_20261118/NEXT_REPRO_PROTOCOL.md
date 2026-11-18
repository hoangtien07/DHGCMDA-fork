# NEXT_REPRO_PROTOCOL — proposal, LOCKED before any run

WP2 deliverable. One protocol, one locked metric, one locked dataset.
NOT launched — training fleet stays off until council approval.

## P.1 Objective

Measure the closest achievable faithful-reproduction of the paper's
reported v3.2 headline (Top-1 F1 = 0.8600) under maximal declared
constraints, and bound the residual gap to data/fold/metric artifacts
(G1–G4 in PAPER_REPRO_GAP_REGISTER) rather than model uncertainty.

## P.2 Locked dataset (dataset_lock)

- Tensor: `v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
  (hash-pinned; already verified in colab_package package_manifest).
- Explicitly NOT the author tensor (G1 open) → every run is labeled
  "strict-rebuild interpretation", never "same data as paper".
- Similarity views: the 4-view stack already provenance-verified
  (S_seq NW/GFF3, S_gene FunSim, S_sem MeSH-2019, S_func train-only).
  Frozen; no new views.

## P.3 Locked folds (fold_lock)

- Same permutation as CVtriplet protocol (rng(0) over positives,
  array_split 5) — already pinned and sidecar-hashed; inner-val =
  rng(2000+fold) 10%. No second split family.

## P.4 Locked primary metric (metric_lock)

- Primary: **generalized Top-1 macro-F1 over K=5 types** using the
  corrected metric (proven identical to the author's 4-type metric on
  v2.0, Plan K §3). Single implementation, source-hash pinned.
- Secondary (reported, not optimized): type-macro AUPR, type-macro
  AUROC, per-type F1 table.
- Forbidden: the shipped `compute_top1_metrics` (verified broken for
  K=5 — drops all type-5 rows).

## P.5 Locked arm (single arm, no sweep)

- Model = U1 (cap10, lam_exist absent — pure type task; U1X exist head
  stays OUT of the type-metric arm to keep comparability with the
  type task; the existence arm is already reported).
- Hyperparameters: the pinned cfg from bank1 contract (unchanged).
- Seeds {1234,42,2025}, 650 epochs, same eval cadence.
- Gates identical to REV2: preflight + fixtures + expected manifest +
  artifact-level verification (indices/truth/scores exported per
  replicate, metrics recomputed from arrays).

## P.6 Analysis plan locked in advance

- Report Top-1 F1 mean±std over 15 runs; per-fold table; per-type F1.
- Report the gap to 0.86 ONLY as "paper number vs our strict-rebuild
  interpretation" with the G1–G4 caveat attached to every figure.
- Pre-registered decision rule: if mean Top-1 F1 ∈ [0.55, 0.86] →
  interpretation is "partial reproduction, gap attributable to
  G1–G4"; outside the band → escalate to council before writing
  conclusions.
- No post-hoc metric switching; any additional analysis must come
  through an amendment like B1–B4.

## P.7 What this protocol does NOT do

- Does not close G1/G2/G3 (needs author artifacts; still missing).
- Does not sweep architectures, seeds beyond the locked set, or
  losses — single arm only.
- Does not claim paper-equivalence; it produces the best honest bound
  achievable on the author release + our rebuild.

## P.8 Cost estimate

Same machinery as CVtriplet-U1X: 15 jobs × ~12 min ≈ 3 h wall-clock
at 2 workers on this VM. Artifacts land in a NEW namespace
`type_task_u1/` — U1, U1X and all previous artifacts untouched.

## P.9 Approval gate

This protocol launches only after explicit council/user approval of
P.2–P.6 (dataset_lock, fold_lock, metric_lock, single arm).
