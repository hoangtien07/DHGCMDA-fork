# TRACK_N_PREREGISTERED_PROTOCOL — đăng ký trước, chưa được phép train

Status: DRAFT awaiting owner approval. Hard gate per council:
no training before protocol + split/metric checks approved.

## 1. Question & hypothesis

Per TRACK_N_RESEARCH_QUESTION: H1 (multi-label type-aware model
beats baselines + U1-static on minority/multi-type strata under
pair-disjoint folds); H2 (ranking persists under entity cold-start);
H3 (existence floor negative-control).

## 2. Locks (all BEFORE any run)

- DATASET_LOCK: strict npz sha 85bb9ea4 (unchanged).
- SPLIT_LOCK: T-pair (reuse verified family, retrospective flag),
  T-miRNA-cold + T-disease-cold (new, indices frozen to JSON+sha
  before training; `default_rng(seed).permutation→array_split`).
- METRIC_LOCK: primary `multihot_macroF1`; secondaries listed in
  TRACK_N_BASELINE_AND_SPLIT_SPEC; k4 forbidden.
- FEATURE_LOCK: every feature/edge train-only; `oracle_feature_canary`
  must pass per fold before training.
- SEED_LOCK: {1234,42,2025}; horizon fixed 650ep for U1-like arms,
  convergence-based for baselines; selection rule = last epoch
  (fixed horizon), no best-val-on-test.
- ARM_LOCK: baselines B1–B5 + exactly one new arm (multi-label
  weighted-BCE + optional PU existence aux head).

## 3. Execution plan (if approved)

1. Build + verify cold-start split indices and canary scripts
   (code-only, no training).
2. Baselines B1–B3 on all three tracks (CPU-cheap).
3. U1 static: reuse frozen predictions for T-pair; new runs only
   for cold-start tracks.
4. New arm: 1 fold × 1 seed sanity → then 5×3 on T-pair →
   cold-start tracks, in that order, only if each stage's
   verification passes.
5. Report: paired fold×seed descriptive stats; negative results
   reported as-is.

## 4. Unlabeled/negative handling

- Type task: labels are multi-hot positive-only; absence of a type
  on a positive pair is treated as negative-for-that-type (standard
  reading; documented as assumption, since HMDD lacks explicit
  type-level negatives).
- Existence (secondary only): PU — unlabeled pairs are candidates,
  not negatives; PR curves on hard-negative strata only; no
  "unknown = negative" claims.

## 5. Uncertainty, effect size, stopping

- Δmacro-F1 ≥ +0.03 over best baseline = target effect for H1.
- Descriptive paired stats only; no fabricated p-values.
- Stop: all preregistered cells done, or H1 falsified on T-pair
  (then cold-start arms may still run for the benchmark report,
  flagged as exploratory).

## 6. External holdout

Candidate external validation: TDRC's HMDD v3.2 variant (713×447)
or v2.0 → v3.2 temporal-like shift — only if entity/type mapping
to the 5-type vocabulary is verifiable; otherwise declared
infeasible and dropped from claims.

## 7. Risks preregistered

- Label noise in HMDD type annotations (unverifiable without
  source curation) → cap on achievable signal; reported as limit.
- Cold-start may collapse all methods near priors → still a
  publishable negative under the benchmark.
- CPU feasibility: baselines hours; U1-like arm ~10-15 min/run
  measured; 5×3×3 tracks ≈ 2-4 h total.
