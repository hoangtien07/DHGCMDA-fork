# 00_SCOPE_AND_CLAIM_BOUNDARY — Track P5 (serves as CLAIM_BOUNDARY.md)

Track: `TRACK_P5_SPEC_AND_PRETRAIN_AUDIT` — P5.0, 2026-11-20.
Base commit: `3fb1998dede6c2efee14d8c4a7cab89ff4f5be79` on branch
`research/track-p5-pretrain`.

## Research question

Can a 5-type (K5) DHGCMDA be built in which every computation is
independently verifiable, consistent with the equations the paper
publishes, and evaluated under a label-leakage-free protocol — and,
only once training is later authorized, does it actually beat strong
baselines on the same protocol?

## Explicit non-goals

- Reaching F1 = 0.8600 (the published v3.2 figure is not an
  optimization target; Track R proved the joint P/R pair unreachable
  under every source-supported convention — conditional result).
- Inferring the authors' evaluator from a single number.
- Claiming the self-built code is the authors' code.
- Re-shaping scoring to look like the paper.
- Re-opening or revising closed Track R conclusions.

## Boundary vs prior tracks

| Track | Status | Relation to P5 |
|---|---|---|
| Track R (reproduction + forensics) | `TRACK_R_FORENSICS_CLOSED_CONDITIONALLY_UNRESOLVED` — closed | Evidence source only; artifacts immutable; numbers never mixed into P5 claims |
| REV4 / REV4.1 | accepted | Metric contracts and errata inherited as constraints |
| Track N | G4 complete, H1 rejected (−0.014 vs best baseline) | Code/metric contracts reusable only where fold/feature/metric semantics are identical |
| Track P5 | open, pre-training audit | New, independently named study — "independent research based on the authors' method", not reproduction of author results |

## Claim hygiene (mandatory labels)

`[PAPER_EXPLICIT]` equation/text · `[AUTHOR_RELEASE]` release code
· `[PRIOR_PAPER]` TDRC/SPLDHyperAWNTF/HGCLAMIR · `[INDEPENDENT_ASSUMPTION]`
· `[MEASURED]` own fixture measurement · `[UNKNOWN]`.
Tier (d/e) content must never be promoted silently into an author claim.

## Profiles

- `P5_BENCHMARK_UNKNOWN_PROVENANCE`: strict tensor + bundled sim views
  (UNKNOWN_PROVENANCE) — exploratory only, no leakage-free claim.
- `P5_SPLIT_AWARE_EXTERNAL`: independently reconstructed four views,
  each audited; "leakage-controlled" only after the provenance gate
  passes, else `EXPLORATORY` / `BLOCKED_FEATURE_PROVENANCE`.
- `P5_RELEASE_K4_CONTROL`: v2.0 K4 release inputs — sanity/control only.

## Locked evaluation namespace (declared now, before any training)

- Primary task: known-positive pair → five-type multi-hot prediction.
- Primary metric: `MULTIHOT_MACRO_F1` (5 classes; thresholds fit on
  inner-validation only).
- Secondary: micro-F1, per-type P/R/F1, minority recall, multi-type
  strata.
- Auxiliary top-1: `TOP1_ANYHIT_HARMONIC_K5`,
  `TOP1_SCALAR_FIRSTPOS_HARMONIC_K5` — separately named conventions,
  never merged and never equated to the authors' evaluator or 0.8600.
- CV_TRIPLET (separate protocol): AUC, `average_precision` (AP) and
  trapezoidal AUPR reported distinctly; F1 with validation-only
  threshold; negative draws and seeds frozen in advance.

## Current authority

Static analysis, specification, portable unit tests, synthetic
forward/backward fixtures, documentation, non-training code drafting.
Forbidden now: optimizer steps on HMDD real data, one-fold pilots,
fleets, test-based selection, author contact, GitHub issues, modifying
Track R/Track N/historical artifacts.
