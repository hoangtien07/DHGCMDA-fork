# NEW_EXPERIMENT_NECESSITY — decision table

Does the REV3 question require new training? Audited against every
existing evidence line (EVIDENCE_INVENTORY.csv).

## Decision table

| Needed element | Available in E08 outer refit? | Verdict |
|---|---|---|
| Pinned strict tensor (unleaked) | YES — bundle_audit + data_audit elementwise-pass | |
| PAIR_CVTYPE fold lock, pair-disjoint | YES — verified by test_real_outer_refit_pair_disjoint (all 30 files) | |
| U1 static + dynamic, 5 folds × 3 seeds | YES — 30 trained runs, logits exported | |
| Pinned metric family | NO code-side, but YES by rescore — row-level data lets metrics be computed post-hoc to exact contract | |
| Count priors on same cells | YES — fold-count priors recomputed, verified vs reported | |
| Blind/confirmatory status | NO — retrospective (post-dev selection of 650ep horizon, fold0 viewed early) | limitation declared |
| Author off_diag_minmax features | NO — 4-view strict bundle only | limitation declared |
| "Faithful to author code" claim | NO — adaptations declared | out of scope |

## Verdict: ALREADY_ANSWERED

Every element of the REV3 question is resolvable from stored
row-level artifacts. REV3 = report wording + metric contracts, not
new compute. Answer (WP4 rescore): U1 harmonic ~0.46–0.49 vs best
prior 0.448 under identical folds/seeds.

## What WOULD require new training (explicitly flagged, NOT done)

1. Any claim about the unmeasured `no_dynamic` baseline or
   minmax-variant arms — never trained.
2. A confirmatory (blind) replication on a fresh fold family with a
   selection rule fixed before development — impossible: every
   stored artifact is post-dev.
3. The single remaining honest-but-adapted trained arm described in
   the program's WP5 (locked arm, 5×3, 650ep, CPU, 2 workers) is
   gated on Gate E — see WP6/GATE_REPORT.
