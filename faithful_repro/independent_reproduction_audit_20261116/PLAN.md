# Independent DHGCMDA reproduction audit

Date: 2026-11-16. Authorized by user: independently execute the reproduction-focused audit.

Goal: distinguish verified equivalence, declared adaptation, and missing evidence before further training.

Scope: read existing inner-train/inner-validation exports only; no checkpoint inference, training, outer-test scoring, pipeline changes, agent messages, publication, or commits. Write only in this audit directory. Execute inline; no delegation is necessary.

Deliverables: `audit.py`, machine-readable results and source inventory, `FIDELITY_MATRIX.csv`, `REPORT.md`, primary-source snapshots, reproducible verification output.

## Predeclared metric diagnostics (before inspecting export scores)

All diagnostics use one argmax over the five score channels, unchanged scores and labels. They are not candidates for selecting the metric closest to the paper.

1. First-label hybrid: replace multihot truth with the first positive channel in the stored order; P=accuracy, R=mean per-class recall over represented true classes, F1=harmonic(P,R). This generalizes the released method's declared truth handling, but does not establish the paper convention.
2. Any-hit hybrid: P=mean(predicted channel belongs to the true set); R=mean per-channel recall over supported channels; F1=harmonic(P,R).
3. Multilabel macro-F1: mean per-channel F1 between argmax-onehot prediction and multihot truth, all C channels, zero denominator -> 0.
4. Multilabel micro-F1: pooled TP/FP/FN across channels.
5. Pair-average F1: mean 2*hit/(1+number of true labels); pair-average P=hit rate, R=mean(hit/number of true labels).
6. Permutation sensitivity of first-label hybrid: all 5! simultaneous permutations of truth and scores; labels/semantics remain aligned. Report the full distribution/range, never choose a permutation.

Synthetic checks: perfect 4-type released metric=1; perfect 5-type+existence released metric=0 valid rows; generalized metric agrees with released method for K=4; perfect single-label predictions yield F1=1 under all conventions; multihot fixture distinguishes conventions; argmax ties and channel-order sensitivity recorded.

## Tasks

- [x] Snapshot primary paper/source repository (commit pin and recursive tree), record unavailable resources honestly.
- [x] Profile reference labels and compare CSV payload and snapshot tensor, including channel/name order.
- [x] Verify frozen inner export hashes, pair/label identities and metadata; calculate every predeclared diagnostic and train-only disease prior on the same validation pairs.
- [x] Trace final executed U1 objective, class-weight scaling, architecture, feature provenance and split assumptions against paper/source.
- [x] Write source-backed fidelity matrix and Vietnamese report with concrete next decisions and unresolved author questions.
- [x] Re-run local verification; compare source hashes before/after; emit artifact inventory (final handoff check).

Ruling: analysis scripts are audit instruments, not production feature changes. Use synthetic mathematical assertions and independent existing metric cross-checks; no new training tests or development branch needed.

Execution rulings: preserve all five channels and recovered singleton rows; do not choose a score-maximizing permutation; score only archived inner train/validation, and describe source-archive identity separately from author-data identity. SHA-verify executed modules instead of assuming live workspace matches. Final objective restores auxiliary sums; the sum-C class-weight scaling comparison is algebraic against a literal raw-weight objective, not an established author gap cause. Online SWH recovery could not be independently repeated; retain that limitation.
