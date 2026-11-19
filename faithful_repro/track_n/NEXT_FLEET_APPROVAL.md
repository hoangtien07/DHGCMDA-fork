# NEXT_FLEET_APPROVAL — what needs owner sign-off

Verdict: **READY_FOR_G4_OWNER_REVIEW** (see GATE_DECISION.json;
G0–G3 PASS, G4/G5 awaiting owner).

## Decision 1 — A1 input spec for the fleet

Protocol draft said "U1 embedding inputs"; the sanity arm used
train-only similarity-row features because generating U1
embeddings is itself a training run. Choose for the fleet:

- (a) keep A1 as similarity-row MLP (cheap, inductive, works
  cold-start for free) — recommended; or
- (b) generate U1-static embeddings per fold first (that IS a
  training fleet by itself) then A1 on top — ~5.5 h extra.

## Decision 2 — authorize G4

T-pair fleet: A1 × 5 folds × 3 seeds (~15 × few-min MLP runs,
sub-hour) + reuse of frozen B0–B2 cells. H1 bar: best baseline
B1_logistic macro-F1 0.5184 → success needs ≥ 0.548 paired mean.

## Decision 3 — authorize G5 (conditional on G4)

Cold-start deep jobs: 2 tracks × 5 folds × 3 seeds. Stopping rule
per protocol §8 (run only if G0–G4 pass; else SKIPPED_WITH_REASON).

## Explicitly not requested

- Any change to primary metric, target Δ, frozen splits.
- External author outreach (still gated separately).
- Additional seeds beyond {1234,42,2025}.
