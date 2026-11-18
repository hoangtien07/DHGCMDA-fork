# REPRO_EVIDENCE_PROGRAM 2026-11-18 — namespace R

One-shot reproduction evidence program per
`DHGCMDA_ONE_SHOT_REPRO_EVIDENCE_PROGRAM_20261118.md` (owner attachment).

- Base commit (PIN): `a6cf4e94cb784a979919a9c566cb2442daa93dc2`
- Work branch: `devin/20261118-repro-evidence-program` (created from PIN)
- Tree ID at start: `13c85b15bbb704be65c2c2c2752eb637aa05d7ac`
- Environment: Python 3.12.16, torch 2.5.1+cpu, numpy 1.26.4,
  sklearn 1.5.2, PyG 2.6.1; venv `.venv-bank1/`; 8 CPU; `PYTHONUTF8=1`.
- `git status` at start: only uncommitted runtime artifacts
  (triplet_u1 bundles npz, recon_targets.npz, eval_exception.txt) —
  no source/report drift.

## Order of work

WP0 (this dir) → WP1 metric/split tribunal → WP2 data lineage →
WP3 fidelity trace → WP4 rescore → WP5 REV3 protocol → WP6 validators
→ Gate E → WP7 thesis narrative → WP8 handoff.

Resumable state: `STATUS.json` (single source of truth).
Decisions: `DECISION_LOG.md`.

## Reading order for the owner

`FINAL_HANDOFF.md` is the single summary. Everything else is evidence.
