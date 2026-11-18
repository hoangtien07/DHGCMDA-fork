# DECISION_LOG

| # | Date (UTC) | Decision | Basis |
|---|---|---|---|
| D1 | 2026-11-18 | Work branch `devin/20261118-repro-evidence-program` created at pinned base `a6cf4e9`; namespace `faithful_repro/repro_evidence_program_20261118/` (R). | Program doc §0/WP0 |
| D2 | 2026-11-18 | v20 `runs/full/fold_*.npz` marked UNAVAILABLE_LOCAL: manifest hashes exist but bytes not on this VM and not in git. Rescore of author-release raw predictions deferred to reconstruction; summary `metric_audit.json` used as-is and labeled. | WP0 rule: never pretend a hash provides the artifact |
| D3 | 2026-11-18 | Rescore target for WP4 = E08 outer_refit npz (pairs/y/logits, 30 files) + priors recomputed from fold counts — full row-level data available. | EVIDENCE_INVENTORY E08 |
