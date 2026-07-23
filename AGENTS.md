# DHGCMDA — Agent Context

The authoritative project state is [docs/status/PROJECT_STATUS.md](docs/status/PROJECT_STATUS.md).
Machine-readable experiment and artifact state is [docs/status/registry.json](docs/status/registry.json).

Read those files before selecting a run. Legacy narrative context was preserved verbatim at
`docs/archive/context/2026-07-23/AGENTS.pre-consolidation.md`.

## Non-negotiable run policy

- Treat all historical `0.697` / `AUC≈0.98` results as **legacy/leaky** unless a registry record
  explicitly identifies the protocol as `honest_v2_fullcv`.
- The canonical v2.0 benchmark is full-bilinear, K=2, full CV, leakage-free and deterministic.
- Do not run P3 multi-label until its registry blocker is resolved in a separate logic-fix branch.
- Use `run_next.sh` or `run_next.ps1`; do not overwrite historical `logs/` or `results/` paths.
