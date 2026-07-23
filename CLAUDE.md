# DHGCMDA — Project Context

This repository now keeps its active state in one place:

- Human status and next-run queue: [docs/status/PROJECT_STATUS.md](docs/status/PROJECT_STATUS.md)
- Machine-readable registry: [docs/status/registry.json](docs/status/registry.json)

The previous long-form session context is retained without edits at
`docs/archive/context/2026-07-23/CLAUDE.pre-consolidation.md`.

## Working rules

1. Use `run_next.sh` on Linux or `run_next.ps1` on Windows for planned experiments.
2. Historical results remain immutable. New runs go under `logs/plan_n/` and `results/plan_n/`.
3. Do not treat legacy/leaky metrics as the honest baseline or publication headline.
4. P3 remains blocked until a dedicated branch corrects multi-label output/loss semantics.
