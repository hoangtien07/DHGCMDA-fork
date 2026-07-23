# DHGCMDA local CPU execution plan

> Status: active, revision 1, 2026-07-23  
> Machine tracker: `docs/status/registry.json`, record `local-cpu-execution-readiness`  
> Scientific source of truth: `docs/status/PROJECT_STATUS.md`

## Decision

The Windows laptop is the primary execution backend for the active v2.0 queue. Runs use CPU
only, CPython 3.12, pinned PyTorch 2.5.1 CPU wheels and a fixed 10-thread budget. Google Colab
work is on hold.

Execution order remains:

```text
local full-bilinear anchor -> P6 -> P2 -> P1 scalar gate -> P5
```

P1 attention remains on hold unless the scalar gate passes its evidence gate. P3 remains
blocked until its output/loss semantics are fixed on a separate branch.

## Runtime contract

- Python: CPython 3.12.13 from the Codex bundled runtime; do not use the installed Python 3.14
  runtime. The repository `venv` is independent after creation.
- Environment: repository-local `venv`, ignored by Git.
- Device: CPU. The training subprocess hides CUDA devices so the requested and actual backend
  cannot diverge.
- Threads: `DHGCMDA_N_THREADS=10` for every control/candidate comparison.
- Encoding: `PYTHONUTF8=1`.
- Power: AC power, lid open, Windows Balanced plan. Sleep and hibernate on AC must remain off.
- Minimum preflight resources: 10 GB free disk, 12 GB installed RAM and 3 GB currently
  available RAM. Six GB available is recommended; lower availability emits a warning because
  the smoke run demonstrated that folds execute sequentially on this 16 GB machine.

All canonical runs use `honest_v2_fullcv`, seeds `0, 42, 1234`, five folds and 650 epochs.
Reduced smoke runs are `screening_only` and never publication evidence.

## Runner reliability contract

The local runner must:

- expose a reusable `baseline` arm;
- allow a seed subset while defaulting to all canonical seeds;
- resume at seed/job boundaries without overwriting previous attempts;
- retry parsing without retraining when training completed successfully;
- write `run_index.json` atomically;
- reject resume when commit, dataset, protocol, dependencies or thread count differ;
- validate result checksums before skipping completed work;
- create compatible paired summaries for baseline-vs-P6/P2/P1 and CE-vs-LDAM;
- preserve P3 as blocked.

Local v1 deliberately does not checkpoint within an epoch or fold. An interruption may lose the
active seed, estimated at 45–60 minutes, but must preserve all previously completed seeds.

## Execution gates

### L0 — Isolate and register local work

- Commit the existing compute-plan documentation.
- Create `codex/local-runner`.
- Register this plan and put the Colab workstream on hold.

### L1 — Build the Windows CPU environment

- Install CPython 3.12 user-level.
- Create `venv`.
- Install `requirements.txt` from the PyTorch CPU index plus PyPI.
- Pass package, version, disk, RAM, dataset and registry checks.

### L2 — Harden and test the runner

- Add baseline, smoke, seed selection, resume and summarize interfaces.
- Add attempt history, checksums, atomic indexes and compatibility fingerprints.
- Add standard-library unit tests and pass all dry runs.

### L3 — Run the local smoke gate

- Run seed 0, 3 epochs, 2 folds under `results/smoke/local/`.
- Require finite loss, no import/DLL error, no CUDA fallback and valid parsed output.
- Interrupt a disposable fake/test job and verify resume semantics.

### L4 — Establish the local canonical anchor

- Run full-bilinear/fixed/K=2 for all three seeds.
- Require 15 folds, no NaN and complete manifests.
- Sanity-check against the verified honest reference: absolute mean difference no greater than
  0.01 AUC and 0.03 Top-1 F1. A larger difference pauses the queue for audit.

### L5 — Run and review P6

- Compare diagonal prediction with the local full-bilinear anchor.
- Pause downstream architectural work if diagonal improves Top-1 F1 by at least 0.01 or AUC by
  at least 0.005 with Holm-adjusted p below 0.05.

### L6 — Run and review P2

- Compare the real sequence view with the same local anchor.
- Adopt only with positive Top-1 delta, Holm-adjusted p below 0.05 and AUC degradation no worse
  than 0.005. A negative/neutral result is still verified evidence.

### L7 — Run and review P1 scalar gate

- Compare the learned gate with fixed fusion.
- Open P1 attention only if Top-1 improves by at least 0.01, Holm-adjusted p is below 0.05, AUC
  degradation is no worse than 0.005 and learned weights do not collapse to a boundary in most
  folds.

### L8 — Run and review P5

- Compare CE and LDAM with score dumps.
- Prefer LDAM only if minority-class recall improves by at least 0.02 while Top-1 decreases no
  more than 0.01 and AUC decreases no more than 0.005.

### L9 — Close out

- Update registry records with run IDs, summaries and decisions.
- Validate all tracking.
- Commit reviewed immutable logs/results and merge the local-runner work.

## Planned command surface

```powershell
.\run_next.ps1 check
.\run_next.ps1 smoke baseline --run-id <date>-local-smoke
.\run_next.ps1 run baseline --run-id <date>-local-anchor
.\run_next.ps1 run p6 --run-id <date>-local-p6
.\run_next.ps1 summarize p6 --run-id <date>-local-p6 --baseline-run-id <date>-local-anchor
```

If a run is interrupted:

```powershell
.\run_next.ps1 run <arm> --run-id <same-id> --resume
```

## Progress tracker

This table and the registry milestone array must be updated together.

| ID | Status | Evidence/result |
|---|---|---|
| L0 | DONE | Documentation commit and `codex/local-runner` branch |
| L1 | DONE | Python 3.12.13 `venv`; pinned CPU packages; disk/data/dependency checks pass |
| L2 | DONE | Reliable runner; 8 unit tests, tracking validator and dry runs pass |
| L3 | DONE | `20260723-local-smoke-r1`: 2 folds/3 epochs completed on CPU with valid metrics |
| L4 | DONE | `20260723-local-anchor-r1`: 15 folds; AUC 0.93601; Top-1 F1 0.61389; reference gate passed |
| L5 | TODO | P6 |
| L6 | HOLD | P2 |
| L7 | HOLD | P1 scalar gate |
| L8 | HOLD | P5 |
| L9 | HOLD | Closeout |

Current progress: **5/10 milestones complete**. Next gate: **L5**.

## Estimated compute schedule

After engineering/setup:

| Run | Expected local CPU time |
|---|---:|
| Local anchor | 2–3 hours |
| P6 | 2–3 hours |
| P2 | 2–3 hours |
| P1 gate | 2–3 hours |
| P5 CE + LDAM | 4–6 hours |

Total expected compute is 12–18 hours, split by arm with a review and tracking update after each.

## Decision log

| Date | Revision | Decision |
|---|---:|---|
| 2026-07-23 | 1 | Make local CPU the primary backend, add seed-level resume, run each arm behind a review gate and put Colab work on hold |
| 2026-07-23 | 2 | Use the bundled CPython 3.12.13 runtime, accept the sequential-fold memory profile after a successful local smoke run and open the canonical anchor gate |
| 2026-07-23 | 3 | Accept local anchor `20260723-local-anchor-r1`; its mean differs from the verified honest reference by only -0.00009 AUC and -0.00121 Top-1 F1 |
