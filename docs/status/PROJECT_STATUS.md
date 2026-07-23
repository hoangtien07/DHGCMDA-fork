# DHGCMDA active project status

> Last consolidated: 2026-07-23 on branch `run-readiness`. This document supersedes
> `EXPERIMENT_STATE.md`, `WAKEUP_CHECKLIST.md`, and long agent-context narratives for active work.

## Current scientific position

The project is now a reproducibility audit plus a controlled improvement study. Historical
headline results must be labelled by protocol:

| Protocol | v2.0 Top-1 F1 | AUC | Interpretation |
|---|---:|---:|---|
| `legacy_leaky` | 0.6969 ± 0.0314 | 0.9875 ± 0.0016 | Legacy code path; test positives leaked into features. |
| `honest_v2_fullcv` | 0.6151 ± 0.0204 | 0.9361 ± 0.0072 | Canonical leakage-free, full-CV baseline. |

The paired leakage gap is +0.0818 Top-1 F1 and +0.0514 AUC in favour of the leaky path.
Use the honest protocol for every new comparison. The historical `0.697 ± 0.003` K=2 result
is retained as a legacy result, not as a generalization headline.

## Protocol contract

Canonical v2.0 arguments are:

```text
--device cpu --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full
--leakage_free --deterministic --loss_mode two_head --exist_weight 0.3
--fusion_mode fixed --epoch 650 --validation 5 --seed {0,42,1234}
```

Primary outcomes are AUC, prevalence-aware AUPR when available, and Top-1 type metrics.
Threshold-dependent binary F1 is secondary until threshold selection is train-only. Do not
compare a clean result directly to paper-reported or legacy/leaky headline numbers.

## Verified findings that remain active

- P7/P9/P10: leakage-free masking includes recomputed GIP; full CV covers all pairs; manifests
  capture provenance.
- v3.2 evaluation had a 4-type metric bug. Corrected evaluation gives non-zero type performance.
- v3.2 training also had a Tissue mapping bug; the fixed diagnostic restored Tissue recall from
  0.000 to 0.479.
- `M_GSM` is association-derived GIP, not miRNA sequence similarity. A real miRBase k-mer view
  (`M_SEQ.txt`) is available but lacks a canonical multi-seed comparison.
- Multi-type pairs are collapsed by the legacy target. P3 is intentionally blocked until output
  and loss semantics are repaired in a separate branch.

## Next-run queue

| Order | Arm | Status | Question | Command surface |
|---:|---|---|---|---|
| 1 | P6 | READY | Does full-bilinear still beat diagonal prediction under clean evaluation? | `run_next.* run p6` |
| 2 | P2 | READY | Does real sequence view beat association-derived GIP under clean evaluation? | `run_next.* run p2` |
| 3 | P1 gate | READY | Does one learned global fusion weight help over fixed 0.6/0.4? | `run_next.* run p1` |
| 4 | P5 | READY | Does LDAM improve per-type recall without degrading global metrics? | `run_next.* run p5` |
| — | P1 attention | HOLD | Run only if scalar gate has a credible signal. | Not exposed by default |
| — | P3 | BLOCKED | Softmax type outputs are incompatible with genuine independent multi-label BCE. | Logic-fix branch required |
| — | Conformal refresh | HOLD | Re-run only from honest, mapping-fixed out-of-fold scores. | After core queue |

Every arm runs seeds `0, 42, 1234`, five folds and 650 epochs. New output is immutable under
`logs/plan_n/<arm>/<run-id>/` and `results/plan_n/<arm>/<run-id>/`.

## Local environment readiness

This checkout currently has no project `venv` for either Windows or Linux. The queue is
configured but `run_next.* check` will correctly fail until the selected runtime is installed:

- Linux: run `./setup_linux.sh` on the target Linux machine.
- Windows: create the pinned Python 3.12 environment using the command in `requirements.txt` /
  the archived context, then use `PowerShell -ExecutionPolicy Bypass -File .\run_next.ps1 ...`
  if the local execution policy blocks scripts.

Do not treat a missing venv as a scientific blocker or alter the registry status of an experiment.

## Resume safely

```text
./run_next.sh status
./run_next.sh check
./run_next.sh dry-run p6
./run_next.sh run p6 --run-id 20260723-p6

.\run_next.ps1 status
.\run_next.ps1 check
.\run_next.ps1 dry-run p6
.\run_next.ps1 run p6 -RunId 20260723-p6
```

`check` validates the environment, registry and required data. `dry-run` does not train.
The report compiler is guarded as legacy; it requires explicit acknowledgement before it can
regenerate a report from pre-audit data.

## Report and history policy

- Keep all historical logs/results in place. The registry labels them; it does not delete them.
- [Legacy report bundle](../archive/reports/2026-07-07/README.md) is reference-only.
- Root `BaoCao_DHGCMDA.docx` is a legacy long-form draft. There is no publication-ready report
  until canonical results and the report-refresh task are complete.
- Historical context is retained under `docs/archive/context/2026-07-23/`.
