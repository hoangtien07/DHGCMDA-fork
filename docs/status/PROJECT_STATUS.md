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

## HMDD v3.2 recovery and P3 readiness

The user approved an HMDD v3.2-only workstream with these locked decisions: exact-artifact-first
with a labelled reconstruction fallback; an 8–10 active-hour artifact-search stop-loss; separate
`paper_compatible_v32` and leakage-free multi-label protocols; and a portable
code/manifest/downloader handoff for a Windows Legion Y540 (i5-9300H, GTX 1650 4 GB, RAM 16 GB).
HMDD v4 is out of scope for this workstream.

- Master plan: [MDA_V32_3_RECOVERY_PLAN.md](MDA_V32_3_RECOVERY_PLAN.md)
- GPT-5.6 Sol work: [MDA_V32_3_SOL_WORKPLAN.md](MDA_V32_3_SOL_WORKPLAN.md)
- GPT-5.6 Terra work: [MDA_V32_3_TERRA_WORKPLAN.md](MDA_V32_3_TERRA_WORKPLAN.md)

The workstream is planned, not executed. P3 remains blocked until dataset identity is classified
A/B/C and the loader, pair-group split, fold-local association-derived features, independent
type logits/BCE-with-logits, five-type evaluator, and truthful device contract pass their gates.

## Next-run queue

| Order | Arm | Status | Question | Command surface |
|---:|---|---|---|---|
| 1 | P6 | VERIFIED | Diagonal reduced Top-1 F1 by 0.06684; retain full-bilinear. | `20260724-local-p6-r1` |
| 2 | P2 | VERIFIED | Real sequence reduced Top-1 F1 by 0.00832; retain GIP. | `20260724-local-p2-r1` |
| 3 | P1 gate | VERIFIED | Scalar gate was neutral; retain fixed fusion. | `20260724-local-p1-r1` |
| 4 | P5 | VERIFIED | LDAM minority recall gain was only 0.00205; retain CE. | `20260724-local-p5-r1` |
| — | P1 attention | HOLD | Run only if scalar gate has a credible signal. | Not exposed by default |
| — | P3 | BLOCKED | Softmax type outputs are incompatible with genuine independent multi-label BCE. | Logic-fix branch required |
| — | Conformal refresh | HOLD | Re-run only from honest, mapping-fixed out-of-fold scores. | After core queue |

Every arm runs seeds `0, 42, 1234`, five folds and 650 epochs. New output is immutable under
`logs/plan_n/<arm>/<run-id>/` and `results/plan_n/<arm>/<run-id>/`.

## Local environment readiness

The Windows checkout now has a repository-local Python 3.12.13 `venv` with the pinned CPU
dependencies. The environment, runner tests, dry runs and local smoke gate pass. Preflight emits
a warning when less than 6 GB RAM is currently free and blocks below 3 GB; close memory-heavy
applications before canonical runs when practical.

## Compute strategy and Colab readiness

The available Windows laptop (Intel Core i5-1345U, 16 GB RAM and roughly 298 GB free in the
supplied System/About capture) is accepted for development, analysis and the current v2.0 CPU
queue. Its Intel UHD integrated graphics is not a CUDA accelerator. Local execution therefore
remains CPU-only and is now the primary execution path.

The local workstream is **COMPLETE** at milestone L9 (10/10 complete). Environment setup, reliable
runner tests, the two-fold smoke gate and the three-seed local canonical anchor are complete.
Anchor `20260723-local-anchor-r1` reproduced the honest reference with mean AUC 0.93601 and
Top-1 F1 0.61389 across 15 folds. P6 rejected diagonal prediction: Top-1 F1 fell by 0.06684
(Holm-adjusted p below 0.000001) and AUC fell by 0.00366, so full-bilinear remains canonical.
P2 did not adopt the real sequence view: its AUC gain was non-significant (+0.00147) while
Top-1 F1 fell by 0.00832. P1 scalar fusion was neutral (Top-1 -0.00040, AUC -0.00007,
Holm-adjusted p=1), so P1 attention remains on hold. P5 retained CE: LDAM changed Top-1 by
+0.00401 and minority mean recall by only +0.00205, below the +0.02 adoption gate. The queue
used CPython 3.12.13, pinned CPU dependencies, ten worker threads, seed-level resume and
per-arm review gates. See
[LOCAL_CPU_EXECUTION_PLAN.md](LOCAL_CPU_EXECUTION_PLAN.md).

The Colab workstream is **HOLD** at milestone C0 (1/10 complete). Production GPU runs are not
ready: the device contract, CUDA dependency lock, fold-level resume, atomic Drive artifacts and
CPU/T4 qualification have not passed.
The active canonical arguments still specify `--device cpu`; T4 output remains qualification
evidence until a reviewed contract update separates execution backend from scientific settings.

- Detailed living plan: [COLAB_FREE_EXECUTION_PLAN.md](COLAB_FREE_EXECUTION_PLAN.md)
- Machine tracker: registry record `colab-free-execution-readiness`
- Pause reason: local CPU is the current priority; no Colab implementation branch is active
- Policy: do not assume a daily GPU quota reset and do not use multiple accounts to evade
  Colab resource limits

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
