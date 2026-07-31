# Active status area

`PROJECT_STATUS.md` is the human-readable source of truth. `registry.json` is the
machine-readable source used by the runner and validator. Historical documents and results
are preserved in place and classified through the registry rather than moved or deleted.

Active implementation plans:

- [`LOCAL_CPU_EXECUTION_PLAN.md`](LOCAL_CPU_EXECUTION_PLAN.md) — primary Windows CPU execution,
  resume and active experiment queue.
- [`COLAB_FREE_EXECUTION_PLAN.md`](COLAB_FREE_EXECUTION_PLAN.md) — local CPU plus optional
  Colab Free/T4 readiness; currently on hold.
- [`MDA_V32_3_RECOVERY_PLAN.md`](MDA_V32_3_RECOVERY_PLAN.md) — HMDD v3.2 recovery and protocol
  repair; S0/S1A/T0/T1A/S1B are complete and the R2/S2 class-B specification is next.

HMDD v3.2 S0/S1A controls:

- [`MDA_V32_3_EVIDENCE_LEDGER_V1.md`](MDA_V32_3_EVIDENCE_LEDGER_V1.md)
- [`MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md`](MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md)
- [`MDA_V32_3_SEARCH_LADDER_V1.json`](MDA_V32_3_SEARCH_LADDER_V1.json)
- [`MDA_V32_3_TERRA_T1A_HANDOFF.md`](MDA_V32_3_TERRA_T1A_HANDOFF.md)
- [`MDA_V32_3_TERRA_T1A_EXECUTION_LEDGER_V1.json`](MDA_V32_3_TERRA_T1A_EXECUTION_LEDGER_V1.json)
- [`MDA_V32_3_SOL_S1B_REVIEW_V1.md`](MDA_V32_3_SOL_S1B_REVIEW_V1.md)
