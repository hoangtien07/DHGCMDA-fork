# Terra handoff — T0 and T1A

**Prepared:** 2026-07-31
**Authorized next wave:** T0 and T1A only
**Scientific scope:** HMDD v3.2 / MDAv3.2-3 only
**Search registration:** `mda-v32-3-artifact-search-20260731-v1`

## Preconditions

Read completely:

1. `AGENTS.md`;
2. `docs/status/PROJECT_STATUS.md`;
3. `docs/status/registry.json`;
4. `docs/status/MDA_V32_3_RECOVERY_PLAN.md`;
5. `docs/status/MDA_V32_3_TERRA_WORKPLAN.md`;
6. `docs/status/MDA_V32_3_EVIDENCE_LEDGER_V1.md`;
7. `docs/status/MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md`;
8. `docs/status/MDA_V32_3_SEARCH_LADDER_V1.json`.

Start T0 only from the reviewed commit containing all eight files. Create
`codex/v32-portable-d1`, confirm the index and tracked worktree are clean, and record the base
commit. An unrelated Microsoft Word lock file was present at S0; do not delete, move, or commit
it.

## Exact authorized work

### T0

- Create/switch to `codex/v32-portable-d1` from the reviewed S0/S1A commit.
- Record branch, commit, platform/Python, and the S0 v3 input inventory.
- Add only narrow ignore rules needed for quarantined downloads and generated search metadata.
- Keep all historical `logs/` and `results/` immutable.

### T1A

- Execute `MDA_V32_3_SEARCH_LADDER_V1.json` in order.
- Use its ledger and candidate schemas without dropping required fields.
- Store candidate bytes outside tracked Git content by default.
- For every downloaded container/member, calculate size and SHA-256, inspect basic
  shape/count/type/entity metadata, and run the mechanical tests in
  `MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md`.
- Preserve exact queries, UTC timestamps, failures, empty results, redirects, duplicates,
  license/redistribution notes, and active minutes.
- Label candidates only with the allowed Terra dispositions. `r4_ready` means “ready for Sol
  review,” not class A.

## Hard prohibitions

- Do not run P3 or any training.
- Do not edit predictor, loss, split, metric, or scientific preprocessing logic.
- Do not reconstruct MDAv3.2-3 or tune filters to 411/271/11,748.
- Do not contact authors, open issues, or send messages.
- Do not search HMDD v4.
- Do not overwrite historical outputs.
- Do not commit raw/recovered/candidate data, `.part` files, credentials, cookies, caches, or
  machine-specific absolute paths.
- Do not declare benchmark identity or resolve semantic ambiguities.

## Stop and escalation

Stop T1A at the first applicable condition:

1. a candidate passes every mechanical prerequisite for `r4_ready`;
2. 5.5 Terra active hours are consumed;
3. every registered ladder step is complete; or
4. the final 60–90 active minutes produce no new candidate class.

Escalate immediately when type semantics, association counts, entity axes, MeSH/MISIM rules,
pruning order, custody, conflicting copies, or license terms require interpretation. Preserve
the bytes and ledger state; do not adapt the protocol.

## Required return package

Return to Sol S1B with:

1. branch/base commit and `git status --short`;
2. exact commands/tools used;
3. the completed query ledger and candidate ledger;
4. immutable container/member hashes and inspection reports;
5. acceptance-test matrix for every candidate;
6. external/uncommitted file list and license notes;
7. active-time accounting by ladder step and stop condition;
8. blockers requiring Sol judgment;
9. one of `credible candidate for S1B`, `search ladder exhausted`, or `budget stop`;
10. no A/B/C declaration and no next-wave implementation beyond T1A.

After return, wait for Sol S1B. R2/R3 reconstruction, P3 changes, and model execution are not
authorized by this handoff.
