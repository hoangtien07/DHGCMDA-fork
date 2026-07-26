# GPT-5.6 Sol work plan — MDAv3.2-3

**Role:** scientific/data-contract owner and red-team reviewer  
**Scope:** HMDD v3.2 only  
**Do not:** run canonical training, bypass P3, contact authors, or perform routine mechanical work

Read completely before acting:

1. `AGENTS.md`
2. `docs/status/PROJECT_STATUS.md`
3. `docs/status/registry.json`
4. `docs/status/MDA_V32_3_RECOVERY_PLAN.md`
5. this file

## S0 — establish the evidence ledger

**Budget:** 1 hour

- Confirm current branch/commit and dirty files.
- Inventory every existing v3 raw/derived artifact and hash.
- Recompute the baseline counts in the master plan.
- Create a ledger distinguishing verified fact, paper claim, inference, contradiction, and open
  question.
- Do not treat an old generated dataset or matching count as a trusted reference.

**Deliverable:** versioned evidence ledger and a list of disputed claims.

## S1A — artifact acceptance and search registration

**Budget:** 1–2 hours

- Define candidate acceptance, chain-of-custody, canonical identity, and rejection tests.
- Register Terra's finite search ladder and query ledger schema.
- Identify high-risk code-archaeology targets where semantic reasoning is required.
- Reuse prior search evidence only when source, timestamp, query, and outcome are inspectable.

Hand off routine search execution to Terra T1A.

## S1B — candidate and stop-loss review

**Budget:** 1–2 hours after Terra T1A; total council S1 budget remains 8–10 active hours

- Review the search ledger for coverage and duplication.
- Inspect only ambiguous/high-value candidates by contents and statistics.
- Decide whether chain of custody supports class-A validation.
- Check whether the final 60–90 minutes produced any genuinely new candidate class.
- Do not contact authors.

**Stop:** when the total S1 budget is exhausted or a credible artifact reaches R4 validation.

**Deliverable:** frozen search ledger and one of:

- exact-artifact candidate;
- no artifact found, proceed to class-B reconstruction;
- legal/provenance blocker.

## S2 — reconstruction specification

**Budget:** 4–8 hours

Write the declarative specification Terra will implement:

- authoritative raw files and category-to-type mapping;
- normalization rules;
- alias policy;
- MeSH snapshot/category-C rule;
- MISIM 2.0 intersection rule;
- exact deduplication grain;
- named degree/pruning variants and their order;
- canonical sort and hash rules;
- required stage counts/set differences;
- A/B/C decision algorithm.

Every rule must include evidence strength and a falsification condition. Do not choose a rule
only because it approaches 411/271/11,748.

**Deliverable:** reviewed reconstruction spec, JSON schema requirements, and tiny edge-case
fixtures.

## S3 — conformance and data sign-off

**Budget:** 4–8 hours after Terra implementation

- Review code against S2 before considering outputs.
- Independently recompute high-impact counts.
- Validate deterministic rebuild hashes.
- Compare every intermediate stage and pruning variant.
- Compare entity/triple sets if an artifact exists.
- Assign A, B, or C and immutable dataset ID.

**Block:** no model code may use an unclassified dataset.

## S4 — multi-label scientific contract

**Budget:** 8–14 hours design/review

Specify and review:

- ordered five-type vocabulary;
- pair-group folds;
- independent existence/type logits;
- BCE-with-logits;
- fold-local association-derived features;
- unobserved-pair sampling semantics;
- train-only threshold selection;
- multi-label metrics;
- separation of `paper_compatible_v32` and `honest_v32_multilabel_fullcv`.

Require tests listed in the master plan. Preserve paper behavior only behind an explicit
reproduction-only adapter.

## S5 — device/runner red-team

**Budget:** 3–6 hours after Terra implementation

- Trace requested/resolved device end to end.
- Verify CPU/CUDA lock and fingerprint separation.
- Review fold-granular resume and artifact integrity.
- Confirm data, split, code, environment, and protocol hashes enter run identity.
- Reject silent CPU fallback, mixed backends, unsafe P3 bypass, or v2-only dataset hashing.

## S6 — pilot and canonical gate review

Sol does not spend time watching routine runs.

Review only:

- micro-smoke evidence;
- correctness/resume smoke;
- 20-epoch timing/VRAM report;
- one-seed five-fold pilot;
- metric and leakage audit.

Approve three-seed canonical execution only when all master gates pass.

## Tasks that should not use Sol

Delegate to Terra:

- PowerShell scaffolding;
- routine downloader implementation from an approved manifest;
- SHA calculation and file inventory;
- lock regeneration;
- JSON schema plumbing;
- tiny fixtures and mechanical test expansion;
- running smoke commands;
- log collection and Markdown formatting.

## Required Sol response format

For every wave:

1. verdict;
2. verified evidence;
3. counterevidence;
4. condition that would make the verdict wrong;
5. blockers versus caveats;
6. exact files/artifacts produced;
7. next authorized Terra task;
8. stop-budget consumption.

## Copyable first-task prompt

> Act as the Sol scientific/data-contract owner. Read all five required files in
> `docs/status/MDA_V32_3_SOL_WORKPLAN.md`. Execute S0 and S1A only. HMDD v4 is out of
> scope. Do not modify P3, run training, contact authors, or fit preprocessing choices to final
> counts. Define exact artifact acceptance, register the finite search ladder, and hand routine
> search execution to Terra. Finish with the required Sol response format and a Terra-ready T1A
> handoff.
