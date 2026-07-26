# GPT-5.6 Terra work plan — MDAv3.2-3

**Role:** bounded implementation, portability, tests, and execution  
**Scope:** implement only from a Sol-reviewed specification  
**Do not:** decide benchmark identity, invent preprocessing rules, bypass P3, or reinterpret metrics

Read completely before acting:

1. `AGENTS.md`
2. `docs/status/PROJECT_STATUS.md`
3. `docs/status/registry.json`
4. `docs/status/MDA_V32_3_RECOVERY_PLAN.md`
5. this file
6. the latest Sol evidence ledger/specification named by the handoff

## T0 — isolate the implementation

**Budget:** 1–2 hours

- Start from the reviewed plan commit.
- Create worktree/branch `codex/v32-portable-d1`.
- Confirm tracked tree is clean.
- Record commit and data-input inventory.
- Add narrow ignore rules for raw/external/intermediate/final data.
- Do not alter historical logs/results.

## T1A — execute the registered artifact search

**Budget:** about 5–6 hours; combined Sol/Terra S1 total must not exceed 8–10 active hours

- Use the exact ladder and ledger schema approved in Sol S1A.
- Perform publisher/repository/archive/scholarly searches mechanically.
- Download candidates only from in-scope public sources.
- Record source, query, timestamp, outcome, size, hash, and rejection reason.
- Inspect archive member names and basic shape/count metadata.
- Do not declare an artifact exact or resolve ambiguous chain of custody.
- Do not contact authors.

Stop when the registered sources are exhausted, no new candidate class appears in the final
60–90 minutes, or the combined S1 budget is reached. Hand the ledger to Sol S1B.

## T1B — portable D1 bootstrap and source verification

**Budget:** 5–8 hours

Implement:

- `scripts/bootstrap_v32.ps1`;
- idempotent source fetch/verify command;
- offline-cache option;
- `.part` download, size/hash verification, atomic rename;
- source manifest plus schema;
- CPU and CUDA environment locks as separate artifacts;
- preflight output for Python, packages, RAM, disk, driver, GPU, CUDA, and paths.

Rules:

- no machine-specific absolute paths in committed manifests;
- no silent dependency or device fallback;
- no raw data commit unless license review explicitly permits it;
- a fresh clone must be able to explain every missing external input.

## T2 — reconstruction pipeline

**Budget:** 10–18 hours

Implement exactly the Sol-approved S2 specification:

- stage-oriented pipeline with declarative configuration;
- official five-file parsing and type mapping;
- normalization/alias mapping;
- typed-triple deduplication;
- MeSH/MISIM filters;
- named pruning variants;
- deterministic canonical sorting;
- stage counts, set differences, manifest, and hashes;
- tiny fixtures and unit tests.

Do not add an undocumented heuristic to approach target counts. Report mismatches instead.

## T3 — packaging and loader tests

**Budget:** 4–8 hours

- Export triple table and multi-hot tensor.
- Preserve every type for multi-label pairs.
- Add shape, type-order, ID-alignment, duplicate, NaN/Inf, symmetry, and round-trip validators.
- Use the v2 directory only as an I/O shape example.
- Add logical dataset IDs for A/B/C outputs.

Stop and return to Sol for R4 classification.

## T4 — device and v3 runner implementation

**Budget:** 8–14 hours

After the data interface is frozen:

- resolve device once after CLI parsing;
- pass resolved device through dataset/model/helpers;
- remove module-global auto-CUDA decisions from the active path;
- fail when requested and resolved devices differ;
- make dataset path/hash/version manifest-driven;
- create job identity `(dataset_id, protocol, backend, seed, fold)`;
- provide atomic result/index writes, exclusive lock, attempts, checksums, and exact-fingerprint
  resume;
- keep CPU and CUDA runs incompatible by fingerprint.

Do not expose P3 as runnable in D1.

## T5 — bounded multi-label implementation

**Budget:** 8–14 hours after Sol S4

On the separate `codex/v32-multilabel-logic` branch:

- parameterize the five-type vocabulary;
- implement pair-group folds;
- provide independent raw logits;
- use BCE-with-logits;
- build fold-local association-derived objects;
- implement reviewed metrics and train-only thresholds;
- add all master-plan tests.

Terra must not choose a new loss, split, threshold, or metric outside the Sol specification.

## T6 — Legion qualification

Execute gates in order:

1. imports/hardware/input hashes;
2. one epoch, one fold, seed 0;
3. three epochs, two folds, including kill/resume;
4. 20 epochs, one fold, timing/thermal/RAM/VRAM capture;
5. one seed, five folds after Sol approval;
6. three seeds only after another Sol review.

Initial machine settings:

- Windows PowerShell;
- CPython 3.12 x64;
- four CPU threads, then compare eight only at timing calibration;
- CUDA environment separate from the existing CPU environment;
- GTX 1650 4 GB treated as unqualified until measured.

Do not reduce scientific hyperparameters as an OOM workaround. Pair-score/evaluation chunking is
allowed only with numerical-equivalence tests. If CUDA fails, create a new CPU run rather than
resuming the CUDA run.

## T7 — handoff and Git hygiene

- Run unit tests and plan validators.
- List generated but intentionally uncommitted artifacts.
- Update registry only after a gate is reviewed and actually achieved.
- Provide `git status --short`, commit scope, manifest IDs, and exact resume command.
- Do not push unless explicitly requested.

## Escalate to Sol when

- a historical input, alias, type mapping, or pruning rule is ambiguous;
- two plausible variants disagree materially;
- only count matching is possible;
- a protocol change would be needed for OOM or speed;
- CPU and CUDA differ beyond the reviewed tolerance;
- a metric or negative-sampling choice is not specified;
- a gate fails three times for the same scientific reason.

## Tasks Terra should not perform

- declaring an artifact exact;
- selecting preprocessing by final count;
- deciding that an inconsistency is a paper typo;
- weakening pair-group or leakage rules;
- changing primary metrics;
- approving canonical results;
- contacting authors or publishing external artifacts.

## Required Terra response format

1. implemented scope and commit/diff;
2. commands run;
3. tests and gates passed/failed;
4. generated artifact IDs and hashes;
5. uncommitted/external files;
6. resource measurements;
7. blockers requiring Sol judgment;
8. exact next/resume command.

## Copyable first-task prompt

> Act as the Terra implementation owner. Read all required files in
> `docs/status/MDA_V32_3_TERRA_WORKPLAN.md`, including the latest Sol specification. Execute only
> the explicitly handed-off T wave. If the handoff is Sol S1A, execute T0 and T1A only. Do not
> make scientific choices, run blocked P3, alter
> historical outputs, or claim exact recovery from matching counts. Keep D1 portable for Windows
> PowerShell, i5-9300H, GTX 1650 4 GB, and RAM 16 GB. Finish with the required Terra response
> format and stop at the next Sol review gate.
