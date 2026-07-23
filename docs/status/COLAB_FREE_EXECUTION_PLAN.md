# DHGCMDA local + Colab Free execution plan

> Status: provisional, revision 1, 2026-07-23  
> Machine-readable tracker: `docs/status/registry.json`, record
> `colab-free-execution-readiness`  
> Scientific source of truth: `docs/status/PROJECT_STATUS.md`

## Decision

The available Windows laptop is adequate for development, data preparation, validation,
analysis and the current honest v2.0 CPU queue. It is also adequate as the control machine for
Google Colab. It is not a local CUDA training machine: Intel UHD integrated graphics cannot
replace an NVIDIA CUDA GPU.

The accepted compute strategy is:

1. Keep the local CPU path as the reproducible fallback and control backend.
2. Treat a Colab T4, when offered, as optional burst capacity rather than guaranteed compute.
3. Use exactly the same scientific code, split definition and protocol on both backends.
4. Develop the Colab adapter on a short-lived `codex/colab-runner` branch, then merge it after
   the readiness gates below pass. Do not maintain a separate Colab model implementation.
5. Do not publish, pool or compare a Colab result until its manifest proves that it used
   `honest_v2_fullcv`.

## Hardware assessment

Assessment is based on the Windows System/About capture supplied on 2026-07-23. The capture is
not copied into the repository because it contains device identifiers.

| Resource | Observed configuration | Decision for the active project |
|---|---|---|
| CPU | Intel Core i5-1345U | PASS for development and current v2.0 CPU runs; sustained runs may be limited by laptop cooling |
| RAM | 16 GB | PASS for the current v2.0 dataset and model; close memory-heavy applications during full runs |
| Storage | 477 GB total, about 298 GB free in the supplied capture | PASS for source, environments and immutable run artifacts; keep a free-space guard |
| Local GPU | Intel UHD integrated graphics | NO CUDA; use CPU locally |
| OS | Windows 11 Pro, 64-bit | PASS through `run_next.ps1` after the pinned Python 3.12 environment exists |
| Colab role | Browser and Drive access from the laptop | PASS as an optional T4 control surface |

This assessment applies to the active v2.0 queue. A materially larger dataset, dense graph,
larger hidden size or new model family requires a fresh RAM, VRAM and runtime benchmark.

## Non-negotiable scientific rules

- All publishable comparisons use `honest_v2_fullcv`: full-bilinear baseline, K=2, full
  leakage-free CV, deterministic mode, 3 seeds and 5 folds.
- The current registry contract explicitly includes `--device cpu`. Until C2 produces a reviewed
  protocol update that separates scientific arguments from the execution backend, CPU remains
  the only canonical publication backend and T4 output is qualification evidence only.
- Historical `0.697` / AUC near `0.98` results remain legacy/leaky unless the registry explicitly
  says otherwise.
- P3 remains blocked. Colab capacity does not remove its output/loss-semantics blocker.
- A reduced-epoch, reduced-seed or single-fold run is screening evidence only. It cannot enter
  a paper table or replace a canonical run.
- Do not combine CPU folds and GPU folds inside one condition. Both sides of a comparison must
  use the same backend and pinned environment.
- Do not use multiple Google accounts to evade Colab resource limits. Google explicitly
  disallows using multiple accounts to work around access or resource restrictions.
- Do not assume that free GPU access resets every day. Colab does not publish a fixed GPU quota
  or reset schedule.

Reference: [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Target execution design

### Work unit

The resumable unit is:

```text
arm / variant / seed / fold
```

Selecting one fold must not regenerate a one-fold CV split. It must select one member of the
same deterministic five-fold split used by the canonical full-CV run.

### Artifact lifecycle

Each work unit has a unique immutable identity derived from:

- Git commit SHA and dirty-tree flag;
- dataset path and content hash;
- protocol and resolved arguments;
- arm, variant, seed and fold;
- Python, PyTorch, PyG, CUDA and driver versions;
- actual device type and GPU model;
- start/end timestamps and exit status.

A work unit writes to local VM storage first. On success it produces a closed artifact bundle
containing its log, metrics, score dump when required, manifest and checksum. Only then is the
bundle copied to persistent storage and marked `complete`. An interrupted copy must never look
complete.

### Resume policy

- The scheduler skips only units whose artifact, checksum and `complete` marker all validate.
- A failed or interrupted unit is rerun under a new attempt directory; completed historical
  output is never overwritten.
- Fold-level resume is required before production Colab use.
- Mid-fold epoch checkpointing is optional at first. Add it only if a measured fold duration or
  interruption rate makes rerunning one fold materially expensive.
- If epoch checkpoints are added, they must include model, optimizer, scheduler and all relevant
  RNG states, and must record the originating commit and dependency lock.

### Google Drive policy

- Use Drive only for persistent inputs and closed artifact bundles.
- Copy an archive to the Colab VM and unpack under local VM storage before training.
- Do not train directly against many small files through `drive.mount()`.
- Copy back only complete, checksummed bundles.
- Code comes from an exact Git commit. Drive must not become an untracked second source tree.

## Work breakdown and acceptance gates

### C0 — Assess and register the workstream

Deliverables:

- hardware decision;
- this plan;
- human-readable status entry;
- machine-readable registry record.

Exit criteria:

- tracking validator passes;
- registry and this tracker agree.

### C1 — Isolate infrastructure work

Deliverables:

- create short-lived branch `codex/colab-runner` from the active readiness branch;
- record the starting commit and clean/dirty state;
- preserve the existing P6 → P2 → P1 → P5 scientific queue.

Exit criteria:

- no scientific default or historical artifact changes merely to enable Colab;
- P3 remains blocked;
- existing CPU dry-run commands remain unchanged or have a documented compatible migration.

### C2 — Make the device and environment contract truthful

Deliverables:

- one authoritative device resolution path;
- a reviewed registry/status update that either keeps CPU as the sole canonical backend or
  explicitly separates a qualified execution backend from unchanged scientific arguments;
- a Colab-compatible, pinned CUDA/PyTorch/PyG environment;
- separate CPU and CUDA dependency locks where platform wheels differ;
- preflight output that reports requested device, actual device and accelerator model;
- manifest fields that match the device actually used.

Exit criteria:

- no GPU result is labelled canonical while the active protocol contract still requires
  `--device cpu`;
- requesting CPU cannot silently use CUDA;
- requesting CUDA fails clearly when unavailable;
- a T4 health check imports all required packages and performs a small tensor operation;
- CPU preflight still works on Windows.

### C3 — Add fold-level selection and safe resume

Deliverables:

- deterministic selection of one fold from the canonical five-fold split;
- immutable unit IDs and attempt directories;
- atomic unit status and checksum;
- skip-complete/resume-incomplete behavior;
- aggregation that accepts only a complete, compatible set of five folds.

Exit criteria:

- running folds separately produces the same fold membership as a monolithic five-fold run;
- interruption after a completed fold does not rerun or overwrite that fold;
- a mismatched commit, dataset hash or protocol is rejected during aggregation;
- the tracking validator and focused resume tests pass.

### C4 — Provide a minimal Colab bootstrap

Deliverables:

- one notebook or small bootstrap surface that checks out an exact commit;
- Drive mount and archive copy to local VM storage;
- pinned environment installation;
- preflight, one-unit launch, artifact validation and copy-back;
- no embedded credentials or account-specific absolute paths.

Exit criteria:

- a fresh Colab runtime can execute the bootstrap without relying on previous VM state;
- loss of the VM leaves the source artifact store consistent;
- notebook output clearly identifies screening versus canonical mode.

### C5 — Harden provenance and storage

Deliverables:

- manifest schema for backend, hardware, dependencies, split identity and checksums;
- bounded storage layout by arm/run/unit/attempt;
- free-space and Drive-copy checks;
- explicit artifact retention policy.

Exit criteria:

- every result can be traced to a commit, dataset hash, seed and fold;
- incomplete artifacts cannot be parsed as final results;
- no historical `logs/` or `results/` path is overwritten.

### C6 — Benchmark and qualify the T4 backend

Run the same fixed seed/fold/configuration on CPU and T4.

Measure:

- setup time;
- data preparation time;
- epoch and fold time;
- peak system RAM and T4 VRAM;
- GPU utilization;
- output schema, sample counts and split hashes;
- AUC, prevalence-aware AUPR and Top-1 metrics;
- deterministic warnings and repeated-T4 variability.

Qualification gates:

- split identity, sample counts, labels and output shapes match exactly;
- no NaN, OOM or silent CPU fallback;
- repeated runs on the same pinned T4 environment are reproducible or any residual variance is
  measured and documented;
- T4 gives a useful end-to-end benefit after setup, provisionally at least 1.5x for a full fold,
  or is retained only as overflow capacity;
- cross-backend metric differences are diagnostic only. Canonical comparisons remain
  same-backend paired runs.

### C7 — Run a controlled P6 pilot

P6 is first because it is already first in the scientific queue and has one variant.

Sequence:

1. run a short screening smoke test;
2. run one complete canonical seed as five resumable fold units;
3. aggregate and compare with the same-backend full-bilinear baseline;
4. deliberately stop one disposable attempt to verify recovery;
5. review artifacts before authorizing the remaining seeds.

Exit criteria:

- all five fold bundles and the aggregate validate;
- recovery loses at most the active fold;
- no protocol or device-manifest mismatch;
- result is registered under `honest_v2_fullcv`.

### C8 — Execute the active queue

After C7 passes:

1. complete P6;
2. run P2;
3. run the P1 scalar gate;
4. run P5 CE and LDAM;
5. keep P1 attention on hold unless the scalar gate has credible signal;
6. keep P3 blocked until its separate logic-fix branch resolves the semantics.

For exploratory method search, use a funnel:

- cheap screening run;
- one full seed;
- full canonical 3-seed × 5-fold confirmation only for promising candidates.

Only the last level is publication evidence.

### C9 — Merge and hand off operations

Deliverables:

- CPU and Colab operating instructions;
- failure/recovery checklist;
- final tracking update with evidence links;
- merge of the validated adapter into the active branch;
- retirement of the temporary integration branch.

Exit criteria:

- a clean checkout can reproduce both preflights;
- the active status and registry contain the final backend decision;
- there is no long-lived scientific divergence between local and Colab code.

## Progress tracker

This table and the `progress.milestones` array in `docs/status/registry.json` must be updated
together. `PROJECT_STATUS.md` carries only the current summary and next gate.

| ID | Status | Evidence | Exit/result |
|---|---|---|---|
| C0 | DONE | This plan, project status, registry | Hardware assessed and workstream registered |
| C1 | TODO | — | Infrastructure branch isolated |
| C2 | TODO | — | Device/environment contract truthful |
| C3 | TODO | — | Fold-level resume validated |
| C4 | TODO | — | Fresh Colab bootstrap succeeds |
| C5 | TODO | — | Provenance and atomic storage validated |
| C6 | TODO | — | T4 benchmark and qualification decision recorded |
| C7 | TODO | — | P6 pilot and interruption recovery pass |
| C8 | HOLD | — | Starts only after C7 |
| C9 | HOLD | — | Starts after production execution is stable |

Current progress: **1/10 milestones complete**. Next gate: **C1**.

## How to change this plan

This is a living plan. Changes are allowed when Colab availability, measured timings or project
priorities change, subject to these controls:

1. update the revision and date at the top;
2. explain the reason in a `Decision log` entry below;
3. update this tracker and the registry in the same change;
4. never relax the honest protocol implicitly;
5. move a milestone backward when new evidence invalidates its exit criteria;
6. keep completed evidence paths immutable; correct them with a dated note rather than silently
   replacing history.

## Decision log

| Date | Revision | Decision |
|---|---:|---|
| 2026-07-23 | 1 | Accept the laptop for local CPU/control use; pursue Colab Free as optional T4 capacity behind C1–C7 readiness gates |
