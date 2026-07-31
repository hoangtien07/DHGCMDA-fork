# MDAv3.2-3 recovery, protocol repair, and Legion execution plan

**Status:** approved plan; Sol S0/S1A complete, Terra T0/T1A pending
**Date:** 2026-07-26  
**Scope:** HMDD v3.2 only; HMDD v4 is explicitly out of scope  
**Target machine:** Windows/PowerShell, Intel i5-9300H, GTX 1650 4 GB, RAM 16 GB  
**Authoritative state:** read `PROJECT_STATUS.md` and `registry.json` before every work session  
**Companion work plans:** `MDA_V32_3_SOL_WORKPLAN.md` and `MDA_V32_3_TERRA_WORKPLAN.md`

## 1. Locked decisions

The user approved:

- **R1 — exact-first with fallback:** recover the original artifact first. If it cannot be
  found, build a deterministic candidate but never call it exact without entity/triple-level
  identity evidence.
- **S1 — search stop-loss:** spend at most 8–10 active hours on artifact search.
- **M2 — two protocols:** keep a `paper_compatible` reproduction track and a separate
  leakage-free, multi-label scientific track.
- **Multi-label default:** a `(miRNA, disease)` pair may have several functional types.
- **D1 — portable Git handoff:** commit code, tests, manifests, hashes, download/rebuild
  scripts, and documentation. Do not commit raw or recovered data by default.
- Use the Legion GTX 1650 only after a separate CUDA qualification gate. CPU remains a valid
  fallback, but CPU and CUDA results must never be pooled under one run identity.

No P3 training is authorized by this plan until the registry blocker is removed on a dedicated
logic-fix branch.

## 2. Council verdict

The independent council did not reach an artificial consensus. Its common minimum conclusion is:

1. Raw HMDD v3.2 is present; benchmark identity and historical preprocessing are the data
   blockers.
2. `MDAv3.2-3 = 411 × 271 × 5` with 11,748 positive typed triples is documented by
   SPLDHyperAWNTF, but the public description is insufficient to prove an exact reconstruction.
3. A simple `degree >= 2` filter cannot produce the target by the published description.
   SPLDHyperAWNTF associates `>=2` with `610 × 333 / 11,182` and `>=3` with
   `411 × 271 / 11,748`.
4. The fact that the stricter published variant contains more triples than the looser variant is
   inconsistent with ordinary nested pruning over one fixed input universe. Treat this as a
   provenance ambiguity, not as permission to tune filters until counts match.
5. The public DHGCMDA Git history currently contains v2.0 data, not the target v3.2 artifact.
6. The current P3 path is not genuine multi-label learning and must remain blocked.
7. The existing runner is a verified v2 CPU runner, not a portable v3/CUDA runner.

### 2.1 Dissent retained

- The data-forensics view considers exact recovery possible only with an archived artifact,
  entity/triple list, or canonical checksum. Matching published counts is not enough.
- The modeling view can continue with a labelled candidate dataset, but only for the scientific
  protocol; it cannot use that candidate to claim exact paper reproduction.
- The systems view considers GTX 1650 acceleration optional. For this small full-graph model,
  CUDA may be no faster than CPU after preprocessing and transfer overhead.

These positions are compatible operationally but represent different evidentiary thresholds. The
A/B/C classification below preserves the distinction.

## 3. Evidence baseline in the current workspace

The six HMDD files under `HMDD_data/MDAv3.2/` are available locally. Direct inspection gives:

| Input view | Evidence rows excluding header |
|---|---:|
| `v3_alldata.txt` | 32,281 |
| `v3_circulation.txt` | 4,120 |
| `v3_epigenetics.txt` | 644 |
| `v3_genetics.txt` | 2,302 |
| `v3_target.txt` | 9,064 |
| `v3_tissue.txt` | 8,230 |
| Five type files combined | 24,360 |

For the five type files, before name normalization or MeSH/MISIM filtering:

- 18,085 exact unique `(raw_miRNA_name, raw_disease_name, functional_type)` triples;
- 6,275 duplicate typed rows;
- 14,218 unique raw `(miRNA, disease)` pairs;
- 2,785 raw pairs with more than one functional type;
- 1,051 distinct raw miRNA names;
- 758 distinct raw disease names.

Minimal `strip + lowercase` normalization collapses one additional record, giving 18,084
typed triples and 14,217 pairs. It also exposes 865 typed triples without a MeSH ID and 46 MeSH
IDs associated with more than one normalized disease name. These are diagnostics, not the
historical normalization specification.

This corrects the earlier local claim of 23,732 five-type rows. That lower number came from an
incomplete category-prefix mapping which omitted categories such as `therapeutic target`,
`transcription factor target`, and `lncRNA target`. The five official type files are the primary
source for top-level type membership unless a recovered artifact proves otherwise.

Existing local derivatives are not the target benchmark:

| Dataset | miRNA | Disease | Typed rows | Unique pairs | Status |
|---|---:|---:|---:|---:|---|
| `v3.2_processed` | 722 | 612 | 13,748 | 13,748 | collapsed single-label |
| `v3.2_wang` | 713 | 447 | 12,534 | 12,534 | collapsed single-label |
| `v3.2_wang_dense` | 385 | 275 | 10,888 | 10,888 | density approximation |
| `v3.2_wang_multilabel` | 713 | 447 | 16,341 | 12,534 | multi-label, wrong benchmark |

## 4. Success classes

### A — exact recovery

All conditions are required:

- credible original/archived artifact or independently verified copy;
- exactly 411 miRNAs, 271 diseases, five declared types, and 11,748 unique typed triples;
- canonical entity and triple lists;
- provenance URL/archive/commit plus SHA-256;
- no duplicate canonical triple;
- a rebuild, if claimed, matches canonical entity/triple sets or their sorted checksums.

Only class A may be described as “the exact MDAv3.2-3 benchmark used by the paper.”

### B — candidate reconstruction

- pinned HMDD 2019.01, historical MeSH mapping, and MISIM 2.0 coverage where obtainable;
- deterministic, staged pipeline with manifests and tests;
- published counts and available intermediate statistics are matched;
- every ambiguous preprocessing choice is declared;
- no entity/triple reference exists to prove class A.

Use the identifier `mda-v32-3-candidate-B-<revision>`. Count matching alone does not upgrade B
to A.

### C — reproducible local derivative

Historical dependencies cannot be recovered or published statistics cannot be matched. The
result may be useful for methods work but must have a new dataset name and may not support a
paper-reproduction claim.

## 5. Critical code blockers and their repair order

### B0 — benchmark identity

Unknown historical entity normalization, MeSH category-C filtering, MISIM 2.0 coverage, duplicate
semantics, and pruning order prevent exact reconstruction.

**Gate:** classify the data A, B, or C before model work consumes it.

### B1 — predictor/loss mismatch

`prepareData.py::read_association_csv` writes one scalar type into each matrix cell. Repeated
typed rows for the same pair therefore follow a last-row-wins rule before training, splitting,
and feature construction.

`hetero_model.py::SimplifiedTypePredictor.forward` emits softmax-normalized type probabilities in
the non-`softmax_5class` path.  
`main_experiments_hetero1.py::_compute_multilabel_bce_loss` then applies ordinary BCE to these
probabilities even though its contract calls for independent logits. The type channels still
compete and cannot represent independent labels correctly.

**Required repair:**

- expose one raw existence logit and five independent raw type logits;
- use `BCEWithLogitsLoss` or `binary_cross_entropy_with_logits`;
- apply sigmoid only for inference/metrics;
- compute type loss only on observed positive train pairs;
- derive any class statistics from the outer-train fold, not the full dataset;
- verify that increasing one type logit does not lower another type probability;
- retain paper-compatible output behavior behind an explicit reproduction-only protocol.

### B2 — single-label data and split contract

`prepareData.py::preprocess_indices` derives folds from a 2-D association matrix and hard-codes
`type1` through `type4`. Multi-type rows can already have been collapsed before splitting.

**Required repair:**

- make the typed triple table/multi-hot tensor the label source of truth;
- build the set of unique pairs first;
- assign each pair to exactly one fold;
- carry all five labels of a pair with that pair;
- assert pair-disjoint train/test folds;
- parameterize the number and names of types.

### B3 — leakage contract

The current `--leakage_free` path masks test positives and recomputes miRNA GIP, which is useful,
but it was written around the single-label matrix.

In the local Wang-derived v3 pipeline, `M_FSM` is also calculated from the full association data,
`M_GSM` is copied from that view, and disease view fallbacks may duplicate another matrix. The
scientific path must not reuse full-data integrated `ID/IM` or similarity fallbacks after a
fold-materialization error.

**Required repair:**

- create a fold-local binary existence matrix from train pairs only;
- derive every association-dependent graph, GIP view, reconstructed view, class prior, sampler,
  and threshold only from the training fold;
- test that changing held-out labels cannot change any train-fold feature hash;
- identify every similarity as biological/external or association-derived; do not mislabel GIP
  as sequence similarity.
- fail closed when fold-local feature materialization fails.

### B4 — negative semantics

HMDD zeros are unobserved pairs, not experimentally verified biological negatives.

**Required repair:**

- call them `unobserved` in manifests and reports;
- reproduce the paper’s negative sampling only in `paper_compatible`;
- in the scientific protocol, seed and log sampling, draw only from train-fold unobserved pairs,
  and keep test sampling fixed before model comparison;
- report the closed-world assumption and add a small negative-ratio sensitivity check only after
  the canonical pipeline passes.

### B5 — five-type evaluation

Some code paths and comments assume four types or five total channels. For v3.2, five types plus
existence/no-association can require six channels depending on protocol.

The legacy evaluator also chooses a binary threshold from the test labels and repeatedly
resamples test negatives to a balanced 1:1 ratio. Those outputs are not prevalence-preserving
scientific estimates.

**Required repair:**

- use an ordered type vocabulary in the dataset manifest;
- remove shape checks tied to exactly five channels;
- add per-type checks including Tissue;
- compute multi-label micro/macro/per-type AUPRC, AUROC where meaningful, F1 at train-selected
  thresholds, LRAP or ranking metrics, Recall@K, subset/exact match only as a strict secondary
  metric;
- compute pooled out-of-fold existence ROC-AUC and average precision against one frozen
  evaluation population rather than metric-time negative resampling;
- keep legacy Top-1 metrics for comparison, clearly labelled as lossy for multi-label pairs.

### B6 — device and runner contract

`run_next.py` hard-codes the v2 dataset, CPU arguments, and hides CUDA.
`main_experiments_hetero1.py`, `trainData.py`, and other modules contain global
`cuda if available` decisions that can ignore the requested device.

**Required repair before GTX 1650 use:**

- resolve the device once after argument parsing;
- record both `requested_device` and `resolved_device`, and fail when they differ;
- pass the resolved device into data, model, and helper code;
- give CPU and CUDA separate dependency locks and run fingerprints;
- add a v3-aware runner whose job unit is `(dataset_id, protocol, backend, seed, fold)`.

## 6. Execution phases and gates

Do these phases in order. A failed gate stops downstream work; it does not trigger manual
count-fitting or an unrecorded fallback.

### R0 — isolate and freeze state

**Owner:** Terra implements; Sol reviews  
**Active time:** 1–2 hours

1. Start from a reviewed clean commit.
2. Create a dedicated worktree/branch, suggested:
   `codex/v32-portable-d1`.
3. Record current commit, worktree status, raw-file hashes, Python/platform information, and the
   A/B/C status `unknown`.
4. Add narrow ignore rules for recovery raw/external/intermediate/final data.
5. Do not modify or overwrite historical `logs/` or `results/`.

**Gate R0:** clean tracked tree, inputs inventoried, outputs isolated.

### R1 — artifact hunt with S1 stop-loss

**Owner:** Terra performs the registered mechanical search; Sol defines acceptance and reviews
ambiguous candidates  
**Active time:** maximum 8–10 hours

Search in this order:

1. SPLDHyperAWNTF supplementary files and the former
   `Ouyang-Dong/SPLDHyperAWNTF_` repository;
2. GitHub forks, branches, commit caches, GHArchive, repository dependencies, and binary files in
   citing implementations;
3. Software Heritage and Wayback/CDX;
4. Zenodo, DataCite, Figshare, OSF, Gitee, GitLab;
5. citing/related benchmark repositories, including files whose names do not contain
   `411`, `271`, or `11748`.

For each candidate binary, inspect shape, nonzero count, entity lists, type distribution, archive
metadata, and hash. A filename or matching count is not sufficient.

Do not contact authors, open issues, or send external messages under this plan.

**Gate R1:**

- credible artifact found → validate under R4;
- no credible artifact after 8–10 hours → freeze the search ledger and start R2 as class-B work.

Resource allocation inside the S1 budget:

- Sol: 1–2 hours to define acceptance tests and high-risk code-archaeology targets;
- Terra: about 5–6 hours to execute the fixed search ladder and collect/hash candidates;
- Sol: 1–2 hours to review candidates, contradictions, and chain of custody.

### R2 — freeze historical inputs

**Owner:** Sol specifies evidence; Terra builds fetch/verify tooling  
**Active time:** 4–8 hours

Required input manifest fields:

- logical source ID, version/date, URL/archive URL, access date;
- license/redistribution note;
- expected size and SHA-256;
- local relative path or external data-root key;
- role in the pipeline and whether it is mandatory for A, B, or C.

Inputs:

- HMDD v3.2 2019.01 five type files;
- historical MeSH descriptor/name/tree mapping sufficient to select category C;
- MISIM 2.0 entity coverage and, if required, functional similarity;
- every recovered alias or canonicalization table.

Downloaders must be idempotent, write `.part`, verify hash/size, and atomically rename. They must
support a documented offline cache. A live URL is not a dataset identity.

**Gate R2:** every input has provenance and a hash, or the missing dependency forces class C.

### R3 — declarative reconstruction

**Owner:** Terra implements from a Sol-reviewed specification  
**Active time:** 10–18 hours

Implement one configurable pipeline, not copied scripts:

1. parse five official type files;
2. map all source subcategories to the five declared functional types;
3. normalize Unicode, whitespace, case policy, miRNA aliases, and disease identifiers;
4. deduplicate exact canonical `(miRNA, disease, type)` triples;
5. map diseases to MeSH descriptors and retain category C according to the pinned snapshot;
6. intersect miRNAs with pinned MISIM 2.0 coverage;
7. evaluate the documented pruning variants:
   - degree measured by unique pairs versus typed triples;
   - one-pass versus iterative pruning;
   - simultaneous versus ordered miRNA/disease pruning;
   - normalization/mapping before versus after deduplication;
   - threshold `>=2` and `>=3` as separate named variants;
8. produce immutable stage-count and set-difference reports.

Canonical outputs:

- `entities_mirna.csv`;
- `entities_disease.csv`;
- `types.json`;
- `triples.csv`;
- `target_multihot.npz`;
- `stage_counts.csv`;
- `manifest.json`;
- `SHA256SUMS.json`.

**Gate R3:** two clean rebuilds produce identical hashes; no pair loses a valid type; no manual
entity selection is used to force the published counts.

### R4 — conformance ladder

**Owner:** Sol  
**Active time:** 4–8 hours

Check, in order:

1. input and post-dedup counts;
2. post-MeSH and post-MISIM entity/triple counts;
3. all named pruning variants;
4. published `MDAv3.2-2 = 610 × 333 / 11,182`;
5. independently published `MDAv3.2-3 = 411 × 271 / 11,748`;
6. per-type counts, multi-label-pair counts, degree distributions;
7. entity/triple set differences against a recovered artifact, if any.

Classify A/B/C. Do not assume `-3` is a subset of `-2`.

**Gate R4:** signed-off A/B/C decision and immutable dataset ID.

### R5 — package for DHGCMDA without label loss

**Owner:** Terra; Sol reviews  
**Active time:** 4–8 hours

- Use `v2.0_495m383D/` only as an I/O format reference.
- Preserve one row per typed triple and a multi-hot tensor.
- Validate IDs, shapes, type ordering, matrix symmetry/diagonal where applicable, NaN/Inf,
  duplicates, type distribution, and hashes.
- Do not replace described biological similarities with GIP while retaining biological labels.
- Commit only scripts, schemas, manifests, expected hashes where lawful, tiny fixtures, and tests.

**Gate R5:** loader round-trip preserves the canonical triple set and all validators pass.

### L0 — create a separate P3 logic-fix branch

**Owner:** Sol leads; Terra adds bounded tests  
**Active time:** 16–28 hours

Suggested branch after R4/R5 merge:
`codex/v32-multilabel-logic`.

Implement B1–B5 in dependency order:

1. manifest-driven five-type vocabulary;
2. pair-group fold builder;
3. fold-local association matrices/features;
4. independent logits and BCE-with-logits;
5. multi-label evaluation and train-only threshold selection;
6. paper-compatible adapter, without weakening the scientific path.

Do not remove the registry block until the model, split, leakage, and metric tests all pass.

**Gate L0:** P3 blocker replaced by a reviewed runnable protocol record, not merely bypassed with
`ALLOW_UNSAFE_P3`.

### I0 — portable Windows/Legion infrastructure

**Owner:** Terra implements; Sol reviews device semantics  
**Active time:** 18–30 hours, partly overlapping R2–R5

Deliver:

- `scripts/bootstrap_v32.ps1`;
- a CPU dependency lock and a separately qualified CUDA dependency lock;
- fetch/verify scripts and offline-cache support;
- v3 runner/config with seed–fold job granularity;
- atomic JSON/checkpoint writes, checksums, exclusive run lock, and exact-fingerprint resume;
- preflight for disk, RAM, NVIDIA driver, CUDA availability, GPU name/capability/VRAM;
- explicit CPU/CUDA run IDs and fingerprints.

The current CPU lock pins `torch==2.5.1+cpu`; it cannot accelerate the GTX 1650. Do not replace it
silently. Create and review a separate CUDA lock compatible with the installed NVIDIA driver.

Start CPU thread qualification at four physical-core-oriented threads; benchmark four versus
eight at the timing gate. Do not inherit the existing ten-thread default from a different CPU
without measurement.

**Gate I0:** a fresh clone on Legion can bootstrap, verify inputs from cache or public sources,
and pass unit tests without absolute paths from the original machine.

### Q0 — qualification, smoke, and timing gates

**Owner:** Terra executes; Sol reviews failures and protocol changes

1. **Q0a preflight:** imports, driver, disk, RAM, input hashes.
2. **Q0b device micro-smoke:** one epoch, one fold, seed 0; capture peak RAM/VRAM and NaN/Inf.
3. **Q0c correctness smoke:** three epochs, two folds; test interruption and exact resume.
4. **Q0d timing calibration:** 20 epochs, one fold; project full runtime with 20% reserve.
5. **Q0e pilot:** one seed, five folds, only after L0 and R4/R5 pass.
6. **Q0f canonical:** seeds `0,42,1234`, five folds, only after pilot review.

GTX 1650 OOM policy:

- first close unrelated GPU consumers;
- measure peak allocated and reserved memory;
- chunk pair scoring/evaluation only after a numerical-equivalence test;
- do not assume a generic batch-size change reduces full-graph memory;
- do not reduce hidden width, graph K, negative ratio, folds, or dataset as an “OOM fallback”;
- if CUDA still fails or strict determinism is unsupported, start a new CPU run ID;
- keep CUDA screening-only if it cannot satisfy the deterministic contract.

**Stop CUDA qualification** if repeated peak VRAM exceeds about 85%, requested and resolved
devices differ, deterministic operations fail, or CUDA provides no worthwhile timing advantage.

## 7. Protocol definitions

### `paper_compatible_v32`

Purpose: reproduce author-facing behavior as closely as evidence permits.

- Requires class A for an exact benchmark claim.
- If only B/C exists, label results `candidate reproduction`.
- Preserve published/code behavior only behind explicit configuration.
- If a single-label collapse rule cannot be recovered, do not invent one. Record the
  paper-compatible type metric as not exactly reproducible.
- Preserve and report any leakage or closed-world negative assumptions.
- Never use this protocol as the primary generalization claim.

### `honest_v32_multilabel_fullcv`

Purpose: primary scientific result on v3.2.

- five independent type logits with BCE-with-logits;
- unique-pair grouped five-fold CV;
- all labels of a pair remain in one fold;
- binary train adjacency is the graph/feature association input; scalar type IDs are never used
  as ordered numeric features;
- type loss is evaluated only on observed positive train pairs;
- train-fold-only association-derived graphs, GIP, priors, sampling, and thresholds;
- deterministic seeds `0,42,1234`;
- full-bilinear K=2 remains the starting architecture unless a registered experiment changes it;
- primary metrics: micro/macro/per-type AUPRC and multi-label F1 with train-selected thresholds;
- ranking and Recall@K secondary; legacy Top-1 reported only for historical comparison;
- unobserved pairs are not described as verified negatives.

Direct numerical comparison between these two protocols must be framed as a protocol gap, not as
an architectural improvement.

## 8. Required automated tests

### Data identity

- parser row counts and accepted type vocabulary;
- exact typed-triple deduplication;
- alias mapping fixtures;
- MeSH category-C and MISIM coverage fixtures;
- deterministic stage hashes;
- A/B/C classifier tests;
- loader round-trip equals canonical triple set.

### Multi-label logic

- a pair with two types remains two-hot through parse, split, loss, and metrics;
- shuffling raw input row order does not change canonical output or hashes;
- five types including Tissue are reachable;
- independent-logit test: increasing one type logit does not suppress another;
- BCE-with-logits agrees with a small hand calculation;
- no softmax in the scientific type head.

### Split and leakage

- train/test pair sets are disjoint;
- folds cover every eligible pair exactly once;
- held-out type labels cannot alter train-fold feature hashes;
- scientific mode fails rather than falling back to full-data `ID/IM` or similarity views;
- GIP and every association-derived object are recomputed from the fold-local matrix;
- train-only threshold selection.

### Runner and machine

- requested device equals resolved device;
- CPU and CUDA fingerprints are incompatible by design;
- input/data/code/lock hashes participate in resume compatibility;
- seed–fold interruption and resume;
- atomic writes and corrupted-artifact rejection;
- no canonical run from a dirty tracked tree.

## 9. Git and portability contract

Suggested sequence:

1. commit this plan on the current branch;
2. push it so the Legion clone can read the same source of truth;
3. create `codex/v32-portable-d1` from that reviewed commit;
4. implement R0–R5 and I0 without P3 training;
5. merge only after the dataset A/B/C gate and portability tests pass;
6. create `codex/v32-multilabel-logic`;
7. implement L0, qualify Q0, then expose v3 arms through `run_next.ps1`.

Do not commit:

- virtual environments, caches, `.part` files, credentials, cookies;
- raw HMDD/MeSH/MISIM or recovered archives when redistribution is unclear;
- generated tensors/similarities, checkpoints, logs/results by default;
- machine-specific absolute paths.

Prefer an external data root such as `D:\DHGCMDA-data\v32`, selected by a documented environment
variable or runner argument. Manifests must store logical/relative paths, not machine-specific
absolute paths.

## 10. Model allocation

Use `MDA_V32_3_SOL_WORKPLAN.md` for high-risk scientific decisions and review.  
Use `MDA_V32_3_TERRA_WORKPLAN.md` for bounded implementation, tests, scripts, and execution.

No Luna plan is created:

- Luna is not an available override in the current environment;
- no unique Luna-only task was identified;
- adding another model handoff would increase coordination cost without materially changing
  scientific or compute risk.

If Luna becomes available later, use it only for bounded log triage or documentation review, not
for benchmark identity, leakage, or multi-label design decisions.

## 11. Time and resource estimate

| Workstream | Active time |
|---|---:|
| R0 isolation/inventory | 1–2 h |
| R1 artifact search stop-loss | 8–10 h |
| R2 historical inputs | 4–8 h |
| R3 reconstruction | 10–18 h |
| R4 conformance | 4–8 h |
| R5 packaging/tests | 4–8 h |
| L0 multi-label protocol repair | 16–28 h |
| I0 portable runner/infrastructure | 18–30 h, partly overlapping |
| Q0 qualification | 4–8 h active plus machine runtime |

Expected total is roughly 8–14 active working days. Q0d must replace speculative runtime
estimates. Existing timing from another i5 CPU is not transferable to the i5-9300H/GTX 1650.

## 12. Stop conditions

Stop and report rather than silently adapting when:

- the R1 search budget expires;
- MeSH/MISIM snapshots or their provenance cannot be recovered;
- only final counts match but entity/triple identity cannot be established;
- a preprocessing choice has no evidence and materially changes the dataset;
- paper statistics remain internally inconsistent;
- a candidate would require hand-picking entities;
- license terms prevent lawful download/storage/distribution;
- requested and resolved devices differ;
- CUDA determinism, OOM, resume integrity, or hashes fail;
- a scientific run would require bypassing the P3 registry blocker.

Every stop preserves the search ledger, manifests, stage counts, hashes, failed gate, and next
question so another task does not repeat the same investigation.

## 13. Prompt for the next task

Use GPT-5.6 Sol first for a short acceptance/specification wave:

> Read `AGENTS.md`, `docs/status/PROJECT_STATUS.md`, `docs/status/registry.json`,
> `docs/status/MDA_V32_3_RECOVERY_PLAN.md`, and
> `docs/status/MDA_V32_3_SOL_WORKPLAN.md` completely. Execute S0 and S1A only.
> Scope is HMDD v3.2 only. Decisions R1/S1/M2/multi-label/D1 are locked. Do not run P3,
> do not contact authors, and do not overwrite historical outputs. Produce the evidence ledger,
> exact artifact-acceptance tests, registered search ladder, and Terra handoff. Do not spend Sol
> time executing routine searches or calculating hashes.

Then use Terra T0 and T1A to execute the registered search. Return to Sol S1B for candidate review
before starting the reconstruction specification.

## 14. Primary references

- SPLDHyperAWNTF: <https://academic.oup.com/bib/article/23/6/bbac390/6720405>
- DHGCMDA: <https://link.springer.com/article/10.1186/s12859-026-06436-w>
- HMDD download page: <https://www.cuilab.cn/hmdd>
- DHGCMDA repository: <https://github.com/CDMBlab/DHGCMDA>
- Former SPLDHyperAWNTF repository:
  <https://github.com/Ouyang-Dong/SPLDHyperAWNTF_>
