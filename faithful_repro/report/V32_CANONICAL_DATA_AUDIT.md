# v3.2 canonical/data audit (bounded, read-only, machine-readable ledgers)

Date: 2026-11-12. Task `task_388c3d5a24a6`. Read-only audit; all tensors, archived
splits, models, and results unchanged. No training/inference/probes.
Conventions: **[F]** observed fact, **[I]** conditional interpretation,
**[U]** unavailable author information. Ledger directory:
`faithful_repro/audit_v32_identity/`.

## 0. Sources audited (hashes in `raw_file_hashes.csv`)

| Artifact | Provenance | Role |
|---|---|---|
| `v3.2_spld_paper/` (Y_multilabel.npy, M_MISIM.npy, D_MESH.npy, triplets.csv, name files, manifest) | Software Heritage snapshot of the SPLDHyperAWNTF lineage (upstream repo deleted — snapshot-only) | claimed paper-fingerprint tensor: 411×271×5, 11,748 typed triplets on 8,735 unique pairs |
| `faithful_repro/data/v3.2_ref_MDAv3.2-3.npz` | built locally from the snapshot | faithful-run reference tensor |
| `faithful_repro/data/v3.2_ref_MDAv3.2-3_4view.npz` | built by `code/build_4view.py` | paper-first 4-view candidate inputs |

## 1. Identity vs matching counts — ESTABLISHED (not just counts)

Measured pairwise between `v3.2_spld_paper` and `v3.2_ref_MDAv3.2-3.npz`:

- `Y_multilabel.npy` == npz `Y` — **elementwise identical** (411×271×5) [F].
- miRNA and disease name files == npz `mir_names`/`dis_names` — identical
  order and values [F].
- npz `sim_m2` == `M_MISIM.npy`, npz `sim_d2` == `D_MESH.npy` — elementwise
  identical (not merely shape-compatible) [F].
- npz `sim_m1`/`sim_d1` are the `'GIP'` marker scalars (de-leaked GIP-on-train
  convention), same as the v2.0 reference layout [F].
- `triplets.csv` = 11,748 rows, header `mi,di,type`, 0-based indices [F].

So identity between the snapshot payload and the reference npz is proven.
What is **not** proven: that the snapshot payload is the paper's own tensor —
it arrives via a sibling-paper lineage (SPLDHyperAWNTF), not DHGCMDA's
release. Fingerprint match to Table-1/2 counts is strong but remains a
lineage claim [I].

## 2. 4-view artifact provenance — do NOT trust without this check

`v3.2_ref_MDAv3.2-3_4view.npz` internals measured:

- `sim_m1` (NW sequence view): built by `build_4view.py` `seqs()` — **name
  heuristic**, not the strict GFF3 evidence method used for v2.0: strip
  `-5p/-3p` from mature.fa keys, try the raw name then name-minus-trailing-
  digit. The v2.0 strict-vs-heuristic comparison showed this class of
  mapping differs on 14,194 cells — so this `sim_m1` is **heuristic-tier**,
  not evidence-mapped; a v3.2 fidelity claim requires a rebuild with the
  strict GFF3 `Derives_from` mapping [F/I].
- `sim_d1` (FunSim disease view): Cheng-style pipeline — unique-term
  MeSH→UI resolution + CTD `marker/mechanism` genes + HumanNet-XC max-LLS —
  same method family as the v2.0 strict build. Coverage 245/271 diseases
  have genes; 26 zero-coverage rows measured [F].
- `sim_m2`/`sim_d2`: passthroughs of `M_MISIM`/`D_MESH`, verified elementwise
  (above). `M_MISIM` provenance: numerically agrees with MISIM 2.0 within
  source precision (maxdiff 4.9e-8, §2 of the post-pilot audit) —
  association-derived as a source-family fact; record-level label overlap
  unmeasured [F/I].
- `Y`, names: identical to reference (verified) [F].

## 3. Canonicalization audit

### 3a. Disease identifiers — MeSH UI merges, explicit evidence only

Rule (v2 policy, coordinator-revised 2026-11-13): group two names iff both
resolve to the same **unique** MeSH descriptor UI in `d2025.bin` (exact term
first, then unique squash-normalized entry-term match). **Grouping is a
precaution only** — the same descriptor can carry narrower (NRW-coded) or
related entry concepts, so same-UI is NOT proof of exact disease identity.
Each group's concept tier is resolved from the descriptor block's
relationship codes: `MH`/`EQV`/plain-entry ⇒ *same-concept-evidence*;
`NRW` or unmarked relationship ⇒ *precaution-only*. Unresolved/ambiguous →
singleton. Y rows/entities are never merged.

Measured: **all 271 disease names resolve to a unique UI**; **9 merge
groups**, all 2-index. Concept tiers from the parsed descriptor blocks
(`ENTRY`/`PRINT ENTRY` relationship codes):

| UI | idx1 | name A | idx2 | name B | rel codes | concept tier |
|---|---|---|---|---|---|---|
| D003110 | 2 | Colon Neoplasms | 178 | Adenocarcinoma, Colon | EQV + plain | same-concept-evidence |
| D001943 | 3 | Breast Neoplasms | 108 | Carcinoma, Breast | MH + plain | same-concept-evidence |
| D010911 | 8 | Pituitary Neoplasms | 195 | Pituitary Adenoma | MH + NRW | precaution-only |
| D013964 | 11 | Thyroid Neoplasms | 168 | Carcinoma, Thyroid | MH + plain | same-concept-evidence |
| D018281 | 66 | Cholangiocarcinoma | 110 | Intrahepatic Cholangiocarcinoma | MH + NRW | precaution-only |
| D016889 | 90 | Endometrial Neoplasms | 177 | Carcinoma, Endometrial | MH + plain | same-concept-evidence |
| D014376 | 113 | Tuberculosis | 118 | Mycobacterium tuberculosis Infection | MH + EQV | same-concept-evidence |
| D008268 | 131 | Age-Related Macular Degeneration | 185 | Macular Degeneration | NRW + MH | precaution-only |
| D012509 | 180 | Soft Tissue Sarcoma | 231 | Epithelioid Sarcoma | plain + plain | precaution-only (unmarked) |

So 6/9 groups have same-concept evidence at descriptor granularity; 3/9 are
precaution-only (one name is a coded narrower concept, or the relationship
is unmarked). Even the same-concept tier is a descriptor-level statement —
grouping remains a leak-control precaution, not an identity proof.

Ledger: `disease_ui_ledger.csv` (all 271 rows), `mesh_alias_evidence.json`
(merge evidence + decision rule), `canonical_group_ledger.csv` (3,699
multi-cell canonical groups).

**New fact vs v2.0:** grouped siblings' Y rows are **NOT identical** in v3.2 —
all 9 pairs carry different type-label patterns (e.g., D014376: types
{2,3,4} vs {0,2,3}). v2.0's merged aliases were exact duplicates; v3.2
records were split or curated separately across name variants. A
test-train crossing therefore risks evaluating a held-out record for a
disease whose same-descriptor sibling's records were seen in training —
a leak *risk under the precaution grouping*, scaled by the concept tier
above, not a proven same-association duplication [F/I].

### 3b. miRNA identifiers — no proven aliases, one ambiguity tier

Evidence check via GFF3 `hsa.gff3` `miRNA_primary_transcript`
Name→accession (`Derives_from`/`ID`) records:

- **0 accession-sharing pairs** among the 338 accession-resolvable names →
  no proven miRNA aliases [F].
- **26 ambiguous families, 80 rows** (`ambiguous_family_groups.csv`): a base
  name absent from GFF3 (e.g., `hsa-mir-16`) coexists in the 411-list with
  its own locus variants (`hsa-mir-16-1`, `hsa-mir-16-2`). The base name has
  no unique accession — it could denote either locus. Measured label
  overlap between a base row and each variant row: pair-set Jaccard
  ≈0.22–0.26 — partial record duplication is *possible* but unproven
  (same-family loci legitimately share associations) [F/I].
- **38 unresolved names** (`dead:`): no GFF3 candidates at all (naming-drift
  era names, e.g., `hsa-mir-92` while GFF3 carries `hsa-mir-92a-*`) —
  singletons, flagged [F].
- **8 expandable names** (`expand:`): base name matches ≥1 GFF3 locus
  variant(s), none of which are in the list — the expansion is
  **evidence-limited, not a uniquely identified precursor**: multiple loci
  may exist and the name does not select one; any sequence assignment must
  declare the union/locus choice as ambiguous [F/I].
- All 411 names are `hsa-` prefixed; no species ambiguity; no name
  duplicates [F]. Ledger: `mirna_group_ledger.csv`.

**Policy consequence:** under the strict v2 rule (merge only on explicit
same-entity evidence), ambiguous families stay split — a *declared residual
alias-risk tier*, not a proven leak. The conservative alternative (merge
families) is recorded but not adopted without a coordinator decision.

### 3c. Canonical-pair contamination and crossings in the ORIGINAL archived split

Archived split policy reproduced read-only: `make_splits(Y, seed=1234,
protocol='type', folds=5, val_frac=0.1)` — index-level positive-pair
shuffle (the policy every archived v3.2 run used).

Crossings are now **segment-labelled** (reconstructed `make_splits`
seed-1234 policy replay — no persisted author fold arrays exist):

| Policy tier | Multi-cell groups | Mixed pos/untyped | Untyped sib cells | test+train/fold | train+val/fold | test+val/fold |
|---|---|---|---|---|---|---|
| Strict (UI precaution groups; mirs singleton) | 3,699 | 616 | 616 | [59,53,51,59,48]=270 | [17,27,22,22,23]=111 | [6,3,8,8,7]=32 |
| Conservative sensitivity (fam merged) | — | 1,353 | 2,259 | ~[273,256,264,275,255] | not measured | not measured |

Ledger: `canonical_pair_crossings.csv` — per-cell segment label +
per-cell type labels.

Reading: ~54 canonical groups per fold have a test cell while a
same-descriptor sibling sits in train — every fold is affected; test+val
crossings (32 total) additionally blur model selection. The grouped split
is **more** necessary for v3.2, not less [F/I].

## 4. v3.2-specific negative-pool note

No binary-only association source exists for v3.2 (unlike v2.0's binary
`multi1` file) → the binary-untyped exclusion gate is **not applicable**
here; only the typed/alias-aware mask matters. Recorded, not assumed [F].

## 5. Prospective split — proposal only (identity + labels verified → permitted)

`audit_v32_identity/split_proposal_seed1234_v2.npz` + manifest
(`split_proposal_manifest_seed1234_v2.json`, npz sha256 `074649bc…`).

**v1 correction (coordinator flag):** v1's val carve took the first 10% of
*weight-sorted* train groups → identical val=721 in all 5 folds, i.e.,
deterministic largest-group selection — not a seeded random carve. v1
bytes preserved (`split_proposal_seed1234.npz` untouched); v2 fixes it:

- Canonical = (singleton miRNA row, disease-UI **precaution** group);
  folds = seeded-random shuffled whole groups → fewest-pairs fold;
  validation = seeded-random whole-group selection targeting ~10% of
  train **pairs** (pair-count target; group granularity gives ≈699 vs 721).
- Result: 8,550 groups → test 1,747 / **val 699** / train 6,289 per fold;
  0 group cross-fold violations; per-fold per-channel counts in manifest.
- **Reproducibility:** `faithful_repro/code/audit_v32_identity.py`
  regenerates every ledger + the proposal from hashed inputs; assertions
  cover Y/name/sim alignment with the ref npz, group disjointness, full
  test coverage of all 8,735 positive pairs, and no size-ranked
  validation. Determinism is checked by **independent replay**, not just
  re-read of stored outputs: shuffled key list and full fold group
  assignment replayed from a fresh stream; per fold the validation
  group selection replayed from its own `SEED*100+f` stream and required
  equal to the recorded val group IDs; val groups asserted a nonempty
  strict subset of that fold's train groups; train/val/test group sets
  asserted pairwise-disjoint with union = all canonical positive groups;
  and every segment array (test/val/train) rebuilt from the recorded
  group IDs must equal the stored fold array elementwise.
- Status: **labeled proposal** — no fitting, no promotion to primary;
  ambiguous families are sensitivity-only (not merged).

## 6. What remains unresolvable from inspected sources

- Whether `hsa-mir-16` (and the other 25 base names) denotes a specific
  locus — needs the HMDD v3.2 name→record mapping or author pipeline [U].
- Whether the SPLDHyperAWNTF snapshot is byte-equivalent to the paper's own
  tensor [I, lineage-only].
- Author fold assignments and whether the paper's evaluation dropped any
  type [U].
- MISIM 2.0 → evaluated-pair record overlap (needs MISIM source records,
  §5-step-3 of the post-pilot audit).

## 7. Recommended next review gate (exactly one)

**Coordinator review of `split_proposal_seed1234_v2.npz` + this audit +
`code/audit_v32_identity.py`.** Decisions already applied in this revision:
strict mir policy retained (families = sensitivity tier only); disease
grouping relabeled as a precaution with per-group concept tiers; val carve
is seeded-random, not size-ranked. Remaining for the gate: accept the v2
proposal (or specify the sensitivity variant), and confirm `sim_m1`
strict-GFF3 rebuild as prerequisite for any v3.2 fidelity claim. No
training/fitting is requested or pending.
