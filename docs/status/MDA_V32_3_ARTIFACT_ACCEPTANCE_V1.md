# MDAv3.2-3 artifact acceptance contract v1

**Registered:** 2026-07-31
**Scope:** HMDD v3.2 only
**Decision owner:** Sol at S1B/R4
**Mechanical executor:** Terra at T1A
**Target claim:** the exact `MDAv3.2-3` benchmark used by SPLDHyperAWNTF

This contract is deliberately stricter than count matching. Terra may mark a candidate
`R4-ready`, `quarantined`, or `rejected-for-class-A`; Terra must not declare class A.

## 1. Candidate bundle

Each candidate is an immutable bundle containing:

1. the downloaded bytes, or an archive snapshot/commit that deterministically yields them;
2. the entity names or an unambiguous ID-to-name mapping;
3. a representation that preserves all five functional labels per miRNA–disease pair;
4. discovery and custody records conforming to
   `MDA_V32_3_SEARCH_LADDER_V1.json`;
5. byte size and SHA-256 for the container and each inspected member;
6. the exact parser name/version and an inspection report.

Do not modify a candidate in place. Derived extracts receive their own hash and retain a pointer
to the source container hash.

## 2. Canonical semantic representation

The acceptance tests operate on typed triples, not a scalar association matrix.

- Required type vocabulary:
  `Circulation`, `Epigenetics`, `Target`, `Genetics`, `Tissue`.
- A source-specific token may map to one of those values only when the mapping is explicit in
  the artifact, its generating code, or a contemporaneous source. An inferred mapping is
  quarantined for Sol review.
- Preserve original entity strings and source IDs. Do not apply alias merging, case folding,
  punctuation repair, or hand-written corrections during acceptance.
- Remove only format syntax: enclosing quotes, field separators, record terminators, and
  explicitly documented index columns. Trimming or Unicode normalization that changes an entity
  key must be reported as a collision, not silently accepted.
- A tensor or collection of five binary matrices is expanded to one triple for each asserted
  type. A scalar cell that stores only one type cannot prove multi-label identity.
- The semantic digest input is the sorted list of JSON arrays
  `[mirna_source_key,disease_source_key,canonical_type]`, one compact UTF-8 JSON value per LF
  terminated line. Sorting is by UTF-8 byte order of the three fields. The digest is SHA-256 of
  those bytes.
- Entity digests use the same rule with sorted
  `[source_key,source_name]` arrays. If names are the keys, repeat the same value in both fields.

Terra records both raw-byte hashes and these semantic digests. Semantic canonicalization never
replaces the raw candidate.

## 3. Exact tests

Every test has one of `PASS`, `FAIL`, or `QUARANTINE`. Missing evidence never passes.

| ID | Test | Exact pass condition | Failure disposition |
|---|---|---|---|
| C01 | Retrieval identity | Final URL/archive ID/commit and UTC retrieval time are recorded; redirects are recorded; downloaded size and SHA-256 are present. | `FAIL` |
| C02 | Primary custody | The bytes come from a publisher supplement, an attributable original-author repository/archive, or an immutable historical capture of either. | `FAIL` for class A; retain as lead |
| C03 | Independent-copy custody | C02 passes, or, when C02 is unavailable, an attributable, timestamped, immutable mirror is bit-identical to a contemporaneous primary checksum or matches a second independently controlled contemporaneous copy at raw or semantic entity/triple digest level. Shared upstream provenance must be disclosed and does not count as independence. | `QUARANTINE` unless one alternative passes |
| C04 | Custody continuity | Every transformation from container to inspected member is recorded by parent hash, member path, extraction tool, member size, and member hash. | `FAIL` |
| C05 | Legal handling | Access method and redistribution status are recorded. Unknown redistribution terms allow inspection and hash recording but forbid committing candidate bytes. | `QUARANTINE` for distribution, not identity |
| F01 | Parse determinism | Two clean parses of the same bytes with the registered parser yield identical entity/triple semantic digests. | `FAIL` |
| F02 | Complete entity mapping | Every association-side ID resolves to exactly one miRNA and one disease key; no orphan, duplicate-ID/conflicting-name, or out-of-range reference exists. | `FAIL` |
| F03 | Five-type reachability | All and only the five required functional types are represented after an evidence-backed token mapping. | `FAIL`; inferred token mapping is `QUARANTINE` |
| F04 | Multi-label preservation | The source can represent two or more types for one pair without overwrite; expansion and reassembly preserve each asserted type. | `FAIL` |
| I01 | miRNA cardinality | Exactly 411 distinct source miRNA keys participate in or are explicitly included by the benchmark entity list. | `FAIL` |
| I02 | Disease cardinality | Exactly 271 distinct source disease keys participate in or are explicitly included by the benchmark entity list. | `FAIL` |
| I03 | Type cardinality | Exactly five declared functional types exist and map bijectively to the required vocabulary. | `FAIL` |
| I04 | Typed-triple cardinality | Exactly 11,748 distinct `(miRNA,disease,type)` triples exist after format-only expansion. | `FAIL` |
| I05 | Duplicate freedom | No two source records expand to the same canonical typed triple. If duplicates exist, both pre- and post-dedup counts are recorded, but the candidate fails the class-A no-duplicate condition. | `FAIL` |
| I06 | Pair consistency | Each distinct pair has one to five types; its labels survive triple→multi-hot→triple round-trip exactly. | `FAIL` |
| I07 | No manual fitting | Entity inclusion is wholly determined by recovered artifact bytes/code; no hand selection, truncation, or parameter tuning was used to reach 411/271/11,748. | `FAIL` |
| I08 | Reference identity | If more than one credible copy exists, raw hashes match or semantic entity and triple digests match with every byte-level difference explained. | `QUARANTINE` if copies conflict |
| R01 | Rebuild identity | A claimed reconstruction matches the recovered reference's miRNA, disease, and typed-triple semantic digests exactly. Matching dimensions/counts alone fails. | `FAIL` for exact rebuild claim |
| R02 | Provenance sufficiency | The bundle identifies the artifact's relationship to SPLDHyperAWNTF and HMDD v3.2 with an inspectable contemporaneous source. | `FAIL` for class A |

## 4. Decision algorithm

1. A candidate is **rejected for class A** when any of C01, C04, F01–F04, I01–I07, or R02
   is `FAIL`. It may still be useful evidence for a class-B/C reconstruction.
2. A candidate is **quarantined** when no mandatory test fails but custody, type-token semantics,
   redistribution, duplicate semantics, or conflicting copies require Sol judgment.
3. A candidate is **R4-ready** only when C01, C04, C05, F01–F04, I01–I08, R02, and at least one
   of C02/C03 pass. R01 is also mandatory when the candidate is claimed to be a rebuild rather
   than an original artifact.
4. Only Sol may convert an R4-ready candidate into class A after content inspection and
   chain-of-custody review. S1B may advance it to R4; S1B does not itself assign the final
   immutable dataset ID.

## 5. Automatic rejection patterns

The following never establish exact recovery:

- a filename containing `411`, `271`, `11748`, `v3.2`, or `-3`;
- a `411 × 271` binary/scalar matrix without entity mappings and five-type preservation;
- final dimensions/counts obtained by degree tuning or manual entity selection;
- any local `v3.2_processed`, `v3.2_wang`, `v3.2_wang_dense`, or
  `v3.2_wang_multilabel` derivative;
- a file whose only provenance is a current local path, screenshot, narrative, or uninspectable
  search snippet;
- a post-publication derivative with no link to original bytes or generating code;
- an archive with corrupt, encrypted, or missing members;
- a parser that silently drops rows, collapses repeated pairs, invents aliases, or substitutes
  GIP/duplicated matrices while preserving biological similarity labels;
- count agreement without entity-set and typed-triple identity.

## 6. High-risk code-archaeology targets

Sol review is required when Terra finds:

1. generation code for `MDAv3.2-2` or `MDAv3.2-3`, especially the non-monotone
   `610 × 333 / 11,182` versus `411 × 271 / 11,748` claims;
2. ambiguous use of “association” that may mean source rows, unique pairs, or typed triples;
3. category-prefix logic for the five HMDD files, including therapeutic, transcription-factor,
   or lncRNA targets;
4. MeSH snapshot/version, category-C selection, descriptor aliases, or tree-number rules;
5. MISIM 2.0 entity coverage, name normalization, or functional-similarity construction;
6. one-pass/iterative, ordered/simultaneous, pair-degree/triple-degree pruning;
7. row-order-dependent duplicate or type-priority behavior;
8. serialized `.mat`, `.npy`, `.npz`, pickle, spreadsheet, or archive objects whose axes,
   index base, type order, or entity maps are implicit;
9. conflicting forks, commit histories, timestamps, hashes, or licenses;
10. an apparent exact artifact lacking any canonical entity list.

These targets are review triggers, not permission to broaden the search beyond the registered
ladder or to fit preprocessing to published counts.
