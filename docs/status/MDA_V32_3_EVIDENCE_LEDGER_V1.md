# MDAv3.2-3 evidence ledger v1

**Frozen:** 2026-07-31
**Scope:** S0 and S1A only; HMDD v3.2 only
**Dataset identity:** unknown
**Search state:** registered, not executed
**P3 state:** blocked and untouched

This ledger distinguishes locally verified observations from publication claims, inferences,
contradictions, and open questions. A local file, generated derivative, matching dimension, or
matching count is not a trusted reference artifact.

## 1. Repository state at S0

| Field | Value |
|---|---|
| Branch | `codex/local-runner` |
| HEAD | `6a5233c576f3fc1880b628df120020d66b6be4b4` |
| Commit time | `2026-07-26T20:34:09+07:00` |
| Commit subject | `research tìm cách khôi phục data v3.2` |
| Index | clean |
| Tracked worktree | clean |
| Untracked pre-existing file | `~$oCao_DHGCMDA.docx` |

The untracked file is a Microsoft Word lock file and is outside this work. It was not deleted,
moved, hashed, or added. S0/S1A did not create a branch, run training, execute P3, contact
authors, or change historical `logs/`/`results/`.

## 2. Locked decisions

The following are constraints, not questions reopened by this wave:

- R1: exact artifact first; deterministic reconstruction is labelled B/C unless identity is
  proved at entity/triple level.
- S1: combined artifact search stops at 8–10 active hours.
- M2: `paper_compatible_v32` and `honest_v32_multilabel_fullcv` remain separate protocols.
- Multi-label: a pair may carry several of five functional types.
- D1: handoff is portable Git code/manifests/scripts/tests; raw or recovered data is not
  committed by default.
- HMDD v4, author contact, P3 execution, and historical-output replacement are out of scope.

## 3. Evidence statements

### 3.1 Verified facts

| ID | Statement | Evidence |
|---|---|---|
| V01 | Seven HMDD-v3.2-named local source files exist: one workbook, `v3_alldata.txt`, and five type files. Their byte sizes and hashes are frozen in section 5. | Read-only inventory, 2026-07-31 |
| V02 | The five type files contain 24,360 rows excluding headers. Ordinal, case-sensitive tuple counting yields 18,085 unique raw typed triples, 6,275 duplicate typed rows, 14,218 unique raw pairs, 2,785 multi-type pairs, 1,051 raw miRNA names, and 758 raw disease names. | Independent Terra recomputation with `csv.DictReader`; agrees with recovery-plan lines 59–80 |
| V03 | Minimal `strip + lowercase` produces 18,084 typed triples and 14,217 pairs; 865 typed triples lack a MeSH ID and 46 MeSH IDs map to multiple normalized disease names. | Independent recomputation; agrees with recovery-plan lines 82–85 |
| V04 | The earlier 23,732-row local claim omitted valid target subcategories. The current five-file row count is 24,360. | Recovery-plan lines 87–90; current counts |
| V05 | `v3.2_processed`, `v3.2_wang`, and `v3.2_wang_dense` each contain one row per unique pair and therefore no multi-type pairs. `v3.2_wang_multilabel` has 16,341 triples, 12,534 pairs, and 2,731 multi-type pairs but has the wrong entity/count identity. | Independent CSV inspection; section 4 |
| V06 | An additional local legacy derivative, `v3.2_filtered_495m383D`, contains 3,938 rows/pairs, 420 miRNA IDs, 180 disease IDs, and no multi-type pairs. | Independent CSV inspection |
| V07 | `prepareData.py::read_association_csv` assigns `association_matrix[i,j] = atype`; repeated pair rows are last-row-wins before splitting. | `prepareData.py:49`, `prepareData.py:76` |
| V08 | `prepareData.py::preprocess_indices` enumerates only `type1` through `type4`; Tissue is not represented in its returned type-index map. | `prepareData.py:232`, `prepareData.py:252-273`, `prepareData.py:348-353` |
| V09 | The non-five-class predictor applies softmax across type logits, while the multi-label loss applies ordinary binary cross entropy to those probabilities. This is not independent-logit BCE-with-logits. | `hetero_model.py:478-490`, `hetero_model.py:612-621`; `main_experiments_hetero1.py:222-264` |
| V10 | Several purportedly distinct local views are byte-identical: processed `D_SSM1/D_SSM2`, processed `M_FSM/M_GSM`, Wang `D_SSM1/D_SSM2`, and Wang `M_FSM/M_GSM`; Wang-multilabel reuses the same corresponding hashes. | Section 5 hashes |
| V11 | `preprocess_v32_wang.py` explicitly uses priority overwrite for a scalar association matrix and labels duplicated disease/miRNA views as temporary or placeholder reuse. | `preprocess_v32_wang.py:8-10`, `preprocess_v32_wang.py:69-74` |
| V12 | `build_v32_dense.py` uses iterative minimum-association threshold 7 to approximate target shape/density; it is not recovered benchmark generation evidence. | `build_v32_dense.py:4-12`, `build_v32_dense.py:20-45` |
| V13 | Registry record `p3-multilabel-clean` remains blocked because the predictor's softmax type outputs are incompatible with independent multi-label BCE. | `docs/status/registry.json`, record `p3-multilabel-clean` |

### 3.2 Publication or prior-review claims

These statements have not been independently re-established from external sources during S0/S1A;
the registered search has not started.

| ID | Claim | Current source | Evidentiary status |
|---|---|---|---|
| P01 | SPLDHyperAWNTF documents `MDAv3.2-3` as 411 miRNAs, 271 diseases, five types, and 11,748 positive typed triples. | Recovery-plan lines 35–36 and 103–114 | paper claim |
| P02 | SPLDHyperAWNTF associates threshold `>=2` with `MDAv3.2-2 = 610 × 333 / 11,182` and threshold `>=3` with `MDAv3.2-3 = 411 × 271 / 11,748`. | Recovery-plan lines 37–40 | paper claim |
| P03 | Public DHGCMDA history contains v2.0 data rather than the target v3.2 artifact. | Recovery-plan line 43 | prior-review claim; recheck in T1A |
| P04 | The former original-project repository is `Ouyang-Dong/SPLDHyperAWNTF_`. | Recovery-plan primary references | prior-review claim; recheck in T1A |

### 3.3 Inferences

| ID | Inference | Basis | Falsification condition |
|---|---|---|---|
| N01 | None of the current local derivatives is class-A evidence. | Their entity/triple counts differ from P01, several collapse labels, and none has qualifying custody. | A recovered primary artifact has matching semantic entity/triple digests and proves one derivative is bit/semantically identical. |
| N02 | Ordinary nested pruning of one fixed triple universe cannot explain a stricter entity threshold producing more retained triples. | P02 reports fewer entities but 566 more associations for `-3` than `-2`. | Recovered code/data shows the variants use different input universes, count grains, mappings, or another documented non-nested operation. |
| N03 | Local filenames do not establish independent biological views. | Byte-identical matrices and explicit placeholder/GIP reuse. | Provenance-bearing source material shows the identical bytes are intentional valid biological views under the claimed definitions. |
| N04 | The exact artifact is not presently established, but absence has not been proved. | No local candidate meets the acceptance contract; T1A is unexecuted. | T1A finds an R4-ready candidate. |
| N05 | Matching 411/271/11,748 alone is insufficient to distinguish exact recovery from fitted reconstruction. | Multiple pruning/mapping choices can affect counts; no reference entity/triple digest exists locally. | A primary artifact or independently verified copy supplies canonical entity/triple identity. |

### 3.4 Contradictions and disputed claims

| ID | Dispute | Evidence on each side | Required resolution |
|---|---|---|---|
| D01 | The published threshold/count story is non-monotone. | `>=3` has fewer entities but more associations than `>=2` under P02. | Recover generating code/artifacts or document distinct universes/count grains; do not call it a typo without evidence. |
| D02 | Five-file raw total was previously stated as 23,732. | Current ordinal count is 24,360; missing target subcategories explain the shortfall. | Treat 23,732 as superseded. |
| D03 | Local Wang files are labelled as distinct similarity views. | Corresponding hashes are identical and preprocessing comments call the views reused/placeholders. | Do not cite filenames as biological provenance; recover actual source definitions. |
| D04 | `v3.2_wang_multilabel` is multi-label but could be mistaken for the target benchmark. | It preserves 16,341 triples on 713×447, not 11,748 on 411×271. | Keep `wrong benchmark` label unless entity/triple identity evidence overturns it. |
| D05 | `v3.2_wang_dense` approximately matches target density/shape and could be mistaken for recovery. | It is explicitly produced by threshold 7 and is 385×275/10,888. | Reject for class A; retain only as a legacy diagnostic derivative. |

### 3.5 Open questions

| ID | Question | Gate affected |
|---|---|---|
| Q01 | Does a primary, archived, or independently verified copy of the exact artifact exist? | R1/R4 |
| Q02 | What does the paper's 11,748 count denote: raw rows, unique pairs, or unique typed triples? | R1/R4 |
| Q03 | What exact five-type source-token mapping and duplicate semantics were used? | R2/R3 |
| Q04 | Which historical MeSH snapshot, descriptor aliases, category-C rule, and tree-number policy were used? | R2/R3 |
| Q05 | Which MISIM 2.0 snapshot, entity coverage, aliases, and similarity data were used? | R2/R3 |
| Q06 | Were degree counts measured over pairs or typed triples, and were filters one-pass/iterative and ordered/simultaneous? | R3/R4 |
| Q07 | Are `-2` and `-3` derived from the same input universe? | R1/R4 |
| Q08 | What are the license/redistribution terms for any recovered archive and its dependencies? | R1/R2/D1 |
| Q09 | Can every axis/type order in a candidate binary be tied to explicit entity/type mappings? | R1/R4 |

## 4. Recomputed baseline counts

### 4.1 HMDD v3.2 source views

| File/view | Rows excluding header |
|---|---:|
| `v3_alldata.txt` | 32,281 |
| `v3_circulation.txt` | 4,120 |
| `v3_epigenetics.txt` | 644 |
| `v3_genetics.txt` | 2,302 |
| `v3_target.txt` | 9,064 |
| `v3_tissue.txt` | 8,230 |
| Five type files combined | 24,360 |

Ordinal case-sensitive five-file statistics:

| Measure | Count |
|---|---:|
| Unique typed triples | 18,085 |
| Duplicate typed rows | 6,275 |
| Unique pairs | 14,218 |
| Multi-type pairs | 2,785 |
| Distinct raw miRNA names | 1,051 |
| Distinct raw disease names | 758 |

The count grain is explicit. “Duplicate typed rows” is `24,360 - 18,085`. Multi-type pairs are
unique raw `(miRNA,disease)` keys associated with more than one of the five file-derived types.

### 4.2 Existing local derivatives

| Dataset | Rows / unique triples | Unique pairs | miRNA IDs | Disease IDs | Multi-type pairs | Classification |
|---|---:|---:|---:|---:|---:|---|
| `v3.2_processed` | 13,748 | 13,748 | 722 | 612 | 0 | collapsed single-label |
| `v3.2_wang` | 12,534 | 12,534 | 713 | 447 | 0 | collapsed single-label |
| `v3.2_wang_dense` | 10,888 | 10,888 | 385 | 275 | 0 | density approximation |
| `v3.2_wang_multilabel` | 16,341 | 12,534 | 713 | 447 | 2,731 | multi-label, wrong benchmark |
| `v3.2_filtered_495m383D` | 3,938 | 3,938 | 420 | 180 | 0 | filtered legacy derivative |

## 5. Local v3 raw/derived data inventory

Hashes are lowercase SHA-256 over the exact local bytes. “Raw” describes the local role; it does
not by itself prove official source custody.

| Path | Bytes | SHA-256 | Classification |
|---|---:|---|---|
| `HMDD_data/MDAv3.2/hmdd_v3.2_raw.xlsx` | 1,332,102 | `6f6bfc311dc4c2e5ed2b870de57d6305f1c3f60ebb9916bfaf69b6069dba801a` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_alldata.txt` | 8,636,722 | `a496591171ef3eb00ce03c44e22bf5e70497b7d9a41b6d9752161011042b89dc` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_circulation.txt` | 1,178,550 | `d91749df3b5e01e7cfe55e9bc2124150b4f6bff84bd19aa628f910a4d77177a7` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_epigenetics.txt` | 179,577 | `8ee034353f7a75d06ec40ed128e1c45713e7586121cac2f9f60ffa18c1b7aea2` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_genetics.txt` | 637,159 | `9b11d8689c192d0610d5a4ee316e5885023370ec8783878cfa9f4d32b8058805` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_target.txt` | 2,439,628 | `a4a27fdf50aa4df6cc2f58d7453f5e911d3687a5e839d22beae09af43f7a39d8` | raw HMDD v3.2 |
| `HMDD_data/MDAv3.2/v3_tissue.txt` | 2,272,786 | `753d0349ecf0bd900dbed9e3c8de901af70f9ed75f1f33f6440d2e13a769110d` | raw HMDD v3.2 |
| `v3.2_processed/D_SSM1.txt` | 3,393,578 | `c7f0362dea0cd90b7c8ed5299d0e1922ada46d2610427ccdfe15a41b36f62ce3` | collapsed derivative |
| `v3.2_processed/D_SSM2.txt` | 3,393,578 | `c7f0362dea0cd90b7c8ed5299d0e1922ada46d2610427ccdfe15a41b36f62ce3` | collapsed derivative |
| `v3.2_processed/disease name.xlsx` | 17,796 | `24c4609ddb8e9e15133a52e263c2ff6c0fd4862abe24a6041f7e3d774d9f44d8` | collapsed derivative |
| `v3.2_processed/M_FSM.txt` | 4,692,278 | `7f922c2ea3e6968109fe4c07434afdee2be12a853fd917421f1e7bbd91125ce3` | collapsed derivative |
| `v3.2_processed/M_GSM.txt` | 4,692,278 | `7f922c2ea3e6968109fe4c07434afdee2be12a853fd917421f1e7bbd91125ce3` | collapsed derivative |
| `v3.2_processed/miRNA name.xlsx` | 15,264 | `af765999c74b4d6274503edb7884f21843edfa6b090ae1a740ee590d02840768` | collapsed derivative |
| `v3.2_processed/multi_all_mirna_disease_pairs_without_negative.csv` | 145,754 | `1ed5e5b32e9c70fffb1b90d1a21a265b7d046e2a1122aeaa0c9a43164d79029a` | collapsed derivative |
| `v3.2_wang/D_SSM1.txt` | 1,798,728 | `1180b1c293634d1c2a11f9a7fc9f74c9f99f5e58f81982b92cc2428c82596e73` | Wang collapsed derivative |
| `v3.2_wang/D_SSM2.txt` | 1,798,728 | `1180b1c293634d1c2a11f9a7fc9f74c9f99f5e58f81982b92cc2428c82596e73` | Wang collapsed derivative |
| `v3.2_wang/disease name.xlsx` | 14,566 | `f3532a2649b1eb7483c31f46c65e7303a3536abbf4422d92778c4473cdc5f31a` | Wang collapsed derivative |
| `v3.2_wang/M_FSM.txt` | 4,576,034 | `99f3665ba8516ea61ea33d04c5276c02490c59354a27b33f7497627237d3c68e` | Wang collapsed derivative |
| `v3.2_wang/M_GSM.txt` | 4,576,034 | `99f3665ba8516ea61ea33d04c5276c02490c59354a27b33f7497627237d3c68e` | Wang collapsed derivative |
| `v3.2_wang/miRNA name.xlsx` | 15,239 | `4b64df02d21820a4f9ada0710c92f74f8fa8224344193ac34ff27d4bed49d7fe` | Wang collapsed derivative |
| `v3.2_wang/multi_all_mirna_disease_pairs_without_negative.csv` | 133,302 | `7428119082ad7fa4d588c484fc38ab6700640e0234ffaa003781e0d9e7281ac7` | Wang collapsed derivative |
| `v3.2_wang_dense/D_SSM1.txt` | 680,900 | `897969dc86e54128915d03f9446c654dbb943c50397e3a363880ab7c61b8494e` | Wang dense derivative |
| `v3.2_wang_dense/D_SSM2.txt` | 680,900 | `897969dc86e54128915d03f9446c654dbb943c50397e3a363880ab7c61b8494e` | Wang dense derivative |
| `v3.2_wang_dense/M_FSM.txt` | 1,334,410 | `1938b429d239fabc31e44fb1991964d2be12633f42897720be178e9794ecac18` | Wang dense derivative |
| `v3.2_wang_dense/M_GSM.txt` | 1,334,410 | `1938b429d239fabc31e44fb1991964d2be12633f42897720be178e9794ecac18` | Wang dense derivative |
| `v3.2_wang_dense/multi_all_mirna_disease_pairs_without_negative.csv` | 112,246 | `cfe5da6125a457e0f1c89a0c549c6d303e33e56ef79f88fb243857c2e3a40ef2` | Wang dense derivative |
| `v3.2_wang_multilabel/D_SSM1.txt` | 1,798,728 | `1180b1c293634d1c2a11f9a7fc9f74c9f99f5e58f81982b92cc2428c82596e73` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/D_SSM2.txt` | 1,798,728 | `1180b1c293634d1c2a11f9a7fc9f74c9f99f5e58f81982b92cc2428c82596e73` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/disease name.xlsx` | 14,566 | `58b2809a6bdc514fa3bae2bb694cfa1ca33b7744d2b4f4740146711c7f3c92ef` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/M_FSM.txt` | 4,576,034 | `99f3665ba8516ea61ea33d04c5276c02490c59354a27b33f7497627237d3c68e` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/M_GSM.txt` | 4,576,034 | `99f3665ba8516ea61ea33d04c5276c02490c59354a27b33f7497627237d3c68e` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/miRNA name.xlsx` | 15,239 | `5826b09646bf35b95304c201432b8681346b38555a1989e0dc2dc90d01541379` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/multi_all_mirna_disease_pairs_without_negative.csv` | 173,813 | `0589daa72d4a42eb964d13cccaec9a18aa2dea5bc67d4dfa3bddc3a548b199a2` | multi-label wrong benchmark |
| `v3.2_wang_multilabel/target_multilabel.npy` | 6,374,348 | `52f79e02b82cbb20d2965d938df9e244527345d6e47d8f9860fe69ef3dcdd522` | multi-label wrong benchmark |
| `v3.2_filtered_495m383D/D_GSM.txt` | 2,687,113 | `0961957bfa7b8b0c5046d7237cb3f6f68bc498ef671f3d2268f05a0f4dabeff1` | filtered legacy derivative |
| `v3.2_filtered_495m383D/D_SSM1.txt` | 828,771 | `aa46a442457a52575d57bf95d9389b1dc29a8a78a1ba9513a5bc1ec9f88cfa88` | filtered legacy derivative |
| `v3.2_filtered_495m383D/D_SSM2.txt` | 832,225 | `c923d664928973463811fd0e911da0f40003db3abb2f64fd84854bebc9db9efa` | filtered legacy derivative |
| `v3.2_filtered_495m383D/disease name.xls` | 52,736 | `ad055c1dc9d8c2c91148f7ab32c8fc92cf4a0afb444bf079c7b7ca71afcf5f4e` | filtered legacy derivative |
| `v3.2_filtered_495m383D/M_FSM.txt` | 1,290,827 | `f241f0c00734b8803d16c666266a26cb699d34721ef5746e9db1953ddb235f80` | filtered legacy derivative |
| `v3.2_filtered_495m383D/M_GSM.txt` | 4,476,587 | `d48b124ada07699c6c8665cc4b43f37c95db43a462b71cde0b6792532b22429a` | filtered legacy derivative |
| `v3.2_filtered_495m383D/miRNA name.xls` | 55,808 | `47886b8b5886d2505791084e2802a5804a3cc3982562215d079eaf6c1b197948` | filtered legacy derivative |
| `v3.2_filtered_495m383D/multi_all_mirna_disease_pairs_without_negative.csv` | 40,338 | `a4de379bf1d659e12babc1dfa8e18edc7d051751309d3aec92181be52aa9fbce` | filtered legacy derivative |
| `v3.2_filtered_495m383D/multi_all_mirna_disease_pairs_without_negative_forcasestudy.csv` | 40,338 | `a4de379bf1d659e12babc1dfa8e18edc7d051751309d3aec92181be52aa9fbce` | filtered legacy derivative |

Historical v32-named log/result artifacts are inventoried separately in section 6 because they
are experimental evidence, not candidate benchmark data. Their presence never establishes
dataset identity.

## 6. Historical v32 log/result inventory

This section is populated from a read-only path/size/SHA-256 pass. Historical files remain
unchanged and retain their existing protocol interpretation.

Inclusion rule: recursively enumerate files under `logs/` and `results/`, then apply the
case-insensitive full-path regex `(?i)(v32|v3\.2)`. This found 36 files: 20 logs and 16 results.

| Path | Bytes | SHA-256 | Classification |
|---|---:|---|---|
| `logs/b3_v32_ce_ep120f3.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | historical log |
| `logs/b3_v32_ce_ep60f2.log` | 14,202 | `9870982e15cb51ae06cee6793118f29eec9bef51e001e53d65e4ce77b3a47e89` | historical log |
| `logs/b3_v32_ldam_ep60f2.log` | 14,207 | `7e8c2b94d766ead5ce9e8c97fe3f04015c3734daaa28fac5a41a1abfa60685cf` | historical log |
| `logs/b3_v32_logitadjust_ep60f2.log` | 14,204 | `bf56c7a682df17232a39d68b12cd6823dd8203a7b08b169f866df9796585b9a7` | historical log |
| `logs/conformal_v32_analysis.log` | 1,003 | `9678c9f6ba464567cfb04a313e9abac15711456e358bac585d5f6218b66ad51f` | historical log |
| `logs/conformal_v32_dump.log` | 46,223 | `2b9e4ae8eb63b4af85787bc12b7aaad258f39ed41c17eda7818c2cd1af702d48` | historical log |
| `logs/council_C1_v32_fb_softmax5.log` | 32,153 | `a1ede86e434f0f5f29a4729782af03aea8fbb537238946d3dfcb7f5a87ac9c06` | historical log |
| `logs/j1_v32_full_bilinear.log` | 32,954 | `e89798da2435893b13e6983de7acd82cd43f0cc38b9cdc69d3418651d0c1f315` | historical log |
| `logs/preprocess_v32_wang.log` | 1,411 | `99563714e2186d3e53ad3e72032d16b7d1241548a397dc1316ca6c650ab5ca0a` | historical log |
| `logs/repro_v32_honest_linux.log` | 32,047 | `c2f3f9cab06f78e0a34e77540fe6a467384ddb0b3305d0da1b9463664261368f` | historical log |
| `logs/tdrc_v32.log` | 54,188 | `95164707a808ca19b098f92e2e41ce7a583918743f357cb608fde1594c3e1511` | historical log |
| `logs/v32_ablation_no_avf.log` | 16,406 | `b8172379f91432d554b4302acde4bfede50483036b9add1a9e92ebb8a1a48334` | historical log |
| `logs/v32_baseline.log` | 65,706 | `9e49cd629b1029b4b484b24ad6955d381a404bb561fa0a533bfb63a86b08b644` | historical log |
| `logs/v32_dense_fullbilinear.log` | 50,034 | `3637788b7360472d79881461c515d84f8aee9179c0160088f2e1aad8a8f1eeac` | historical log |
| `logs/v32_orchestrator.log` | 4,730 | `77ce29d29cc2d7bf80afba98ee77f0d94b7ed2d2d1f12cc7bccb895ba88a3aa4` | historical log |
| `logs/v32_wang_baseline.log` | 16,501 | `0a20bbe40f7f22d536b75596b90e788b6a33a03785477910ec90655e6c3faa26` | historical log |
| `logs/v32_wang_correct_fullbilinear.log` | 50,157 | `f6e3e7c9954acd85af72f5392e628b0c62b415bdf19ce45e714ddb28c084e914` | historical log |
| `logs/v32_wang_correct_fullbilinear_650.log` | 65,931 | `829ff521bf0c2afa2859cd4659b55e4d831666d5123152293c058a6bc97264f6` | historical log |
| `logs/v32_wang_correct_metric.log` | 48,299 | `7b793a567f49fffffb3ca37f2329448b74f0697c952a9085d78116ed7d52894b` | historical log |
| `logs/v32_wang_multilabel_baseline.log` | 16,426 | `3bc487c9914e00a9a9eaac2144a5068ac66afe047b8b815054b17f42072f9788` | historical log |
| `results/a16_v32_loss_reweight.json` | 809 | `4706879413cf64874148297541275e576d63d0283a931a1cde5eedf2e29b465b` | historical result |
| `results/baseline_TDRC_v32.json` | 496 | `f3c11e631f332b056501a26725c8710ec56edab0a962c4ca983c1bb363526ebd` | historical result |
| `results/conformal/v32_conformal_report.json` | 7,340 | `507fcace59af0548d0485060d93712a6a3fdbd437855ff58b16ab39fbc62c779` | historical result |
| `results/conformal/v32_dump/fold0.npz` | 109,776 | `2dd40795ea0fa26c722ccc040fcc4b8bc24c75e09e0425eee16a5a64660a9b99` | historical result |
| `results/conformal/v32_dump/fold1.npz` | 109,776 | `1fec6f2002e41a92f0724254a3443705d6b4655c2d54b38d89ef7522f49b8458` | historical result |
| `results/conformal/v32_dump/fold2.npz` | 109,776 | `7d70ec179e0f717efef721fc5e34bf5cf97e356ca5550076e4dd418974cbfe1a` | historical result |
| `results/conformal/v32_dump/fold3.npz` | 109,776 | `5993d642a56d7510029df8ed069faa8105c25c763aee463096b59c6c23a4135b` | historical result |
| `results/conformal/v32_dump/fold4.npz` | 109,776 | `50a4d2cf7034ab4e35d21a392aca38401ee4fffdd0225ae46601fc9e3e43bc3c` | historical result |
| `results/council_C1_v32_fb_softmax5.json` | 383 | `3d7254dbd3c8a19b9a11f9394b141c85aac9be341cad216feb6894807f863db5` | historical result |
| `results/repro_v32_honest_linux.json` | 390 | `18f3aa7cf440e4fbfb8b230f17911643c8284b699a40448234efd4d072a1837f` | historical result |
| `results/v32_baseline_partial.json` | 643 | `66b70d2b7eb627d19f1951dee10958f664165870dcab65783c391a4ca9ec9b89` | historical result |
| `results/v32_dense_fullbilinear.json` | 1,170 | `b843da8fcdb70470e4a76ca52ae904ad2c785176ec6b2644839b07fbddacfc66` | historical result |
| `results/v32_wang_baseline_partial.json` | 971 | `884fcdccc72e55bcd38b70efa91db42e924abacb45b3ff5a22028a159da82b13` | historical result |
| `results/v32_wang_correct_fullbilinear.json` | 986 | `17af163e2ee4da82aaab8e20d26c71eb807245b5f182df1e54328f6f8fb39be2` | historical result |
| `results/v32_wang_correct_metric_baseline.json` | 969 | `5d68018d58c178e35c15a5ac4bae62937ed28cbefffa598ce8c8f531441b01de` | historical result |
| `results/v32_wang_multilabel_baseline.json` | 619 | `da615a94e2a85fd290b056cd1308819b1774d3d0c7b2d6e64638ffe01f45ced8` | historical result |

## 7. S1A registration and stop-budget account

- Acceptance contract:
  `docs/status/MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md`.
- Registered finite ladder and query/candidate ledger schema:
  `docs/status/MDA_V32_3_SEARCH_LADDER_V1.json`.
- Terra execution handoff:
  `docs/status/MDA_V32_3_TERRA_T1A_HANDOFF.md`.
- Machine registry record:
  `v32-artifact-search-s1-20260731`.

Budget accounting is conservative:

| Work | Charge/status |
|---|---:|
| Sol S0 | 1.0 active hour charged |
| Sol S1A | 2.0 active hours charged against S1 |
| Terra T1A | 0.0 active hours used; cap 5.5 |
| Sol S1B | 0.0 active hours used; 1.5 reserved |
| Combined S1 | 2.0 of 8–10 active hours charged; 6–8 remain |

No artifact search was executed in S1A. The next authorized action is Terra T0/T1A, followed by
Sol S1B. R2/R3, P3, model changes, training, and author contact remain unauthorized.

## 8. Mechanical command record

Terra used read-only commands equivalent to:

```powershell
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --porcelain=v1
rg --files | rg -i '(^|/)(hmdd_data/mda.?v3|.*v3\.2|.*mda.?v3|.*wang)'
Get-FileHash -Algorithm SHA256 -LiteralPath <artifact>
```

Counts were recomputed with Python `csv.DictReader` over the five tab-delimited official files,
using ordinal tuple sets at `(type,miRNA,disease)` and `(miRNA,disease)` grains. A preliminary
case-insensitive PowerShell uniqueness pass was discarded because it collapsed one
case-sensitive raw distinction. No count in this ledger relies on that discarded pass.
