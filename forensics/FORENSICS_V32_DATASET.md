# FORENSICS_V32_DATASET — HMDD v3.2 artifact reconstruction

Date: 2026-09-23. Session: Devin (continuation of forensic track R0/R1/R2).
Target fingerprint (paper Table 1): **411 miRNA × 271 disease × 11,748 typed triplets**, per-type
`[genetics=1155, epigenetics=403, circulation=2293, target=3997, tissue=3900]`.

## 1. Source inventory (VERIFIED)

| Source | Shape | Notes |
|---|---|---|
| Raw CuiLab `HMDD_data/MDAv3.2/v3_*.txt` (5 files, current snapshot) | 1,049 mir × 758 dis × **18,084** unique (mir,dis,type) triplets | per-type `[1869, 579, 3304, 6207, 6125]`; 14,217 unique pairs |
| `hmdd_v3.2_raw.xlsx` Sheet3 | binary matrix **853×591, 12,446 ones** | = SPLHRNMTF "MDAv3.2_data1" fingerprint — binary lineage, NOT typed |
| TDRC processed (`v3.2_wang_multilabel/…without_negative.csv`) | 713×447, **16,341** triplets / 12,534 pairs | TDRC type order [target,circu,epic,genetic,tissue] |
| v3.2 similarity dirs (`v3.2_wang`, `v3.2_processed`, `v3.2_wang_multilabel`) | — | `M_FSM==M_GSM` and `D_SSM1==D_SSM2` numerically → only **2 unique views**, not paper's 4 |

## 2. Paper's stated preprocessing is falsified (VERIFIED)

Paper claims "identical standardized preprocessing … miRNAs and diseases with fewer than
two experimentally verified associations are excluded".

- degree≥2 on raw (pairs or typed): **722×612×17,615** — nowhere near 411×271×11,748.
- No degree-only filter (typed/pair, single-pass/iterative, threshold 2–8) reproduces the fingerprint.
- No universe intersection (wang names, v2.0 names, mesh-nonempty, doid/icd10cm/omim/hpo combos) reproduces it.

## 3. Real lineage: SPLDHyperAWNTF (PLAUSIBLE→STRONG)

The v3.2 artifact is the lab's `MDAv3.2-3` tensor first used by **SPLDHyperAWNTF**
(DOI 10.1093/bib/bbac390, same CDMBlab/Ouyang-Dong group). Its repo
`Ouyang-Dong/SPLDHyperAWNTF_Model` is **deleted (404)**, no GitHub mirrors found; dataset not on
Zenodo/Figshare; paper Data Availability cites only `cuilab.cn/hmdd`.

Documented lineage filter (from SPLHRNMTF, DOI 10.1186/s12864-024-10729-w, same group, VERIFIED text):
- miRNA kept iff: has sequence in **miRBase** AND exists in **MISIM 2.0** (1,044 miRNAs).
- disease kept iff: found in **MeSH** AND tree category == **'C'** (diseases).

## 4. Reconstruction attempt (STRONG — structure solved, exact numbers blocked by source drift)

Implemented lineage filter on current raw files:
- eligibility: `mir ∈ miRBase(current hairpin.fa) ∩ MISIM2.0` × `dis ∈ MeSH-category-C` → **761×563×13,257**.
- + iterative typed-degree pruning `mir≥4, dis≥4` → **411**×340×12,347 — **miRNA count matched EXACTLY (411)**.
- best 3-way match `mir≥4, dis≥7` (typed, iterative) → 406×266×11,970 `[1239,439,2249,3956,4087]` — within ~2% on every dimension but not exact.

Conclusion (STRONG): the true pipeline is **eligibility filter (miRBase∩MISIM × MeSH-C) followed by
iterative degree pruning ~mir≥4 / dis≥7 on typed degree**. The residual ~2% gap is consistent with
the SPLD artifact being built on an **older HMDD v3.2 snapshot** (~2021-2022) — the CuiLab txt files
are living documents, and neither the 2022 snapshot nor MDAv3.2-3 itself is publicly archived.

Exact-match stopping rule NOT met. Status: **OPEN — artifact unrecoverable from public sources;
requires author data or the 2022 HMDD v3.2 snapshot.**

## 5. R2 approximation artifact — `v3.2_lineage_approx/` (BUILT 2026-09-23)

Built by `forensics/build_v32_r2_dataset.py` (hash-logged in `manifest.json`):
- **406 mir × 266 dis × 11,970 triplets** (per-type [gen 1239, epi 439, circ 2249, target 3956, tissue 4087]).
- `Y_multilabel.npy` — multi-hot [406,266,5] tensor (type order: genetics, epigenetics, circulation, target, tissue).
- `M_MISIM.npy` — real MISIM 2.0 functional similarity, all 406 miRNAs covered (mean 0.302).
- `D_MESH.npy` — Wang-style MeSH-tree semantic similarity computed from tree numbers (decay 0.5;
  7 "[unspecific]" pseudo-diseases have no MeSH path → zero row). Sparse: 70% off-diagonal zeros, mean 0.041.
- NOTE: earlier `mesh_trees.json` had 152 corrupted entries (char-level junk) — refetched & fixed;
  eligibility result unchanged (dataset counts identical before/after fix).

## 6. Implications

- v3.2 paper numbers (Top-1 F1 0.86, AUC 0.9181) are on an unreleased curated tensor (density 10.5% vs raw 2.3%).
- Local control experiment (Plan K): matching the paper's density (10.3%) did NOT improve Top-1 F1 (0.3336 vs 0.3301) → density alone is not the gap driver; entity selection + 4 similarity sources matter.
- Any R1/R2 rebuild should use the reconstructed pipeline above as the closest documented approximation and report it as an approximation, not the paper dataset.

## Evidence paths
- `/tmp/misim/miRNA_name.txt` (MISIM 2.0 names, 1,044)
- `/tmp/hairpin.fa` (miRBase hairpins → `/tmp/mirbase_names.json`)
- `/tmp/mesh_trees.json` (MeSH UI → tree letters; 536 C, 31 SCR, 23 non-C)
- `/tmp/mat853.npy` (xlsx Sheet3 binary matrix)
