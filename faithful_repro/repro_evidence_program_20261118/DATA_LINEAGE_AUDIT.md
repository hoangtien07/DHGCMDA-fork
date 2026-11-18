# DATA_LINEAGE_AUDIT — v3.2 candidate chain and filter forensics

WP2 output. Sources: `data_audit.json`, `EXTERNAL_LINEAGE.md`,
strict npz re-measured in this session (numbers below are computed,
not copied).

## Chain

HMDD v3.2 raw (cuilab, `HMDDv3.xlsx`, 1049×758×~18k cells)
→ SPLDHyperAWNTF benchmark **MDAv3.2-3** (411×271×5, 11,748 links;
SPLD paper describes a <3-associations filter and excludes the 6th
evidence group `other`)
→ local `reference` snapshot (sha `135e2050…`)
→ local `strict` tensor
`data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
(sha `85bb9ea4…`; elementwise-identical labels + ordered names to the
SPLD snapshot and reference — `data_audit.json` PASS).

## Re-measured strict tensor

- Shape 411×271×5; **11,748 typed links; 8,735 positive pairs;
  2,110 multi-type pairs (24.2%)**; per-type counts
  [1155, 403, 2293, 3997, 3900] (order: genetics, epigenetics,
  circulation, target, tissue).
- Dimensions match the paper's Table 1 exactly — but matching
  DIMENSIONS is not matching entity lists/fold membership (G1/G2
  remain open; identical stats also held by the pre-existing SPLD
  benchmark, so the match alone does not prove DHGCMDA used that
  byte stream).

## Filter forensics (do NOT tune to match 11,748)

| Description | Source | Measured on strict tensor |
|---|---|---|
| "<2 associations" entities dropped | DHGCMDA text | 2 miRNA (hsa-mir-1179, hsa-mir-1244) have exactly 1 positive pair — would violate a literal pair-level <2 post-filter |
| "<3 associations" for MDAv3.2-3 | SPLD text | 10 miRNA + 1 disease have <3 pairs — would violate a literal <3 post-filter |

Interpretation: both texts describe filters applied at an earlier
processing stage or with a different counting unit (typed links vs
pairs; pre- vs post-curation). The local tensor matches NEITHER
literally, and we cannot decide which unit/stage the authors meant
without their script. This is `UNRESOLVED` — recorded, not forced.

## Similarity / feature provenance (four families)

| Feature | In bundle | Source | Association-derived? |
|---|---|---|---|
| S_seq (miRNA) | view_mi_v1 | NW alignment on GFF3 strict | NO — external sequence data |
| S_func (miRNA) | view_mi_v2 | rebuilt train-only per fold | train-only (leakage-controlled) |
| S_gene (disease) | view_dis_v1 | FunSim (CTD + HumanNet-XC) | NO — external |
| S_sem (disease) | view_dis_v2 | MeSH-2019 rebuild by disease name | NO — external |

Shipped M_MISIM/D_MESH (association-derived) are NOT in the bundle —
verified earlier this session on real arrays. `bundle_audit.json`:
`functional_semantic_rebuilt_equal=true`, `heldout_in_mask=0`.

Contrast v2.0 release: `prepareData.py` builds the full association
matrix before the split; oracle read of test labels → Top-1 F1 1.0
(feature_oracle.json). That is a REAL leakage path in the author
release; in our v3.2 stack the analogous channel is closed.

## 4-vs-5 types implication

Paper p.24 evaluates "four association types" while v3.2 carries five
(+ tissue). The released metric drops type-5 (WP1). Whether the paper
collapsed tissue, dropped it, or scored a different label space is
`UNRESOLVED` — affects every Top-1 number's denominator.

## External artifacts

Upstream main@06f9ba3 unchanged; no author data/folds/eval release
→ `EXTERNAL_ARTIFACT_REQUIRED`. Draft request:
`ARTIFACT_REQUEST_TO_AUTHORS.md` (not sent — needs owner authorization).
