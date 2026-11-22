# P8.1 Bundled Feature Lineage — strict npz `v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz` (sha 85bb9ea4…)
Entity order: mir_names (411), dis_names (271) — identical order to
`v3.2_spld_paper/{miRNA,disease}_names.txt` (verified string-equal).

## Per-matrix audit (measured 2026-11-23)
| view | shape | sha256 (first 16) | off-diag stats | lineage | match class | label-dependency |
|---|---|---|---|---|---|---|
| sim_m1 | 411×411 | dad11e4df3bc54d2 | mean 0.2733, zero 18.5% | **own deterministic rebuild**: strict miRBase GFF3 `Derives_from` parent→mature NW alignment (+1/−1/−2/−1), max over arms, off-diag minmax, uncovered→0/1 (npz `extra.view_m1`; method audited in V32_CANONICAL_DATA_AUDIT) | DETERMINISTIC_REBUILT (sources absent on this VM — re-verification blocked) | LABEL_FREE by construction |
| sim_m2 | 411×411 | 31e31ee4d449839b | mean 0.3129, zero 0% | **EXACT bitwise passthrough** of SPLDHyperAWNTF shipped `M_MISIM` (max\|Δ\|=0 vs `v3.2_spld_paper/M_MISIM.npy`, same entity order) | EXACT match → association-derived upstream | **TRAIN/UPSTREAM ASSOCIATION-DERIVED — see MISIM audit** |
| sim_d1 | 271×271 | 382658cb2382c954 | mean 0.0803, zero 35.5% | **own deterministic rebuild**: Cheng-style FunSim from CTD `marker/mechanism` + HumanNet-XC LLS (npz `extra.view_d1`) | DETERMINISTIC_REBUILT (sources absent on this VM) | LABEL_FREE for MDA labels (CTD/HumanNet contain no miRNA–disease edges) |
| sim_d2 | 271×271 | fd8654ba3ae129bb | mean 0.0603, zero 72.1% | **EXACT bitwise passthrough** of SPLDHyperAWNTF shipped `D_MESH` (max\|Δ\|=0 vs `v3.2_spld_paper/D_MESH.npy`) | EXACT passthrough; **NON-MATCH vs our mtrees2019+meshToID+HMDDv3 semantic rebuild** — 28.9% cells differ, max\|Δ\|≈0.985, coverage 262/271 | label-free *by construction class* (MeSH tree semantics), but shipped recipe/version unverifiable |

## Rules applied
- Exact/approx/non-match assigned by deterministic alignment + numerical
  diff only; no view judged label-free from its name.
- `v3.2_lineage_approx` matrices (406×266) do not share entity order —
  excluded from comparison.
- sim_m1/sim_d1 rebuild scripts and their input SHAs are documented in
  `build_4view.py` + `V32_CANONICAL_DATA_AUDIT.md`; source files
  (hairpin/mature fa, CTD, HumanNet, hsa.gff3) are NOT present on this VM —
  bitwise re-verification deferred, method-level audit stands.

## Consequence
- Clean-by-construction-class: sim_m1, sim_d1 (own rebuilds, no MDA input).
- Exact-match contaminated-class: sim_m2 (MISIM passthrough).
- Passthrough unverifiable: sim_d2 (shipped D_MESH ≠ our rebuild; cannot
  certify which MeSH version/method produced it).
