# REPRO_LEDGER — consolidated forensic findings (2026-09-23)

Documents: `FORENSICS_V32_DATASET.md` (dataset provenance), `FORENSICS_MODEL_PATH.md`
(released-code execution trace + leakage).

| # | Finding | Label |
|---|---|---|
| F1 | Paper preprocessing claim ("<2 associations excluded") cannot produce 411×271×11,748 from raw files (yields 722×612×17,615) | VERIFIED |
| F2 | v3.2 artifact = lab's `MDAv3.2-3` (SPLDHyperAWNTF lineage); repo deleted, not archived | STRONG |
| F3 | Documented lineage filter = miRBase∩MISIM2.0 (mir) × MeSH-category-C (dis), per SPLHRNMTF | VERIFIED (text) |
| F4 | Filter + iterative typed-degree pruning (mir≥4) reproduces **exactly 411 miRNAs**; best full config (mir≥4, dis≥7) → 406×266×11,970 (~2% off) | STRONG |
| F5 | Residual gap consistent with older raw snapshot (~2021-22), not with any tested filter — artifact unrecoverable publicly | PLAUSIBLE |
| F6 | All local "v3.2" similarity dirs contain only 2 unique views (M_FSM==M_GSM, D_SSM1==D_SSM2) vs paper's 4 | VERIFIED |
| F7 | Released code collapses multi-label pairs to scalar (23% signal lost on v3.2); type_indices tracks only types 1–4 | VERIFIED |
| F8 | Three-channel transductive leakage: GIP on full matrix; full matrix in view features+KNN; test edges in HGT | VERIFIED |
| F9 | Evaluator hardcodes 4 types → every v3.2 Top-1 = 0.0 regardless of model (synthetic-perfect-predictor proof) | VERIFIED |
| F10 | Reproduced binary AUC ≈ paper (0.92 vs 0.9181) is leakage-inflated, not generalization match | STRONG |
| F11 | Predictor is diagonal bilinear, paper says bilinear; full_bilinear fix exceeds paper on v2.0 (+6.4%) — v3.2 unchanged | VERIFIED |
| F12 | R2 leakage-free run on approximated v3.2: Top-1 hybrid F1 = 0.30, AUC 0.887 — same range as leaky runs → leakage is not the main gap driver vs paper 0.86 | VERIFIED |

## Track status

- **R0 (released-code reproduction)**: complete — code cannot score v3.2 at all; binary metrics are leaky.
- **R1 (paper-faithful)**: blocked — paper dataset (411×271×11,748) + author eval + 4 real similarity sources are unreleased. Closest documented approximation = F4 pipeline.
- **R2 (leakage-free)**: **COMPLETE 2026-09-23.** `forensics/run_v32_r2.py` on `v3.2_lineage_approx`
  (406×266×11,970, Y[m,d,5] preserved, per-fold masked GIP/features/edges, train-only class weights,
  multilabel BCE, full_bilinear, 300ep, seed 0, 5-fold CV_triplet):
  **Top-1 hybrid F1 = 0.3015** (micro-P 0.3850, macro-R 0.2478; sklearn macroF1 0.2307), **AUC 0.8865, AUPR 0.8967**.
  Per-fold: acc 0.378-0.399, AUC 0.885-0.896 — stable.
  Result (`forensics/r2_v32_results.json`): the honest number sits IN THE SAME RANGE as the leaky
  corrected-metric runs (0.27-0.36) → transductive leakage was NOT the main gap driver to paper 0.86;
  the dominant factors remain the unreleased curated dataset + author eval + missing similarity views.

## Recommended next steps (proposed, pending user)

1. Email CDMBlab asking specifically for `MDAv3.2-3` tensor + SPLDHyperAWNTF preprocessing script + author eval code (draft exists: `email_to_CDMBlab.md`).
2. Try Internet Archive Wayback for `Ouyang-Dong/SPLDHyperAWNTF_Model` (offline during this session — retry).
3. R2 rebuild on closest approximated v3.2 (F4 pipeline output) as the honest-baseline artifact.

## F13-F17 — ARTIFACT RECOVERED (2026-09-23, session 3)

| ID | Finding | Evidence |
|---|---|---|
| F13 | `Ouyang-Dong/SPLDHyperAWNTF_Model` recovered via **Software Heritage** snapshot (visit 2024-08-21, `swh:1:rev:ea5215ae9639b316a14c4797b49a34c43d018b88`) — full repo incl. `HMDD_data/MDAv3.2_3/` + model code | VERIFIED |
| F14 | `MDAv3.2_3` = **EXACT paper artifact**: 411×271×11,748, per-type [target 3997, circu 2293, epic 403, genetic 1155, tissue 3900], 8,735 unique pairs, 2,110 multi-type pairs, + real `mi_fun_sim` (411²) + `DSSM` (271²) + entity names | VERIFIED |
| F15 | TDRC `HMDD3.2_processed.rar` recovered from `BioMedicalBigDataMiningLab/TDRC` git history (blob 96ee66d7): 713×447×16,341 typed, names + Dis_sim | VERIFIED |
| F16 | 411×271 is NOT derivable from the 713×447 universe by degree pruning (any typed/pair, single/iterative k≤8: triplets bottom out ~11,799 at 292×235), pair-ntypes filtering, or misim∩mirbase (568 mirs) — SPLD built from a different/earlier raw snapshot | STRONG |
| F17 | Lineage map: raw txt 1049×758×18,084 → lab binary 853×591×12,446 → TDRC 713×447×16,341 → SPLD 411×271×11,748 | VERIFIED |

## R3 (paper-exact data) — COMPLETE

- Dataset dir `v3.2_spld_paper/` built from recovered artifact (Y[411,271,5] multi-label, real sims).
- Raw CSVs persisted to `HMDD_data/MDAv3.2-3/`; recovered SPLD code at `forensics/spld_recovered/`; TDRC data at `HMDD_data/TDRC_v3.2_processed/`; Y713+names at `forensics/`; pipeline-format dir `v3.2_spld_paper_pipeline/` (dataset branch registered in param.py).
- Canonical triplet list + SHA256: `forensics/MDAv3.2-3.canonical.tsv`, `MDAv3.2-3.sha256.json`.

## F18-F19 — EXACT-ARTIFACT EVALUATION RESULTS (2026-09-23)

| ID | Finding | Evidence |
|---|---|---|
| F18 | Honest multilabel eval (`run_v32_r2.py`, leakage-free, full_bilinear, 300ep×5fold) on **exact paper artifact**: **Top-1 hybrid F1 = 0.3097, AUC = 0.8952, AUPR = 0.9039** — statistically identical to approximated dataset (0.3015/0.8865) → **dataset is NOT the gap driver** | VERIFIED (`forensics/r3_spld_exact_results.json`, `logs/r3_spld_exact.log`) |
| F19 | Released (leaky, scalar-collapsed) pipeline `run_v32_correct_metric.py --dataset v3.2_spld_paper_pipeline` on exact artifact: **Top-1 F1 = 0.1889, binary AUC = 0.878, AUPR = 0.881** — scalar collapse loses ~26% triplets; binary metrics ≈ paper (0.9181/0.9271 within ~4 pts) but Top-1 collapses | VERIFIED (`logs/r3_leaky_spld_exact.log`) |

## R3 verdict

**Tier A data achieved** (exact artifact `HMDD_data/MDAv3.2-3/` + canonical SHA256 `c8f36c77...` at `forensics/MDAv3.2-3.sha256.json`), **yet paper's 0.86 remains unreproduced**:
- honest eval: 0.31 | released pipeline: 0.19 | paper claim: 0.86
- → Gap is attributable ONLY to the author's private training/eval code (released code is a rewrite) or to an eval-protocol difference not documented anywhere.
- SPLDHyperAWNTF original code (recovered) running on its own artifact as artifact-authenticity check (paper reports F1≈0.53 on MDAv3.2-3) — `max_iter=100`.

| F20 | SPLDHyperAWNTF original code (recovered) on its own artifact, `max_iter=100`, pair-fold Top-1 eval: **P=0.6219, micro-R=0.4624, macro-R=0.5069 → F1≈0.56** — matches SPLD paper ~0.53 on MDAv3.2-3 → **artifact authenticity CONFIRMED** | VERIFIED (`/tmp/spld_run.log`) |

## FINAL VERDICT (2026-09-23)

Same artifact, same Top-1 pair-fold protocol:
- SPLDHyperAWNTF (its own authors' method): F1 ≈ **0.53–0.56**
- DHGCMDA honest multilabel: **0.31** | DHGCMDA released (leaky+collapse): **0.19** | DHGCMDA paper claim: **0.86**

=> 0.86 is ~1.6x the reference method's score on identical data+protocol — almost certainly produced by an eval protocol or training code never released (not by the dataset, which is now proven exact).

| F21 | **Top-1 recall ceiling proof**: paper's stated CVtype = pair-fold + 1 prediction/pair → recall ≤ 8735/11748 = **0.7435 (micro)** or ≤ mean(1/pos_i) = **0.8642 (macro)**. Paper claims R=0.9421 → **mathematically impossible** under the documented protocol on MDAv3.2-3 | VERIFIED (arithmetic on recovered artifact) |
| F22 | Table-3 baseline consistency: SPLDHyperAWNTF row in DHGCMDA Table 3 = **P=0.6219, R=0.4624** — exact match to our independent run of SPLD's own code/artifact (P=0.6219, micro-R=0.4624, macro-R=0.5069) → baselines were computed with SPLD's eval (micro-R), while DHGCMDA's own R=0.9421 is unreachable by any single-pred-per-pair metric | VERIFIED |
| F23 | CDMBlab/DHGCMDA absent from Software Heritage; SPLHRNMTF (binary eval) + MRGBMDAT (ternary/binary edge task) recovered & eliminated as eval-code sources; sibling-leakage baseline on triplet-fold gives F1=0.216 | VERIFIED |

## Private-code hunt verdict

No public trace of the author's actual eval/training code (SWH, GitHub forks, sibling repos all checked). Moreover the claimed Top-1 R=0.9421 exceeds the recall ceiling of the documented protocol (0.8642), so the reported number **cannot** have come from the protocol the paper describes — the true eval was either per-triplet, multi-prediction, or a different metric/split; only the authors can confirm.

| F24 | **Similarity provenance (M0)**: `mi_fun_sim_3.2_3.csv` = **MISIM 2.0 exactly** (|diff|~1e-10 over 10k pairs) → label-derived (HMDD associations) → leaky if used static → excluded from primary track. `DSSM3.2_3.csv` corr 0.90 vs TDRC Dis_sim (MeSH/DO ontology, max 1.19) → ontology-derived, conditionally STATIC_EXTERNAL. R3 run previously used MISIM as static features → flagged as containing partial leakage | VERIFIED |
| F25 | M1 golden evaluator: `eval_top1.py` reproduces SPLD fold membership EXACTLY (seed0 shuffle+slice, 5×1747 pairs) and metric semantics identical to SPLD internal loop on synthetic scores | VERIFIED |
| F26 | M2 early: **simknn_gip (vote over GIP_mi(train)×(DSSM|GIP_d)) = legacy-F1 0.5599 ≈ SPLD full tensor model 0.56** — published model only matches a trivial similarity-vote baseline; cooccur_prior floor = 0.394 | VERIFIED (pending other models) |

| F27 | **Golden SPLD replay**: P=0.6219/miR=0.4624/maR=0.5069 → legacy-F1 **0.5585**, per-fold 0.5505–0.5647; seeded-deterministic (identical to prior unseeded run + Table-3 baseline row). Artifacts: golden_spld_eval.json + pred_fold*.npy | VERIFIED |
| F28 | **M2 tournament verdict**: `simknn_gip` (0-parameter GIP×DSSM vote) = **0.5599** — statistically TIED with SPLD (paired bootstrap Δ=+0.0014, CI95 [−0.0075,+0.0104]); `simknn_misim` (leaky) = 0.5799 (+0.021, sig → leak channel real but modest); DHGCMDA encoder honest = **0.4481** (−0.11 below SPLD); J-prior +0.002; bilinear 0.29 / distmult 0.24 | VERIFIED |
| F29 | M3: DHGCMDA (fixed multilabel head, DSSM+train-GIP, honest eval) folds 0.427–0.479 → mean **F1=0.4481, macro-AUPR=0.3531** — paper's architecture underperforms a zero-parameter similarity vote by ~11pts on its own benchmark | VERIFIED |

| F30 | **Final-suite vote ablations** (golden pair-fold, full metrics): `gip_only` (miRNA-side train-GIP vote only) = **0.5910** — BEST OBSERVED performance, above every dual-view variant and above the leaky control; `misim_dssm` (LEAKY) = 0.5799; `fmisim_dssm` (fold-reconstructed MISIM) = 0.5733; `gip_gip` = 0.5655; `simknn_j` (GIP+DSSM+J) = 0.5610; `simknn_gip` (GIP+DSSM) = 0.5599; `dssm_only` = 0.4173. DSSM disease-side vote *hurts* (−3.1pt vs gip_only); J prior +0.001 negligible → performance comes from miRNA-side similarity + local propagation, not multi-view machinery | VERIFIED (`final_suite_results.json`) |
| F31 | **Learned-model seed robustness** (golden folds fixed, init seeds 0/1/2): bilinear F1 = 0.2904/0.2822/0.2753 (mean 0.283±0.006); distmult = 0.2427/0.2598/0.2550 (mean 0.252±0.007) — stable, far below vote. Repeated pair-CV seeds 0/1/2 for `simknn_gip` (secondary robustness): 0.5599/0.5616/0.5627 (mean 0.561±0.001) — vote result robust to fold draw | VERIFIED (`m2_baseline_results_seed{1,2}.json`, `final_suite_results.json`) |
| F32 | **SPLD MISIM channel measured** (final-validation): golden replay ×3 similarity configs → static MISIM F1=**0.5585**, fold-reconstructed MISIM F1=**0.5570** (−0.0015), no-MISIM (MGSM/GIP only) F1=**0.5594** (+0.0009). SPLD's score is MISIM-independent (its tensor model does not exploit the leaky static matrix) → legacy comparison is effectively clean. Contrast vote: static MISIM adds +0.020 | VERIFIED (`golden_spld_eval_{fold,none}.json`) |
