# PAPER_REPRO_GAP_REGISTER — DHGCMDA main reproduction

WP2 of COUNCIL decision 2026-11-18. Ordered blockers by impact on
comparability with the paper, then the paper↔code↔U1 mapping with
sources, then what can honestly be run on the author release.
Everything cited is verifiable in-repo or at the named upstream URL.

## A. Blockers ranked by impact on paper comparability

| # | Blocker | Impact | Status |
|---|---|---|---|
| G1 | **Paper v3.2 tensor unreleased** — curated 411×271 subset (5 types, 10.5% density) never published; only the shipped v2.0 npz + raw HMDD v3.2 exist | Without the author's pair universe, fold assignment and curated entity list, ANY v3.2 number is measured on a different dataset → cannot be claimed "same experiment" | BLOCKED (upstream main@06f9ba3 unchanged; no data release) |
| G2 | **Author v3.2 fold split unreleased** | Even with the tensor, test fold membership is unknown; our rng(0)-permutation split is an assumption | BLOCKED |
| G3 | **Author v3.2 eval code absent** — released `Calculate_Metrics.compute_top1_metrics` hardcodes 4 types (DROPS all type-5 rows, all length-6 predictions) → emits F1=0.0 on v3.2 regardless of model | The released metric cannot score v3.2 → authors used different (unreleased) eval code. Verified 3 ways in CLAUDE.md §12 (synthetic perfect predictor scores 0.0) | VERIFIED BUG, workaround exists (corrected general-K metric gives identical numbers on v2.0) |
| G4 | **Task definition mismatch** — paper reports "CV_type Top-1 P/R/F1 = 0.7915/0.9421/0.8600" on "four association types" (p.24) while v3.2 has five; our U1 reports multi-label macro-F1 on type logits | Metric formula (harmonic mean) matches, but label space differs (4 vs 5) and multi-label collapse (23.3% of pairs) is handled differently | STRUCTURAL — cannot be closed by code; only by author spec |
| G5 | **Feature stack mismatch** — paper uses 4 similarity sources (Wang MeSH + functional + disease-gene + miRNA-sequence) per §3.4.13; our best U1 build uses rebuilt train-only variants of the same families | Same families but different matrices; not byte-identical → residual confound | MEASURED (density control showed density is NOT the gap) |
| G6 | **Pipeline leakage asymmetry** — author pipeline's zero-bank includes held-out positives (PU); our rebuild normalizes this; paper's own preprocessing choices unknown | Minor relative to G1–G4 | DOCUMENTED |

**Bottom line**: gap 0.86-paper vs ~0.27-0.36 ours is dominated by
G1+G2+G4 (data curation + folds + label spec), NOT by model quality —
the density-controlled experiment (v3.2_wang_dense 0.3336 ≈ wang
0.3301) already showed data volume is not the lever. Solo closing is
infeasible (<5% estimated); needs author artifacts.

## B. Paper ↔ author-code v2.0 ↔ U1 v3.2 mapping (sources)

| Element | Paper | Author code v2.0 | U1 v3.2 (ours) |
|---|---|---|---|
| Predictor | "bilinear predictor" | `SimplifiedTypePredictor` diagonal bilinear + learnable T (upstream_hetero_model.py L417–515) — degenerate vs full bilinear (measured: full beats diag +5.9% on v2.0, J-1) | `DiagBilinearType` (port of diag form) + declared exist_rel adaptation |
| Loss | Eq.32 combined | `0.3·focal_exist + 0.7·type_CE(β=0.99999,ls=0.1)` (upstream_main L56–161) — 5 tricks not in Eq.32 | Identical objective + `0.3·exist` term only in U1X (amendment) |
| Negatives | not specified | `zero_index` ≤10× once, PU (includes held-out) | `neg_exist_idx` pinned rng(seed+77), same PU semantics |
| Eval | Top-1 F1 per type, 5-fold | `test_optimized`: channel-0 binary on equal-count sampled zero cells, 5 np seeds | 5 seeds {0..4} pinned, strata-aware |
| Data | 411×271×5 (v3.2), 495×383×4 (v2.0 shipped) | shipped v2.0 tensor | strict v3.2 rebuild (Y from HMDD v3.2, hash-pinned) |

Adaptations A1–A10 already classified in `DESIGN_REVIEW.md` §4
(port-faithful / adaptation / new-construction / unknown).

## C. What CAN honestly run on the author release today

| Experiment | Feasible now? | What it measures |
|---|---|---|
| v2.0 full pipeline as shipped | YES (already done: Top-1 F1 0.5521–0.6350 vs paper 0.5970) | Author-code faithfulness on shipped data |
| v2.0 ablation set | YES (done: reversal vs paper Fig.4 — terminal finding) | Robustness of paper's own claims |
| v3.2 on our tensor + corrected metric + pinned folds | YES (done: 0.27–0.36 range across configs) | OUR interpretation — explicitly labeled, not paper-equivalence |
| v3.2 on author tensor + author folds + author metric | **NO** — requires G1/G2/G3 artifacts | The only comparison that would count |
| CVtriplet author-style | PARTIAL — v2.0 eval replicable; v3.2 blocked by same G-blockers | Our U1X run = controlled supplement (accepted as negative) |

## D. Non-negotiable missing artifacts (to close G1–G4)

1. `data_v3.2` tensor (or the curation script + entity lists).
2. The 5-fold index files for v3.2.
3. The eval code that produced the v3.2 Top-1 numbers.
4. Clarification of the "four types" vs five-type label space.

Until those exist, every v3.2 claim from us stays labeled
"our interpretation, not paper-equivalent evidence".
