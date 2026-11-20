# FEATURE_PROVENANCE_DAG — label→input lineage (also satisfies FEATURE_DAG.md)

Classification: `LABEL_FREE` · `TRAIN_DEPENDENT` ·
`EXTERNAL_POTENTIALLY_LABEL_DEPENDENT` · `UNKNOWN`.
Three-layer model inherited from `FOUR_VIEW_PROVENANCE_CORRECTED.md`
(Track R, closed): (a) raw four similarity inputs; (b) GIP-augmented
IM/ID matrices computed on the full association matrix; (c) direct
full-association features/graphs.

## DAG

```
HMDD v3.2 raw (cuilab) ──filter──> strict tensor Y[411,271,5]   [LABEL_FREE-as-source]
        │                                        │
        │                                   A = Y.sum(2)>0 (FULL, incl. test)
        │                                        │
        │                        ┌───────────────┼──────────────────────┐
        │                        ▼               ▼                      ▼
        │              GIP kernels (full-A)   association edges    neg_pool labels
        │              DGSM/MGSM (release)    HGT m-d/d-m          (existence/CL)
        │                        │               │                      │
        │                        ▼               ▼                      ▼
        │              IM/ID = (raw+GIP)/2   HGT graph ei      contrastive targets
        │              [LEAKED vs held-out]  [TRAIN_DEPENDENT]  [TRAIN_DEPENDENT]
        │
        ├─> bundle sim_m1 (=M_GSM raw?)   miRNA sequence view     [UNKNOWN_PROVENANCE]
        ├─> bundle sim_m2 (=M_FSM raw?)   miRNA functional view   [UNKNOWN_PROVENANCE]
        ├─> bundle sim_d1 (=D_SSM2 raw?)  disease-gene view       [UNKNOWN_PROVENANCE]
        └─> bundle sim_d2 (=D_SSM1 raw?)  disease semantic view   [UNKNOWN_PROVENANCE]

Public reconstruction lineage (each audited):
  miRBase hairpin.fa(0b315c60)+mature.fa.gz(52d69eeb) ──NW align──> recon sim_m1   [LABEL_FREE — sequence only]
  misim_similarity.zip(6b5da04c) ──> recon sim_m2   [EXTERNAL_POTENTIALLY_LABEL_DEPENDENT — MISIM built on known MDA upstream]
  CTD_curated_genes_diseases.tsv.gz(0422d3bd)+HumanNet-XC.tsv(d45931ef) ──FunSim──> recon sim_d1  [LABEL_FREE for MDA labels — verified no MDA input]
  mtrees2019.bin+meshToID.csv ──semantic──> recon sim_d2   [LABEL_FREE — MeSH tree only; d2025.bin(82a6d0f1) is the newer NLM MeSH 2025 file, NOT MISIM]
```

## Per-path masking audit (locked CV_PAIR folds)

| Data path | What must exclude test labels | Status |
|---|---|---|
| A used in features/graphs/CL | train pairs only | faithful.py L329-332 builds `Ytr`/`A` from `fold['train']` — correct; `diag_leak=0` default; leak path exists only under explicit `--diag_leak=1` flag |
| GIP recomputation | computed on `Af`(train) | L335-336: `gip(Af)` only when bundled view absent (ndim==0); bundled views bypass recomputation — provenance UNKNOWN |
| KNN/HGCN G | built on X from train-masked A | L337-340 — inherits A masking |
| HGT edges ei | A edges from `Aft` | L345: `Aft=Af`=train under default; θ-threshold m-m/d-d edges use recon Ŝ (model output, train-driven) |
| Contrastive positives | `inter_loss` pos = `At` (train) | L402 passes `At`; negatives sampled from complement — includes held-out pairs as *potential negatives* only via similarity, acceptable since labels unseen; flagged for canary |
| Existence negative pool | excludes test/val pairs by coords | L347-356 `excl` covers test/val/test_neg(s) — verified by test_split_and_provenance |
| Reconstruction targets | `S_M=sim_m2, S_D=sim_d2` | L343 — bundled views: label-dependency UNKNOWN (not certified) |
| Class weights w_k | train counts `Ytr` | L357 — correct |
| Threshold fitting | inner-val only | P5 contract; CVtype currently evaluates top1 directly (no threshold); triplet threshold from `fold['val']` (L420, L433) — correct |
| Metric inputs | `Y[test pairs]` full labels | L369 — test labels only at evaluation |

## Canary design (synthetic / controlled copy only)

`tests/test_split_and_provenance.py::test_canary_heldout_label_invariance`
perturbs held-out labels in a synthetic dataset, **re-runs the full
feature-build path** (Ytr → A → GIP → X → G → edges → negative pool) and
asserts every train-visible tensor is bit-identical. Perturbing Y while
keeping precomputed sim_* fixed does NOT count as a clean-provenance
proof — noted as limitation, not evidence.

## Verdicts per profile

- `P5_BENCHMARK_UNKNOWN_PROVENANCE`: **G2_PASS_EXPLORATORY_ONLY** —
  masking paths audited clean at train-visible level, but bundled views
  are UNKNOWN_PROVENANCE; no leakage-free biological claim.
- `P5_SPLIT_AWARE_EXTERNAL`: **BLOCKED_FEATURE_PROVENANCE** for the
  functional view — MISIM is association-derived upstream and cannot be
  shown label-free w.r.t. held-out pairs without fold-conditional
  reconstruction; no silent replacement used. The other three views
  audit LABEL_FREE (miRBase seq, CTD+HumanNet gene, MeSH semantic).
  → Profile incomplete ⇒ exploratory downgrade, no clean claim.
- `P5_RELEASE_K4_CONTROL`: sanity-only on v2.0 release inputs.
