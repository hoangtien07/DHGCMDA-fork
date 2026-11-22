# P8 Feature Provenance Readiness — four similarity views + label lineage
Sources audited per `FEATURE_PROVENANCE_DAG.md` (Track P5 G2) and
`FOUR_VIEW_PROVENANCE_CORRECTED.md` (Track R). No view replaced; no
benchmark input modified.

## Per-view verdict
| view | content | label-dependence | train-only reconstructable |
|---|---|---|---|
| sim_m1 | miRNA sequence similarity | **LABEL_FREE** — miRBase hairpin+mature, NW alignment (shas 0b315c60, 52d69eeb); sequence only | YES — public reconstruction exists |
| sim_m2 | miRNA functional similarity | **EXTERNAL_POTENTIALLY_LABEL_DEPENDENT** — MISIM (6b5da04c) built on known MDA upstream; cannot certify label-free | PARTIAL — raw MISIM inputs public, but upstream MDA dependence unverifiable |
| sim_d1 | disease-gene functional | **LABEL_FREE for MDA labels** — CTD + HumanNet-XC FunSim (0422d3bd, d45931ef); audited, no MDA input | YES |
| sim_d2 | disease semantic | **LABEL_FREE** — MeSH tree semantic (mtrees2019.bin + meshToID); MeSH-2025 d2025.bin is unrelated | YES |
| bundled sim_* inside tensor | whichever the release bundled | **UNKNOWN_PROVENANCE** — the npz's stored views are not certified to match the reconstructable ones | NO for the stored arrays themselves |

## Derived-path leakage audit
| path | status |
|---|---|
| GIP kernels on FULL association matrix | **LEAKED vs held-out** if computed on complete A — in the P5 runner `gip(Af)` uses train pairs only, but only when a bundled view is absent; bundled views bypass recomputation → inherited UNKNOWN |
| association edges / HGT ei / contrastive positives | TRAIN_DEPENDENT — masking verified (L329–356; canary test exists) |
| existence negative pool | train-correct — test/val coords excluded |
| class weights w_k | train counts only — correct |

## Can a fold's training features encode its held-out labels?
- Direct paths (A masking, pools, edges): no — verified by masking audit
  and the held-out-label-invariance canary.
- Indirect paths: **yes, potentially** — sim_m2 (MISIM upstream MDA
  dependence) and any bundled view whose construction used the full
  association matrix could encode held-out pair information. This cannot
  currently be verified or excluded.

## Lineage graph
```
HMDD v3.2 raw ──> strict tensor Y[411,271,5] (sha 85bb9ea4…)
  ├─ sim_m1  [LABEL_FREE; recon via miRBase 0b315c60/52d69eeb]
  ├─ sim_m2  [LABEL-DEP?; MISIM 6b5da04c — MDA-dependent upstream]
  ├─ sim_d1  [LABEL_FREE; CTD 0422d3bd + HumanNet d45931ef]
  ├─ sim_d2  [LABEL_FREE; MeSH mtrees2019]
  └─ bundled sim_* [UNKNOWN — not certified]
      └─ GIP/features/edges on train-A only [TRAIN_DEPENDENT — verified]
```

## Unverifiable-now dependencies
1. Exact construction of the bundled sim_* arrays (bitwise provenance).
2. MISIM's upstream MDA usage — label-contamination possible, unmeasurable.
3. Whether the release's recon targets equal any public reconstruction.

## Readiness implication
A prospective leakage-controlled evaluation must either (a) certify the
bundled views against train-only reconstructions, or (b) use only the
three views with LABEL_FREE or documented-dependence status. No view was
replaced or modified in this audit.
