# P8.1 Feasibility Decision Matrix

| option | evidence exists | missing | scientifically defensible? |
|---|---|---|---|
| **A. Exact four-view author-benchmark reproduction** | strict tensor (411×271×5, 11,748 labels, sha verified); bundled 4 views; shipped SPLD inputs bitwise-identified | author fold partitions, author seeds/config, MISIM+D_MESH recipes, original eval code | **Partially** — benchmark-behavior claims only; cannot claim author-exact reproduction (folds/config unknown) nor clean generalization (sim_m2 contaminated). Defensible as Track P5 exploratory evidence — which is exactly what G5–G7 produced. |
| **B. Four-view leakage-controlled reconstruction** | certified label-free rebuilds for sim_m1 (strict GFF3) and sim_d1 (FunSim); semantic rebuild candidate for sim_d2 | train-masked functional view to replace sim_m2 (feasible per U1); certified match for D_MESH or declared substitute; frozen config | **YES — as declared variant.** Feasible: swap sim_m2→train-masked functional rebuild and sim_d2→certified semantic rebuild, re-split, re-run under T3 protocol. NOT author-faithful (it's a 4-view *reconstruction* variant). |
| **C. Explicit three-view label-free research variant** | sim_m1 + sim_d1 certified; sim_d2 replaceable by verified mtrees2019 rebuild (28.9% diff → declared substitute) or dropped | decision which disease view(s) retained; metric comparability to 4-view work broken | **YES — cleanest defensible path.** Fewer views = different model; must be labeled a new research variant, not reproduction of DHGCMDA's feature set. |
| **D. Independent external/temporal validation** | none — no external MDA dataset or temporal cut audited yet | external source identification, entity mapping, provenance audit of that source | **Not yet actionable** — requires sourcing + auditing an independent dataset; design exists in REV1 but no evidence to execute. |

## Recommendation (design-only)
B and C are the only scientifically defensible forward paths for
generalization claims. A remains valid only for benchmark-behavior
reporting (what Track P5 already established). D needs new source work.
No execution authorized by this matrix.
