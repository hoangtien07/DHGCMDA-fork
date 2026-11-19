# FOUR_VIEW_SOURCE_PROVENANCE — WP2

Two input variants preserved (never conflated):
- **RECOVERED_BENCHMARK_VIEWS**: sim_m1/sim_m2/sim_d1/sim_d2 inside the strict tensor (as shipped in the recovered MDAv3.2-3 bundle).
- **PUBLIC_RECONSTRUCTED_4VIEW**: independently built views from public sources (ext/seqgene). Not asserted to be the author's features.

## Per-view trace (paper ↔ released code ↔ our tensors)
| view | paper description | released code v2.0 | recovered bundle | public reconstruction | provenance class |
|---|---|---|---|---|---|
| miRNA sequence | miRBase seq similarity | bundled matrix (v2.0) | sim_m1 | `ext/seqgene/mature.fa` + hairpin.fa, NW alignment, sha in ext/seqgene/SHA256SUMS | INDEPENDENT_EXTERNAL (reconstructed); bundle view UNKNOWN_PROVENANCE |
| miRNA functional | MISIM functional sim | bundled M_FSM (association-derived → LEAKED in release) | sim_m2 | `ext/misim_similarity.zip` (d2025.bin MISIM) | bundle: UNKNOWN_PROVENANCE; reconstructed MISIM: INDEPENDENT_EXTERNAL but note MISIM is association-derived upstream |
| disease gene | disease-gene sim | bundled M_GSM (GIP full-assoc → LEAKED) | sim_d1 | CTD curated + HumanNet-XC (`ext/seqgene/CTD_curated_genes_diseases.tsv.gz`, `HumanNet-XC.tsv`) | reconstructed: INDEPENDENT_EXTERNAL; bundle: UNKNOWN_PROVENANCE |
| disease semantic | MeSH Wang semantic | D_SSM1/D_SSM2 | sim_d2 | mtrees2019.bin MeSH-2019 Wang | INDEPENDENT_EXTERNAL (reconstructed); bundle UNKNOWN_PROVENANCE |

## Classification rules applied
- INDEPENDENT_EXTERNAL: built from public source data without touching association labels.
- TRAIN_ASSOCIATION_DERIVED: computed from (train-masked) association matrix — admissible only with train-only masking (U1 bundle does this; released v2.0 code does NOT → leaked).
- UNKNOWN_PROVENANCE: bundle-shipped matrices we cannot byte-trace to a public recipe → unsafe-for-clean-claim. "Not equal to full-A GIP" does NOT prove absence of leakage.

## Finding
`RECOVERED_BENCHMARK_VIEWS` sim_m1/sim_m2/sim_d1/sim_d2 remain UNKNOWN_PROVENANCE: stats match the paper's dataset description but we cannot prove which recipe produced them. A CLEAN_REPRO_K5 run must therefore use PUBLIC_RECONSTRUCTED_4VIEW, not the bundle views.
