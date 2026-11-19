# INPUT_EQUIVALENCE_MATRIX — paper ↔ release ↔ recovered ↔ reconstructed

| input | paper | released code (06f9ba3) | recovered bundle (strict npz) | public reconstruction | equivalence |
|---|---|---|---|---|---|
| typed assoc tensor | 411×271, 11,748 links, 5 types [paper p.24 stats] | v2.0 495×383×4 only | 411×271×5, 11,748 links — L1 match | n/a (labels are the thing under test) | L1 statistics match; L2/L3 (byte/recipe) unverifiable |
| folds | 5-fold CV mentioned | seed-fixed KFold in code | n/a | recovered partitions (independent_v32_outer_refit) | NOT equivalent — author folds unknown |
| miRNA seq view | yes | bundled matrix | sim_m1 | NW alignment from miRBase fasta | recipe-level equivalence UNKNOWN |
| miRNA func view | yes | bundled (leaked) | sim_m2 | MISIM d2025.bin | recipe-level UNKNOWN |
| disease gene view | yes | bundled (leaked) | sim_d1 | CTD+HumanNet-XC | recipe-level UNKNOWN |
| disease semantic view | yes | bundled | sim_d2 | MeSH-2019 Wang | recipe-level likely close; byte UNVERIFIED |
| evaluator | "four types" text | K4-only code | n/a | locked metric suite (ours) | NOT equivalent — K4 drops K5 |
| loss/objective | Eq. in paper | implementation w/ extra tricks | — | U1 adaptation | see MODEL_LOSS_FIDELITY_TRACE |
