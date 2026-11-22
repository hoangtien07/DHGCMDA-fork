# P8.1 MISIM Dependency Audit — bundled sim_m2 / shipped M_MISIM

## What the bundled matrix is
`sim_m2` (411×411, sha 31e31ee4d449839b) is a bitwise-identical passthrough
of `M_MISIM` shipped in the SPLDHyperAWNTF MDAv3.2-3 bundle (Software
Heritage snapshot 2024-08-21 of Ouyang-Dong/SPLDHyperAWNTF_Model). The npz
metadata itself states: *"shipped M_MISIM passthrough (source-family
association-derived; overlap unmeasured)"*.

## Construction dependency
MISIM (Wang et al., functional miRNA similarity) computes miRNA–miRNA
similarity **from their associated disease sets**: each miRNA is represented
by its known associated diseases, and similarity is derived from the
semantic similarity of those disease sets. Therefore:
- The matrix is association-derived **by definition**, not incidentally.
- A held-out pair (m,d) remains encoded: if (m,d) contributed to the
  upstream association table, removing it from the evaluation split does
  not remove its influence on sim_m2 — the similarity was computed with it.
- This is source-level label leakage, orthogonal to pair-level masking.

## Auditability classification
| question | status |
|---|---|
| Construction method auditable? | PARTIAL — MISIM recipe published; the shipped file's exact upstream MDA snapshot/version is NOT documented in the bundle |
| Which MDA table fed it (HMDD version, date)? | UNVERIFIABLE — `misim_similarity.zip` not present on this VM (only sha 6b5da04c recorded); no manifest of its source snapshot |
| Overlap between its upstream labels and our 11,748 labels? | UNMEASURED — cannot compute without identifying the source table |
| Reconstructable from outer-train info? | YES in principle — U1 demonstrated `functional_similarity(train_mask)` rebuilding a train-masked equivalent; the bundled matrix is NOT train-masked |
| Independence from held-out labels establishable? | **NO** with current evidence |

## Verdict
`sim_m2` → **`BLOCKED_FOR_CLEAN_EVALUATION`**.
Not implicitly clean; cannot be used in any evaluation that claims
leakage-controlled generalization. A train-masked functional rebuild
(proven feasible in U1) is the only admissible substitute, and only under
a declared variant — never a silent swap of the benchmark input.

## Secondary note — sim_d2 (D_MESH passthrough)
Shipped disease semantic matrix: label-free by construction class (MeSH
tree only) but does not match our certified rebuild (28.9% cells differ).
Status: `UNCERTIFIED_LABELFREE_CLASS` — usable only after the shipped
recipe/version is identified, or as an explicitly declared reconstruction
variant.
