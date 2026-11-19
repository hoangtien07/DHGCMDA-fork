# FOUR_VIEW_PROVENANCE_CORRECTED — REV F1.1 + documentation errata

Corrected against the actual released code (`pinned_release/prepareData.py` L190-398) and hashed source artifacts.

## Three-layer distinction (errata — do not conflate)
The release pipeline contains three distinct kinds of inputs:

1. **Raw four similarity inputs** — the shipped text files `M_GSM.txt`,
   `M_FSM.txt`, `D_SSM2.txt`, `D_SSM1.txt`. These are raw source matrices as
   distributed; the release code loads them directly. Whether each *shipped
   file itself* embeds association information is a separate question (see
   per-view trace) — we do **not** claim every raw source view is necessarily
   GIP-augmented.
2. **GIP-augmented IM/ID matrices** — computed inside the release pipeline:
   `preprocess_similarity_matrices` (L190-224) computes Gaussian (GIP) kernels
   DGSM/MGSM **from the full association matrix** and merges
   `ID = (dd_mat + DGSM)/2`, `IM` likewise (zero-masked). **These released
   model inputs are association-augmented ⇒ LEAKED relative to held-out
   labels.** The leakage conclusion stands at this layer.
3. **Direct full-association features/graphs** — the heterogeneous graph
   structure and any association-derived adjacency/features built from A
   inside the training code. These consume the full association matrix by
   construction ⇒ also LEAKED for cold-start evaluation.

## Release-code raw-view mapping (authoritative)
`prepareData.py`:
- **D_SSM1.txt** → disease semantic similarity (disease View 2 base) — L332-339.
- **M_FSM.txt** → miRNA functional similarity (miRNA View 2 base) — L342-349.
- **D_SSM2.txt** → disease-gene features (disease View 1 base) — L369-376.
- **M_GSM.txt** → miRNA sequence features (miRNA View 1 base) — L379-386.
- In the released pipeline each raw view is then merged with a full-A GIP
  kernel into IM/ID (layer 2 above).

## Corrected per-view trace
| view | release input | released augmentation | our bundle (strict npz) | public reconstruction (source + hash) | association-dependency | possible test-label exposure | class |
|---|---|---|---|---|---|---|---|
| miRNA seq (view 1) | M_GSM.txt | +GIP-MGSM (full A) | sim_m1 | miRBase `hairpin.fa` (sha 0b315c60), `mature.fa.gz` (sha 52d69eeb), NW alignment | release: YES (GIP on full A); bundle: UNKNOWN; recon: NO | release YES; bundle unknown; recon none | recon: INDEPENDENT_EXTERNAL; bundle: UNKNOWN_PROVENANCE |
| miRNA func (view 2) | M_FSM.txt | +GIP-MGSM (full A) | sim_m2 | `misim_similarity.zip` (sha 6b5da04c) — MISIM | release YES; **MISIM itself is association-derived upstream** (Wang MISIM built on known miRNA–disease associations) | recon via MISIM: **possible — not clean by default** | TRAIN_ASSOCIATION_DERIVED unless proven otherwise |
| disease gene (view 1) | D_SSM2.txt | +GIP-DGSM (full A) | sim_d1 | `CTD_curated_genes_diseases.tsv.gz` (sha 0422d3bd) + `HumanNet-XC.tsv` (sha d45931ef) — FunSim | release YES; recon: disease–gene links, NOT miRNA–disease labels | recon: none for MDA labels | recon: INDEPENDENT_EXTERNAL (caveat: external DB ≠ automatically clean — verified no MDA-label input) |
| disease semantic (view 2) | D_SSM1.txt | +GIP-DGSM (full A) | sim_d2 | **MeSH tree**: `mtrees2019.bin` + `meshToID.csv` (semantic rebuild) — distinct from **`d2025.bin`** = NLM MeSH 2025 descriptor file (sha 82a6d0f1), a newer/different source artifact | release YES; recon: MeSH tree structure only | none | recon: INDEPENDENT_EXTERNAL |

## Explicitly distinguished (council requirement)
- **`d2025.bin`** = NLM MeSH 2025 descriptors binary — candidate newer source for the disease-semantic view; NOT MISIM.
- **`misim_similarity.zip`** = MISIM (Wang) miRNA functional similarity — association-derived upstream; cannot be labeled CLEAN solely because it is downloadable/public.
- **`mtrees2019.bin`** = MeSH tree 2019 used in the actual rebuild (separate from d2025.bin).

## Rule applied (council): "external public DB" ≠ "clean"
A view is INDEPENDENT_EXTERNAL only if its construction does not consume
miRNA–disease association labels (incl. upstream). MISIM fails this test;
CTD/HumanNet and MeSH tree pass (verified: no MDA-label input); bundle views
remain UNKNOWN_PROVENANCE — "not full-A GIP" does not prove cleanliness.
