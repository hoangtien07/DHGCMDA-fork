# DATASET_IDENTITY_EVIDENCE — strict tensor vs paper v3.2 (REV4 WP-A0)

Levels: L1 same summary statistics → L2 same labels → L3 same input
data → L4 author-confirmed. Each claim cites measurement.

## L1 — Same summary statistics: STRONG MATCH

| Statistic | Paper (published) | Strict tensor (measured) | Match |
|---|---|---|---|
| Shape | 411×271, 5 types | 411×271×5 | exact |
| Typed links | 11,748 | 11,748 | exact |
| Typed-link density | ~10.5% | 10.5476% | exact |
| epigenetics proportion | 3.43% | 403/11,748 = 3.43% | exact |
| target proportion | 34.02% | 3,997/11,748 = 34.02% | exact |
| Other per-type | (paper table) | 1,155 / 403 / 2,293 / 3,997 / 3,900 | per-type counts consistent |

## L2 — Same labels: PLAUSIBLE, UNPROVEN

- Strict tensor is elementwise-identical to the SPLD-derived
  reference (data_audit PASS, sha 135e2050 → 85bb9ea4): lineage
  raw HMDD v3.2 → SPLD MDAv3.2-3 (<3-pair filter, 6th group dropped)
  → strict.
- Filter ambiguity stands: 2 miRNA have <2 pairs, 10 miRNA + 1
  disease have <3 → neither the paper's literal "<2" text nor the
  SPLD "<3" rule alone reproduces the boundary. UNRESOLVED which
  exact filter the paper applied; outcome matches anyway.
- Entity names match the public HMDD v3.2 vocabulary; whether the
  paper used identical entity ordering/selection is not provable
  from summary statistics alone.

## L3 — Same input data: NOT ESTABLISHED

- The author's evaluation input consists of tensor + per-fold
  similarity features + fold indices; similarity views are rebuilt
  independently (provenance-clean, not byte-identical).
- Byte-level equality of the author's fold files/feature matrices
  cannot be demonstrated without author artifacts.

## L4 — Author-confirmed: NO

- No author response. Draft request ready (ARTIFACT_REQUEST_TO_AUTHORS).

## Verdict

L1 exact + L2 plausible → the strict tensor is very likely the same
public benchmark the paper used, but equivalence of *evaluation
input* (folds/features) remains unproven. G1 is reclassified from
"tensor unknown" to "eval-input unverified".
