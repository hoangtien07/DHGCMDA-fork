# G7 Closure Errata Rev-1 — corrections to synthesis + A2 reports
Historical artifacts preserved; numbers below verified against fold result.json.

1. **P5 per-class AUROC corrected**: class-3 AUROC = **0.5621–0.6471**
   (never below 0.5); class-4 AUROC = **0.4151–0.4656** (the sub-0.5 class).
   The synthesis table's class-3 row ("0.50–0.56") is corrected to this range.
2. **A2 class-3 AP improves on ALL four folds vs A1** (not mixed):
   A2 0.6204/0.5862/0.6789/0.6492 vs A1 0.5070/0.5348/0.5435/0.5368.
   The prior "0.33–0.68, mixed on f2,f3" statement is retracted — AP gain is
   uniform; AUROC gain is uniform too (0.642–0.683 vs 0.571–0.590).
   Ranking losses on classes 1 and 2 stand as reported.
3. **BCE `pos_weight` semantics**: `pos_weight=w` multiplies **only the
   positive loss term** (as `w·log σ(z)` in BCEWithLogits); negative terms
   are unweighted. The synthesis description is corrected accordingly.
