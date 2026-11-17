# Missing author information needed to resolve reproduction

Prepared 2026-11-16. This is a local question list; nothing has been sent.

1. Could you provide the exact processed HMDD v3.2 tensor used in Table 3, ordered miRNA/disease/type names, and its SHA256? Our recovered SPLD-lineage tensor matches all Table 1/2 counts but this does not establish record identity.
2. Was tissue included in CV_type evaluation? Table 1/2 specify five types, but section 3.2 says four and the public model/metric hardcode four.
3. How are multiple true types handled for a test pair? Please provide the Top-1 P/R/F1 implementation, channel order, zero-division policy, fold aggregation and per-fold scores/predictions.
4. Table 3 has v3.2 P/R/F1 .7915/.9421/.8600; the supplied article-in-press PDF Fig4(a) has .6005/.5253/.5678. Were these produced with different configurations or is a correction available?
5. Does the two-association filter count raw experimental evidence records, unique pairs, or unique typed links? Before or after mapping/filtering? The recovered tensor includes hsa-mir-1179 and hsa-mir-1244 with one unique typed link each.
6. Please provide the sequence, functional, gene and semantic matrices, database versions and entity-mapping rules; indicate whether association-derived features use all pairs or only each fold's training associations, and whether held-out pair existence is assumed known.
7. For Eq32-34, what reductions and weight normalization were actually used? Are weights rescaled to sum to C? How are the reconstruction masks, InfoNCE sum/mean, margin reduction, existence objective and label smoothing handled? The released code and equations differ.
8. Please provide fold assignments/seeds, inner-validation procedure, checkpoint selection, training horizon, optimizer/scheduler, and environment for the Table 3 experiment. Is the released main commit the code that generated v3.2 results, or is a separate version available?
