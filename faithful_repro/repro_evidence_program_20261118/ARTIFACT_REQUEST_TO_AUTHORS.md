# Draft request to DHGCMDA authors — NOT SENT (owner authorization required)

Subject: Request for v3.2 evaluation artifacts — DHGCMDA reproduction

Dear Dr. Sun and co-authors,

We are independently reproducing "DHGCMDA" (BMC Bioinformatics 2026)
for a reproducibility study. The released code covers HMDD v2.0; we
could not locate the v3.2 evaluation assets. Could you share:

1. The v3.2 label tensor or curation script producing the 411×271×5
   Table-1 dataset (entity lists for miRNA/disease and the typed-link
   table are sufficient).
2. The 5-fold split indices used for the v3.2 CV_type and CV_triplet
   evaluations (or the splitting script + seed).
3. The evaluation code that produced the v3.2 Top-1 precision/recall
   scores — the released `compute_top1_metrics` handles only four
   types and cannot score the five-type setting.
4. Clarification whether "four association types" in the evaluation
   section refers to the v2.0 label space, and how multi-type pairs
   (multiple association types for one miRNA–disease pair) were
   scored in Top-1.

We will cite any shared artifacts and report reproduction status
faithfully. Thank you.

— (unsigned; contact details to be filled by owner)
