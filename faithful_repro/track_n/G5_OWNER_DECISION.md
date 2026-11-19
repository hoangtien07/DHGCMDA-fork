# G5_OWNER_DECISION — required before any cold-start training

G4 completed: A1 T-pair fleet done, H1 REJECTED (−0.0136 vs +0.03 target).

Cold-start deep-model training (mirna_cold / disease_cold) is **NOT AUTHORIZED** and remains blocked until the owner decides:

Options for owner:
1. **STOP Track N here** — G4 gave a clean negative on the pair track; cold-start arms would test the same MLP design whose pair-track hypothesis already failed.
2. **Authorize cold-start anyway** — justified only if the cold-start hypothesis is considered independent (similarity-row information may matter relatively more for unseen entities; note corrected B2_labelprop already leads mirna_cold at 0.4569).
3. **Redesign arm first** — e.g. the original U1-embedding input spec, which requires its own embedding fleet (larger budget, new approval).

Recommendation [suy luận]: option 1 or 3 — A1's input space (raw similarity rows) is what B1 logistic already exploits linearly; the small A1−B1 gap (−0.01) suggests limited non-linear signal in those features. A cold-start rerun of the same input spec is unlikely to flip H1's sign.
