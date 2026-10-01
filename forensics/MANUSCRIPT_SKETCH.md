# Manuscript sketch — DHGCMDA reproducibility study (draft skeleton)

Working title: *Reproducing DHGCMDA: artifact recovery, evaluation audit, and a
minimal leakage-free baseline for multi-type miRNA–disease association
prediction on HMDD v3.2*

Venue candidates: ReScience C (replication track), or a
reproducibility/short paper at a bioinformatics venue (e.g. Briefings in
Bioinformatics reproducibility section, JBI letter, GigaScience technical note).

## Four contributions (narrative spine)

1. **Artifact recovery.** The paper's exact evaluation dataset (MDAv3.2-3:
   411 miRNA × 271 disease × 11,748 five-type triplets, with MISIM/DSSM
   similarity matrices) was believed unreleased; we recovered it byte-exact
   from the deleted companion repo SPLDHyperAWNTF_Model via Software Heritage
   (SHA256 c8f36c77…, committed under `HMDD_data/MDAv3.2-3/`).
2. **Reproducibility audit.** Anchored replication: our independent run of
   SPLDHyperAWNTF on the recovered artifact reproduces its Table-3 baseline
   row in DHGCMDA exactly (P=0.6219, micro-R=0.4624) — validating both the
   artifact and our evaluation harness. Under the same documented pair-fold
   protocol, DHGCMDA's reported Top-1 recall 0.9421 is *mathematically
   impossible*: a single prediction per test pair bounds macro-recall ≤
   0.8642 (micro ≤ 0.7435) on this artifact. The paper's evaluation therefore
   used code/metrics that were never released.
3. **Evaluation & provenance correction.** Provenance analysis shows
   `mi_fun_sim` = MISIM 2.0 (label-derived → leaky as a static input);
   DSSM is ontology-derived (conditionally static). Released DHGCMDA code
   cannot score v3.2 at all (hard-coded 4-type evaluator silently drops every
   sample). We define a three-track protocol: PRIMARY HONEST (train-only GIP
   + external DSSM), LEGACY COMPARISON (SPLD as published), LEAKY CONTROL
   (static MISIM).
4. **Minimal leakage-free baseline.** A zero-parameter similarity vote
   (row-normalized GIP over the train fold) reaches legacy-F1 = 0.5910 —
   the best observed performance among all tested methods, statistically
   indistinguishable from or above the published tensor SOTA
   (SPLD 0.5585; paired bootstrap with the dual-view vote: Δ=+0.0014,
   95% CI [−0.0075, +0.0104]). No statistically significant advantage over
   the similarity-voting baseline is detected under this protocol. The
   publicly available DHGCMDA implementation scores 0.4481 under honest
   evaluation — 11 points below the vote.

## Proposed structure

1. Introduction — task, the 0.86 claim, why it matters.
2. Materials — HMDD v3.2 lineage (raw 1049×758 → TDRC 713×447 → MDAv3.2-3
   411×271); the five association types; similarity sources and their
   provenance classes.
3. Methods — artifact recovery (Software Heritage); golden SPLD pair-fold
   evaluator (verified fold-for-fold); provenance audit; three-track
   protocol; baselines (vote variants, matrix factorization, encoders).
4. Results —
   T1: leaderboard (frozen, three tracks);
   T2: paired bootstrap Δ table;
   T3: vote ablation grid (GIP / DSSM / J / MISIM-static / MISIM-fold);
   F1: protocol recall-ceiling diagram (why R=0.9421 is impossible);
   F2: leakage channel quantification (static vs fold MISIM);
   F3: leaderboard bar chart by track;
   T4: per-fold + seed robustness table.
5. Discussion — where performance actually comes from (miRNA-side similarity
   + local propagation); implications for benchmark hygiene in MDA
   prediction; limits (private eval code never found; MISIM reconstruction
   approximate, corr≈0.80 to static).
6. Reproducibility statement — repo, ledger, deterministic seeds, runtimes.
7. Conclusion.

## Required wording discipline (from review)

- "we detect no statistically significant advantage over the
  similarity-voting baseline under this protocol" — NOT "adds nothing".
- "the publicly available/reconstructed DHGCMDA implementation" — NOT
  "the paper's own architecture".
- "best observed performance" for ~0.59 — "ceiling" is reserved for the
  proven 0.8642 macro-recall bound.
