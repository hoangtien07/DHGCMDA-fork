# G7-A2 Errata Rev-1 — corrections to G7_A2_RUN_RESULTS.md / MECHANISM
Originals preserved; all checkpoints and results unchanged.

1. **P5 AUROC correction**: in G6/P5 it was **class 4** whose AUROC was
   below 0.5 on all folds (0.415–0.466), not class 3. The run report's
   comparison sentence that placed the sub-0.5 class with class 3 is
   corrected here.
2. **Ranking gains are NOT uniform**: class 3 gains AUROC/AP under A2
   (AUROC 0.64–0.68 vs A1 0.57–0.59; AP 0.33–0.68 vs 0.41–0.54 on f1,f4 —
   mixed on f2,f3), while **class 1 deteriorates** (A2 AP 0.108–0.115 on
   folds 2–4 vs A1 0.203–0.237; AUROC also down vs A1 on several folds)
   and class 2 shows no consistent gain.
3. **"Calibration failure" is NOT established** by threshold-floor
   behavior alone: τ at the grid floor shows the *decision* layer did not
   improve, but no reliability/calibration measurement was taken. The
   correct claim is that post-threshold F1 worsened; the mechanism
   (calibration vs. score distribution shape) is unresolved.
4. **A2 is not a pure link-function change**: it also changes objective
   normalization (per-pair softmax sum → element-wise BCE mean over a
   5-dim multi-hot vector) and positive-class weighting semantics
   (weighted CE on the positive term → `pos_weight` inside BCE on both
   terms). Effects cannot be attributed to the sigmoid link alone.
5. **Scope of the negative result**: A2 is negative *relative to A1* on
   these exposed folds under this exact configuration — it is **not** a
   universal refutation of independent-Bernoulli multilabel modeling.
