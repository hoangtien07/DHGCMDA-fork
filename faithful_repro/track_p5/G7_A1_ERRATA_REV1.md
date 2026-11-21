# G7-A1 Errata Rev-1 — corrections to G7_A1_RUN_RESULTS.md / GATE
Originals and all checkpoints preserved unchanged.

1. **Clip-active range corrected** to **74.3–84.2%** of steps
   (f1 535/650=82.3%, f2 364/490=74.3%, f3 547/650=84.2%, f4 544/650=83.7%).
2. **Gradient wording corrected**: reconstruction no longer dominates the
   measured conditioned gradient (recon 0.029–0.068 < type 0.24–2.67),
   but **intra-view remains larger than type on folds 1, 2 and 4**
   (intra 0.67–1.51 vs type 0.24–0.69); only fold 3 has type > intra
   (2.67 vs 0.93). A1 did not make the task gradient dominant on all folds.
3. **Convergence NOT established**: 3/4 A1 runs reached the 650-epoch
   ceiling (f1/f3/f4; f2 early-stopped at 490). Results are at-cap
   snapshots, not converged training.
