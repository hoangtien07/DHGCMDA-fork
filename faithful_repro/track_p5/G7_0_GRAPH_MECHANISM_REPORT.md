# G7.0 Graph Mechanism Report — persisted selected G/ei (folds 1–4)

## Selected-epoch connectivity (ei edge counts)
| fold | assoc (both dirs) | miRNA-sim | disease-sim |
|---|---:|---:|---:|
| 1 | 6289 | 63 728 | 7 412 |
| 2 | 12 578 | 15 322 | 148 082 |
| 3 | 6289 | 8 968 | 4 016 |
| 4 | 6289 | 10 134 | 4 554 |

(assoc doubles only when direction pairs merged differently; see raw JSON.)

- Densities vs max (mi²≈168k, d²≈73k): fold1 miRNA-sim ≈38%, disease-sim ≈10%; fold2 disease-sim ≈**148k/73k — exceeds N(N−1)**, i.e. duplicate/self edges persist under θ-filtering of a dense reconstruction; folds 3–4 sparse (≤6%).
- The selected graphs span extremes (6% → >100% dense) across folds — connectivity at checkpoint-selection time is wildly heterogeneous, consistent with G6's refresh-oscillation evidence. NO causal claim on F1 made.
- S reconstruction distribution (eval-mode forward, mm01): S_m median ≈0.40, ~30% mass inside [0.4,0.6], ~33% ≥0.5; S_d median ≈0.25, ~14% inside [0.4,0.6], ~9% ≥0.5. The learned similarities sit within a narrow sub-range hugging the 0.5 boundary — small score drift flips tens of thousands of edges, mechanistically consistent with refresh instability (supportive, not causal proof).
