# G5.1 Prediction-Collapse Audit — fold0 × seed1234 (POST_HOC)

Label: **POST_HOC_DESCRIPTIVE_DIAGNOSTIC** — not a preregistered benchmark.

## Independent recompute (from committed TP/FP/FN + fold manifest y)
n_test = 1747; support = [201, 74, 471, 788, 772]

| class | model P | model R | model F1 | pred-pos rate | always-pos null F1 | Δ(model−null) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.1607 | 0.5771 | 0.2514 | 0.413 | 0.2064 | **+0.0450** |
| 1 | 0.0811 | 0.3649 | 0.1327 | 0.191 | 0.0813 | **+0.0514** |
| 2 | 0.2696 | **1.0000** | 0.4247 | **1.000** | 0.4247 | **0.0000** |
| 3 | 0.4511 | **1.0000** | 0.6217 | **1.000** | 0.6217 | **0.0000** |
| 4 | 0.4419 | **1.0000** | 0.6129 | **1.000** | 0.6129 | **0.0000** |

- Recomputed F1 identical to stored (self-consistent artifacts).
- **Null always-positive macro-F1 = 0.3894; model = 0.4087; Δ = +0.0193.**
- Entire margin comes from the two low-support classes (0,1).

## Why the three high-support classes equal the null
Classes 2–4 have fitted thresholds 0.05 (grid floor). At τ=0.05 every test pair's
softmax probability exceeds τ → predicted positive on 100% of pairs → recall=1,
precision=support/n — numerically identical to an always-positive predictor. The
val-fitted τ did not lift above the floor because inner-val F1 is also maximized
by predicting all-positive when a class's positives are ~30–45% of pairs and the
model's score ordering is weak.

## Reading
The 0.4087 is real as measured, but ≈97% of it is obtainable by predicting
positive everywhere on 3/5 classes. Discriminative signal concentrates in
classes 0 and 1 (Δ +0.045/+0.051), which are the minority classes.
