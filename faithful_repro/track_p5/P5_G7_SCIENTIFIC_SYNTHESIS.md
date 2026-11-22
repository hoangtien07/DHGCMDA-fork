# P5 × G7 Scientific Synthesis — three configurations, folds 1–4, seed 1234
All runs exploratory: folds were exposed during G5/G6; profile
`P5_BENCHMARK_UNKNOWN_PROVENANCE` ceilings every claim to
`exploratory_only`. SDs are population (ddof=0). No significance claims.

## Configurations
| cfg | objective | amendment |
|---|---|---|
| P5 | l_type softmax-CE + lam3·‖S−Ŝ‖²_F (sum) | — (locked baseline) |
| A1 | same, recon reduction **mean** | G7_A1_RECON_MEAN |
| A2 | A1 + 5 independent sigmoids, BCE-with-logits (pos_weight=w) | G7_A2_TYPE_SIGMOID_BCE |

## Primary metric (MULTIHOT_MACRO_F1, outer test)
| fold | P5 | A1 | A2 |
|---|---:|---:|---:|
| 1 | 0.4220 | 0.4385 | 0.4138 |
| 2 | 0.4205 | 0.4599 | 0.4199 |
| 3 | 0.4143 | 0.4664 | 0.4471 |
| 4 | 0.4248 | 0.4748 | 0.4181 |
| mean±sd | 0.4204±0.0038 | **0.4599±0.0134** | 0.4247±0.0131 |

Paired Δ: **A1−P5 = +0.0395±0.0141 (4/4 positive)**;
**A2−A1 = −0.0352±0.0145 (4/4 negative)**; A2−P5 = +0.0043±0.0167 (mixed).

## Per-class AUROC (mean, folds 1–4) — collapse tells the story
| class | P5 | A1 | A2 |
|---|---:|---:|---:|
| 0 | ~0.66 | 0.67 | 0.66 |
| 1 | ~0.71 | 0.81 | **0.72** |
| 2 | ~0.54 | 0.67 | 0.58 |
| 3 | 0.50–0.56 (class-4 <0.5!) | 0.58 | **0.67** |
| 4 | **0.415–0.466** | 0.60 | 0.57 |

Positive-pred rates (classes 2–4): P5 1.0 everywhere → A1 0.27–1.0
(real spread, τ off floor) → A2 ≈1.0 returned on folds 1,2,4.
Cardinality: P5 ~3.6; A1 ~3.0; A2 2.8–3.5.

## Gradient attribution (checkpoint-conditioned, same protocol)
| cfg | type grad | recon grad | reading |
|---|---:|---:|---|
| P5 | ~0.06–0.07 | **8.8k–16k** | recon dominates combined grad norm ~10⁵× |
| A1 | 0.24–2.67 | 0.03–0.07 | balance restored; intra>type on 3/4 folds |
| A2 | 0.07–0.16 | 0.02–0.08 | recon still small; type < intra on all folds |

Preclip grad norms: P5 10³–10⁵ → A1/A2 0.4–5.3 (clip active 74–84% vs 100%).

## Observations (measured)
- Recon sum→mean: largest single measured change; F1 up 4/4, gradient
  balance restored, decision collapse partially released.
- Sigmoid-BCE: ranking mixed (class-3 up, class-1 down), F1 down 4/4,
  decision layer re-collapsed — the pre-threshold score distribution
  changed in a way the fixed grid could not exploit.

## Supported interpretations
- H1 (recon distorts task gradients): consistent evidence — one locked-key
  change produced both gradient rebalancing and predictive gain.
- H3 (objective mismatch): NOT supported as stated — the independent-
  Bernoulli variant did worse at decision level here.

## Unproven hypotheses / open
- Whether A1's gain survives unseen seeds/fresh folds (all folds exposed).
- Whether calibration specifically explains A2's floor-τ behavior.
- Author-faithful reproduction and biological generalization — outside
  the scope of every measurement here.
