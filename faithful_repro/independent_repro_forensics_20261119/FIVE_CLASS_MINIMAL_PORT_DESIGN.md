# FIVE_CLASS_MINIMAL_PORT_DESIGN — minimal-change executable port (design only, NOT authorized)

Port scope: smallest diff to released pipeline that (a) accepts K=5 typed
labels, (b) scores with a defined evaluator. NO implementation yet (WP4 gate).

## Changes required vs release
| site | release (K4) | port (K5) |
|---|---|---|
| predictor output | existence + 4 | existence + 5 (full bilinear per type) |
| type loss target | scalar 1..4 | OPTION A: first-positive scalar 0..4 / OPTION B: multi-hot weighted CE |
| label tensors | v2.0 495×383×4 | strict 411×271×5 |
| evaluator | K4 hardcoded | locked K5 suite (Track N metrics_g2c formulas) |
| folds | shipped KFold | recovered locked partitions |
| views | bundled v2.0 | PUBLIC_RECONSTRUCTED_4VIEW (clean claim) or bundle views (benchmark claim) — must be named per run |

## Must-fail tests (before any training)
- loss denominators on a 2-label row (fixtures exist)
- mask check: no train step may read held-out labels (canary exists)
- K=5 predictor dims (5 type logits + existence)
- graph refresh touches edge_index only, not prebuilt G
- similarity scale in [0,1] fixture
