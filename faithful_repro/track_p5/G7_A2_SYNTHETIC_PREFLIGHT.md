# G7-A2 Synthetic Preflight — code + fixtures only, zero real-data steps

## Coverage (6 new tests, suite 94/94 PASS)
- `test_a2_amendment_unlocks_exactly_two_keys` — only recon_reduction + type_objective change; all other locks enforced.
- `test_a2_only_via_registered_amendment` — locked without amendment; real-context override rejected; A1 resolves softmax_ce unchanged.
- `test_a2_sigmoid_probs_not_softmax_and_logits_preserved` — output = sigmoid(logits/temp) elementwise; last_logits stashed; BCE on logits ≠ BCE-on-probs (unnormalized logits genuinely used).
- `test_a2_synthetic_run_stable_multi_positive` — full train_fold completes, finite F1, cardinality > 1 (multiple positive type predictions per pair).
- `test_a2_test_label_perturbation_invariance` — stripping test labels does not change fitted thresholds or outcome.
- Dry-run regression (earlier fix) — amendment reflected in resolved args: verified `recon_reduction=mean`, `type_objective=sigmoid_bce`, `lam3=1.0`.

## Integrity
faithful.py untouched (patch is instance-level in p5_runner only);
checkpoint/G/ei persistence path unchanged; G5/G6/G7-A1 artifacts intact.

## Proposed command (NOT executed)
```
p5_runner.py --once --fold F --seed 1234 --profile P5_BENCHMARK_UNKNOWN_PROVENANCE \
  --amendment G7_A2_TYPE_SIGMOID_BCE --out runs/g7a2_foldF_s1234 --max-wall-s 1800
```
F ∈ {1,2,3,4}, paired against G7-A1 results on the same folds.
