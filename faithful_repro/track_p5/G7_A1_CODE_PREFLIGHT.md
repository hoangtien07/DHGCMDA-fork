# G7-A1 Code Preflight — amendment mechanism only, zero real-data steps

## Change (isolated)
- `p5_runner.py` sha256 `9ae82d66…` (was carrying no amendment mechanism).
- New `REGISTERED_AMENDMENTS = {'G7_A1_RECON_MEAN': {'recon_reduction': 'mean'}}`.
- `args_from_lock(..., amendment=...)` swaps exactly the registered keys in a
  real context; every other `LOCKED_HYPERPARAMS` entry is re-enforced after
  overrides AND amendment. Unregistered ids → `LockViolation`.
- CLI `--amendment` recorded in `resolved_args`, run manifest, result context;
  default out dir gains the amendment suffix (no overwrite of G6 artifacts).
- `faithful.py` already implements `l_rec` mean/sum via `--recon_reduction`;
  no model/loss-code change was needed there.

## Synthetic regression coverage (4 new tests, 88/88 suite PASS)
- amendment unlocks only `recon_reduction`; all other locks enforced.
- unregistered amendment id rejected.
- other locked-key override still rejected alongside a valid amendment.
- isolation: same-seed `sum` vs `mean` runs → identical step-0 `l_type`/`l_intra_w`, `l_rec_w` reduced by >100× under `mean`.

## Proposed command (NOT executed)
```
.venv-bank1/bin/python faithful_repro/track_p5/p5_runner.py --once \
  --fold F --seed 1234 --profile P5_BENCHMARK_UNKNOWN_PROVENANCE \
  --amendment G7_A1_RECON_MEAN \
  --out faithful_repro/track_p5/runs/g7a1_foldF_s1234 --max-wall-s 1800
```
for F ∈ {1,2,3,4} — four runs only, no fold-0 retraining.
