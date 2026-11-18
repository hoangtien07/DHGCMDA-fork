# AMENDMENT — CVtriplet-U1 REV2, blocker B1–B4 (COUNCIL conditional approval 2026-11-18)

Verdict: CONDITIONAL_APPROVAL_FOR_5X3. REV2 design/model/hypothesis/
objective/seeds/split/horizon/primary-metric UNCHANGED. This amendment
records the four engineering fixes required before the full run.
PROTOCOL_CVTRIPLET_U1_REV2.md is not edited (history preserved).

## B1 — Provenance / status decomposition

- `baseline_logistic.py` was edited after the fold0 sanity run (tuple
  unpack fix in `BL.run`). New source hashes are pinned by a fresh
  preflight; the old `preflight.json` is preserved as
  `preflight_rev2_sanity.json`.
- fold0 crash history kept: `runs/fold0_seed1234/exception.txt`,
  `completion.json` (TRAIN only), `eval.json` (sanity schema),
  `verification.json` (sanity schema) — none edited.
- Run status is decomposed: `TRAIN_PASS` (completion.json),
  `EVAL_PASS` (`eval_full.json` + `eval_arrays.npz` written),
  `VERIFY_PASS` (`verification_full.json`). `status.json` derives
  `RUN_PASS = all three` — no single-word status anymore.

## B2 — Baseline negative pools

Baseline fit negatives previously sampled from the whole `~Y`
(transductive-oracle: includes held-out positives AND eval negatives —
not information-matched to U1X).

Fixed to two explicit variants, reported separately:

- `mask_zero_pairs` (INFO-MATCHED): negatives = triplet cells on
  pairs absent from train_mask — the same negative pool U1X's
  existence head sees (PU semantics: held-out-positive pairs are
  mask-zero, identical to upstream zero_index).
- `all_unknown` (ORACLE-ONLY, reported separately, never used for
  the info-matched comparison): the old `~Y` sampling.

train/eval negative overlap is measured per replicate and reported.

## B3 — Artifact-level verification

`run_eval` now exports `eval_arrays.npz` per run: for every eval
replicate — pos_idx, neg_idx, y_true, strata codes, U1X scores, both
baseline scores. `verify.py` recomputes AUROC/AP/F1@0.5 from those
arrays independently (sklearn, fresh call) and asserts bitwise/1e-12
agreement with `eval_full.json`. The synthetic metric fixture is no
longer counted as experiment-result verification (it stays as a
unit fixture only).

## B4 — Full launcher

`run_5x3.py`: iterates the 15 locked contracts, ≤2 workers via
subprocess, per-job `runs/foldN_seedS/` with own launch.json /
progress.log / completion.json / eval_full.json / eval_arrays.npz /
verification_full.json / status.json. Existing run dirs are NEVER
overwritten — occupied dirs are skipped and recorded; fold0 keeps its
sanity artifacts and gains eval_full/verification_full alongside.
Any provenance drift or verification failure stops that job and is
recorded as failed — supervisor does not self-fix.
