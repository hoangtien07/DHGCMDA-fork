# REVIEW — U1_FIDELITY_GAP_AUDIT + FB1 implementation (2026-11-15)

Independent audit of `report/U1_FIDELITY_GAP_AUDIT_2026-11-15.md` and,
per coordinator follow-up, the FB1 implementation:
`stages/full_bilinear_readout.py`, `nested_u1_fb1_runner.py`,
`stages/config_u1_fb1.json`, `stages/test_fb1_readout.py`,
`build_fb1_pkg.py`, `report/FB1_FULL_BILINEAR_PROTOCOL_2026-11-15.md`.
Verified against executed sealed source and manifests; fixture probes
run (no fits, no eval scoring, no edits).

## 0. VERDICT

**Audit: accurate and well-tagged.** Its equation↔code table matches
what I independently verified across the earlier review chain
(inter objective, readout parameterization, dynamic refresh, CE-on-
positives, "NO existence head"), and its [F]/[I]/[U]/[C] discipline is
sound — every "no demonstrable defect" claim survives: remaining
deviations are declared contracts, paper-internal conflicts, or
genuine underspecification.

**FB1 implementation: correct, launch-ready, minor wording fixes
only.** 13/13 own tests + all my fixture probes pass; injection,
provenance, and packaging are consistent with the REV4 contract
pattern. One MEDIUM-LOW confound understatement (I1) and small nits.

## 1. AUDIT-DOCUMENT VERIFICATION

- **R1 claim verified**: `DiagBilinearType` is `Σ_d mi_d·W_kd·di_d`
  (model.py:184-188 einsum `pc,tc,pc->pt`) — diagonal special case of
  the standard `z_miᵀW_kz_dis`. Param counts correct (5×128=640 vs
  5×128×128=81,920 readout-W). "Silently chosen" is slightly strong —
  the diagonal form is declared in the module docstring; "undeclared
  relative to the paper" is precise.
- **Paper wording**: reading "bilinear predictor" (§2.7) as a dense
  per-type matrix is the defensible term-of-art; the audit correctly
  tags it [U] and never claims the diag arm is "wrong".
- **v2.0 comparability**: audit correctly limits J-1 (full .6350 vs
  diag .5996) to "descriptive numerics — cross-pipeline alignment
  unestablished, untested on U1/v3.2". This restraint is right: J-1 ran
  on a different pipeline/metric regime; only the *direction* (rank
  matters) carries over.
- **R2 disposition correct**: existence head is paper-internal [C]
  (§2.7 promises, Eq.32 absent); properly deferred to coordinator.
- **R3/R4/R5**: recon targets (mi functional + dis semantic,
  coverage-masked), margin fallbacks, and selection convention all
  match the declared config/manifests I verified earlier.
- **FB1 design quality**: consumed-splits-only framing, order-statistic
  labeling, "sensitivity not improvement" language, integrity-only
  stops — all correctly bounded.

## 2. FB1 IMPLEMENTATION — verified (fixture probes, executed)

- **Math**: `einsum('pc,tcd,pd->pt')` = `mi[p]ᵀW_tdi[p]/T` — manual
  per-type loop reproduces logits exactly (probe P4); same proj
  stacks + learned-T clamp as sealed readout.
- **Diag anchor**: `PaperGuidedC4FB('diag')` — state_dict bitwise
  identical to `PaperGuidedC4` **and identical RNG stream position**
  (probe P1 — flag-off consumes zero extra draws).
- **Full arm init**: all 50 non-readout state entries identical to
  arm-A init (P2); `type_W` (5,R,R) present, `type_rel` gone; param
  delta scales correctly (toy +1,200; real +81,280 = 5·128·127).
- **Optimizer coverage**: post-swap `named_parameters` exposes
  `readout.type_W` + proj params only (P3) — discarded diag params
  never reach the optimizer (module replaced before opt build).
- **Determinism**: same seed → identical full init (P6).
- **Injection works**: training code resolves `c4m.PaperGuidedC4`
  lazily (u1_runner.py:379 import + :412 call) → the in-process
  monkeypatch produces `FullBilinearType` readouts (P5); restored in
  `finally`.
- **Config**: diff vs executed arm-A manifest cfg = exactly
  `{stage:'U1-FB1', readout_mode, fb1_note}` — all else byte-identical
  (seed 1234, 650ep, all λs, dynamic 5/.5, unit_mode).
- **No gate breakage**: `stage` is hardcoded 'U1' in
  manifest/ckpt/meta (u1_runner.py:179/225/357/629/701) — cfg's
  'U1-FB1' does not reach the expected-manifest stage check (which
  compares literal 'U1'); cfg incl. readout_mode is recorded
  identically in frozen+expected → contract consistent. Preflight has
  no unknown-key rejection → `readout_mode` passes.
- **Runner**: fails closed without `readout_mode:full_bilinear`;
  `train_nested(cfg,data,None,out)` = arm-A path (signature verified);
  `--source-contract` verified **before** np.load; fb1 provenance
  block appended post-run (runner+readout shas, exposure text).
- **Source contract**: `--emit-source-map` yields 25 keys incl.
  `stages/full_bilinear_readout.py` + runner itself; map is
  launcher-agnostic (path-relative to module dir → works under
  `/content/fb1_pkg/src`); exact key-set match required.
- **Package**: sealed `pkg_root/src` byte-for-byte + 3 FB1 files +
  bundle npz (sha `a9e8e2b2…` via `--data-sha`) + recon meta/audit
  (required at lam_recon=1.0) + manifest-hash gate (fail-closed on any
  file mismatch) + pinned deps matching `env_require` + no-clobber out.
- **Tests**: 13/13 pass (hand-computed logits, per-type grad
  independence, gradcheck, anchor equality, param delta, mode
  validation).

## 3. FINDINGS

**I1 (LOW-MED) — RNG/init confound understated.** Protocol §5
limitation reads "extra draws for W_t". In fact the **entire readout
module** is re-drawn under 'full': `mi_proj`, `dis_proj`, `type_W`,
`temperature` (~66k + 82k params) are drawn after base construction,
while arm-A's proj stacks (drawn mid-construction) are discarded. So
even the projection weights differ between arms at init — the
init-time confound is the whole readout, not only W_t. This is the
correct design (diag proj can't seed a full W), but the limitation
sentence should say "all readout parameters re-initialized" — a
reader could otherwise assume proj stacks were shared/identical at
init.

**I2 (LOW) — audit "silently chosen" phrasing.** The diagonal form is
declared in `DiagBilinearType`'s own docstring; the gap is that the
*paper-facing* choice was never justified. "Undeclared relative to
paper §2.7" is more accurate than "silently".

**I3 (LOW) — `readout_mode` attribute asymmetry.** `PaperGuidedC4FB`
sets `self.readout_mode` only in the 'full' branch — harmless
(diag path never reads it) but an explicit `'diag'` attr would make
introspection symmetric.

**I4 (informational) — provenance coverage is adequate but indirect.**
`manifest_frozen.json`/`ckpt` carry `cfg.readout_mode` (sufficient);
the explicit `fb1` block lives only in `manifest.json` — consistent
with the arm-B amendment pattern. Emitted contract keys are OS-path-
relative (backslashes on Windows) — emitted+verified on the same host,
fine.

**I5 (informational) — shipped-but-unpinned file.**
`stages/test_fb1_readout.py` is in the pkg but not in the source
contract (never imported at run) — correct behavior, noted for
completeness.

## 4. ANSWERS TO THE SPECIFIC CHECKS

- *Runner injection*: works — lazy `c4m.` resolution + in-process
  patch + finally-restore (probed).
- *Optimizer/seed effects*: type_W participates; non-readout init
  identical; deterministic; confound = whole-readout re-init (I1).
- *Source/input pinning*: 25-key contract verified-before-load;
  data-sha `a9e8e2b2…`; pkg manifest gate fail-closed.
- *Matched comparison*: sealed arm-A reused verbatim — clean control;
  same selection convention → comparable order statistics, both
  exposed (descriptive, as framed).
- *Paper wording*: audit's [U] tagging is correct; FB1 measures one
  plausible reading's consequence — properly scoped.
- *Forward/gradient tests*: 13/13 + independent hand-computed probe —
  correct.
- *Production launch readiness*: ready after gates — package,
  contract chain, fail-closed paths all verified; only I1's wording
  fix recommended before launch.

## 5. RECOMMENDATIONS

1. Amend protocol §5 + audit: "the entire readout (proj stacks, W_t,
   temperature) is re-initialized under 'full'" (I1).
2. Optionally set `self.readout_mode='diag'` in the diag branch (I3).
3. Soften "silently chosen" → "undeclared relative to paper §2.7" (I2).
4. Launch: proceed — no blocking defects.
