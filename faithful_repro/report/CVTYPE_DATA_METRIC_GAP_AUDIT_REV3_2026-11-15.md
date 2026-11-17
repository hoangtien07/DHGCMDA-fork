# CV_type v3.2 data & metric gap audit — REV3 — read-only — 2026-11-15

**Supersedes:** `CVTYPE_DATA_METRIC_GAP_AUDIT_REV2_2026-11-15.md` (sha
`e820c714…`) and REV1 (`4f4e3dfe…`), both preserved byte-identical.
**REV3 corrections:** (1) R1/R2 is a sensitivity comparison — no
bracketing or bounds claimed without formal assumptions; (2) held-out
*pair existence* vs held-out *type* distinguished — whether exclusion is
leakage depends on the declared CV_type deployment information set;
(3) association-derived artifacts labeled *unknown provenance / potential
label contamination* — transductive node access and outcome leakage are
separate dimensions; (4) selection phase is inner-training-only with an
exact post-selection refit rule; (5) fold grouping defined via connected
components of constraint edges; entity-overlap across a boundary is
allowed for distinct pairs unless a cold-start estimand is chosen;
(6) the v2.0 cross-paper comparison is descriptive numerics only.

**Scope:** read-only audit; no training, checkpoint loads, Colab, test
scoring, sweeps, rebuilds, or frozen-artifact edits.
**Tags:** **[F]** measured/file-backed fact · **[I]** inference · **[U]**
unknown · **[C]** contradiction.

Prior reads: `OVERNIGHT_PIPELINE_REPORT_2026-11-15.md`,
`PAPER_SCHEDULE_LITERATURE_CORRECTION_2026-11-15.md`,
`COORDINATOR_PROTOCOL_AUDIT_REVIEW_2026-11-14.md`,
`POST_PILOT_SOURCE_METRIC_GAPS.md`, `V32_CANONICAL_DATA_AUDIT.md`,
`OBJECTIVE_METRIC_DIAGNOSTIC_REV2_2026-11-14.md`,
`U1_650_INDEPENDENT_ASSESSMENT_REV2_2026-11-14.md`,
`V2_RECORD_IDENTITY_LEDGER.md`, `MISIM_SOURCE_RECORD_OVERLAP_AUDIT.md`,
`AUTONOMOUS_EXECUTION_STATUS.md`, CLAUDE.md Plans I–K.

Paper: [DHGCMDA, BMC Bioinformatics 2026](https://link.springer.com/article/10.1186/s12859-026-06436-w);
local text `lit/_pdf_text/p01–p42`; coordinator render
`lit/fig4b_coordinator_5x.png`. MISIM 2.0: Li et al.,
[PMC6602518](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602518/).

---

## 1. Ranked unresolved gaps

### G1 — Metric/target definition unresolved at the source (highest impact)

- **[F/C]** Internal v3.2 CV_type conflicts: Fig4(a) F1 = **.5678** vs
  Table 3 = **.8600** (coordinator-verified). Fig4(b) labels F1
  **.8768**, AUROC **.9420**, AUPR **.9439** vs Table 4 **.8674 /
  .9181 / .9271**. Neither can be silently selected as the sole target.
- **[F/C]** p24 describes CV_type over "**four** association types" while
  v3.2 has **five**; the released metric (`Calculate_Metrics.py`
  ~L309–331) hardcodes 4 types and drops every length-6 prediction →
  `valid_samples=0` → Top-1 F1 ≡ 0 on v3.2 (proven three ways, Plan K).
- **[I]** The reported numbers were therefore produced by something other
  than the released metric on the released data as inspected — an
  unreleased eval implementation, a different data/version artifact, or
  publication/transcription differences. Which applies is **[U]**.
- **[F]** Our benchmark (argmax-onehot vs multi-hot **macro-F1**, EN-
  weighted CE, `paper_guided/metrics.py`) is a declared new-group
  convention; identity to the paper's unnamed Top-1 averaging is not
  provable (micro-over-types, macro-over-pairs, per-fold-then-mean, and
  other conventions are arithmetically compatible with one scalar).
- **[I]** Any "gap to paper" is a gap to an unresolved definition AND
  unreleased curation — no %-reproduction figure is meaningful, and no
  convention may be selected post-hoc.

### G2 — Association-set curation unreleased and unmatched (high)

- **[F]** Three distinct v3.2 preprocessings exist (raw cuilab
  1049×758×18,084 @2.3%; TDRC-lineage 713×447×12,534 @3.9%; paper
  411×271×11,748 @10.5%). The paper's 411×271 filter is not recoverable
  → **[U]**.
- **[F/I — bounded]** The density-matched control (385×275 @10.3%)
  scored ≈ wang baseline (.3336 vs .3301): **bounded evidence** that
  matched density alone did not reproduce the gap at one design point.
  Not proof density is not a driver — entity composition confounds, one
  observation, one split/metric.
- **[F]** Strict GFF3-mapped rebuild (`85bb9ea4…`): 137,268 sequence
  cells differ (max Δ0.9024), 40 uncovered rows, 1,250 disease-view
  diffs ≤5.96e-8 — declared adaptation, not author-feature identity.
- **[U]** Paper's four views: only `S_func`/`S_sem` shipped matrices are
  direct evidence; `S_seq`/`S_gene` construction unrecoverable
  (`POST_PILOT_SOURCE_METRIC_GAPS.md` §1).

### G2a — Feature provenance and information regime are separate
###     dimensions (REV3 correction)

Two independent labels are required for every input feature; conflating
them was an REV2 error:

| Dimension | Levels |
|---|---|
| **A. Construction provenance** | (i) recipe recoverable & split-aware recomputable; (ii) documented external snapshot/provenance inspected; (iii) **unknown** |
| **B. Information regime** | inductive · transductive-node-access (unlabeled structure of held-out entities) · **potential label contamination** (association-derived features may encode held-out outcomes) · unknown |

- **[F/I]** `M_MISIM` traces to MISIM 2.0 computed from HMDD
  associations; `M_FSM` correlates with shipped-label disease overlap
  (Pearson 0.0898, Spearman 0.3327 — consistent with association-derived
  construction, unproven). → Label **A: unknown provenance / B: potential
  label contamination** — *not* automatically "transductive":
  transductive access to *unlabeled* test-node structure is a legitimate
  declared regime; encoding *held-out outcomes* inside features is a
  different, more severe concern, and the two must not be merged.
- **Evidence gates (fail closed):** (1) split-aware reconstruction —
  recompute the feature per fold from train-only associations with a
  traceable recipe; (2) inspected external provenance independent of the
  evaluated label set; (3) otherwise the arm runs only under a clearly
  labeled contamination-flagged regime and its numbers are reported
  separately. Unknown provenance defaults to the strictest label for any
  inductive claim; silence about the regime is not permitted.

### G3 — Split/protocol fidelity (high)

- **[F/U]** Paper: two five-fold protocols; authors' fold assignment
  unrecovered **[U]**. Our splits: declared adaptations, seeded
  whole-group split v2 (sha `074649bc…`) — not author folds.
- **[F]** Canonical audit: 9 same-MeSH disease groups; **26 ambiguous
  miRNA families / 80 rows**; grouped siblings' Y rows not identical →
  same-concept records can cross a random split → leak *risk* under the
  precaution grouping, magnitude unquantified **[I]**.
- **[F]** All arms to date: ONE fixed fold0 split × three seeds — seed
  sensitivity, not fold variability or a test estimate; selected maxima
  carry up-to-65-look selection bias.
- **[F]** Priors dominate type-any-hit: disease .6567 / product .6667 vs
  best model .5579 — any-hit alone does not separate the model from
  conditional priors.

### G3a — Negative sampling information regimes (REV3 correction)

Three regimes exist; none is claimed deployable, unbiased, or bounding:

| Regime | Definition | Information used | Open issues |
|---|---|---|---|
| **R1 — observed-zeros** | Negatives from cells not positive in the train partition; held-out mask never consulted. | Train positives only. | Held-out positives are sampled as negatives at their *actual* held-out prevalence — unknown. Train prevalence **estimates** that rate only under stated sampling assumptions (i.i.d. cells); under group splits the effective contamination rate differs systematically — an estimator, **not an upper bound**. |
| **R2 — mask-exclusion** | Negatives exclude held-out positive cells via the full label matrix. | Held-out labels. | **Whether this is leakage depends on the declared CV_type information set:** if the deployment task assumes pair *existence* known and predicts only *type*, excluding known-positive cells may be legitimate; if pair existence itself is held out, exclusion uses hidden outcome info. Claims of "optimism" or "deployability" require a stated deployment set — none is asserted here. |
| **R3 — declared-proxy** | Candidates restricted by a train-only rule (e.g., group-consistent zeros). | Train positives + grouping metadata. | Changes the negative population estimand; biases cross-arm comparison unless shared by all arms. |

- **Held-out existence vs held-out type (REV3):** CV_type hides *types*
  for pairs whose existence may or may not be treated as known; CV_triplet
  hides existence. The negative-bank rule, evaluation population, and any
  exclusion must be declared against one specific information set; the
  same exclusion can be legitimate under one and leakage under the other.
- **Prespecified proposal:** **primary = R1**, contamination rate
  *estimated* (not bounded) from train prevalence under declared sampling
  assumptions, with group-split sensitivity reported. **Sensitivity =
  R2**, labeled with its information-set dependence — R1 vs R2 is a
  **sensitivity comparison**: their difference measures regime
  sensitivity, and no direction or bound on contamination, optimism, or
  performance is claimed without formal assumptions.

### G4 — Objective/trajectory evidence internally consistent, bounded
###     interpretation (medium)

- **[F]** Weighted val CE worsens while argmax metrics improve at
  selected points (both arms); EN weighting concentrates excess on rare
  channel 1. Observed divergences only.
- **[F]** Static-graph arm: selected macro-F1 .4324 vs dynamic .4216
  (3 seeds, same split); final .4305 vs .3953; static CE slightly worse.
  **[I]** Conditional one-split sensitivity — not generalization, not a
  paper-ablation claim.

### G5 — Contradictions & non-findings kept visible

- **[C]** Fig4a .5678 vs Table3 .8600 — unresolvable from release.
- **[C]** "Four association types" text vs five-type release.
- **[C]** Released metric cannot score released v3.2 data.
- **[F — REV3 corrected]** `full_bilinear` scored .6350 vs the paper's
  reported .597 on v2.0 Top-1: **descriptive numerics only** — our
  corrected general-K metric is arithmetically identical to the released
  metric on 4 types **[F]**, but paper fold assignment and averaging
  convention are unrecovered **[U]**, so cross-paper metric alignment is
  not established and no "exceeds paper" claim is made. The v3.2
  ~.27–.36 range under our declared metric is likewise descriptive.
- **[F]** HMDD v4.0 raw acquired — separate study, never a v3.2
  substitute.

## 2. Prespecified bounded evaluation contract (proposal — nothing
##     runs without a separate gate)

**Estimand declaration:** pair-level CV_type under an explicitly declared
information set — held-out **pair-type** records are scored; whether pair
*existence* is treated as known is declared per the regime, matching the
negative-bank regime chosen (G3a). Entity-disjoint (cold-start) is **not**
the default estimand; grouped entities may legitimately appear on both
sides of a boundary for *distinct pairs* unless a cold-start estimand is
explicitly substituted.

**Fold construction:** constraints are "must-be-same-fold" edges between
entities (26 ambiguous miRNA families, 9 same-MeSH disease groups, plus
any declared pair-level constraints). Folds assign **connected
components** of this constraint graph atomically by a seeded rule; the
component list, the fold-assignment file, and the group ledger are
hash-pinned BEFORE any training.

**Inner/outer split:** each fold's held-out segment is outer-eval only;
each fold's training partition splits into inner-train / inner-val by the
same component-atomic rule (a second, disjoint split file, hash-pinned).

**Selection phase (inner-training-only):** while selecting candidates,
*all* derived objects — association-derived features (when recomputable),
negative bank, EN weights/mass, prior statistics — are constructed from
**inner-train labels only**; inner-val is touched solely to score and
rank candidates. No candidate, including priors, is fit on inner-val
labels; priors use exactly the available labels the model sees and the
identical information set.

**Post-selection rule (declared, choose once before runs):** primary =
**refit**: after a candidate is selected, features/negatives/EN/priors
are recomputed on the *full outer-train* partition (inner-train ∪
inner-val) and the selected candidate is retrained with locked
hyperparameters and the declared horizon before a single outer-eval
scoring. Alternative = no-refit, evaluated identically; whichever is
declared applies uniformly to all arms. No per-arm refit choice.

**Arms (frozen before run):** (1) dynamic paper-guided pipeline (sealed);
(2) static-graph control (sealed `6e768d0e`); (3) train-only conditional
priors — disease, miRNA, product — fit on outer-train observed positives
with declared smoothing constants, matched information set; (4) declared
degenerate floor.

**Feature-regime rule (per G2a):** every arm carries labels A and B.
Unknown-provenance association-derived artifacts run only in a
contamination-flagged regime, reported separately, never ranked against
clean-provenance arms.

**Predeclared endpoints (all reported regardless of outcome):** full val
curves, final-epoch AND max-selected metrics (labeled order statistics),
macro/micro-F1, any-hit, weighted CE + floor, per-class F1/TP/FP/FN,
paired hit-overlap vs priors, and the R1/R2 sensitivity pair. Primary
metric declared NOW: argmax-vs-multi-hot **macro-F1**. AUROC/AUPR only
under an explicitly declared convention, never mixed with paper numbers.

**Metric diagnostics:** alternative definitions (micro-over-types,
macro-over-pairs, pooled-vs-per-fold averaging, hybrid Top-1 P/R/F1
formulas) may be defined a priori and computed on frozen exports as
**diagnostics only**, with explicit limits: no arithmetic match can
identify the paper's convention; no mismatch proves the converse; no
diagnostic may select a convention or support a reproduction claim.

**Completion & stop conditions:** all 5 folds complete for every arm
regardless of interim outcomes — underperformance is a reported result,
never a stop; no fold abandoned for unfavorable numbers. Hard stops are
integrity/safety only: hash drift, gate/lineage failure, corrupted
inputs. An integrity stop yields a labeled partial-protocol report — no
post-hoc completion or redefinition.

## 3. Prioritized next actions

1. **[Highest]** Metric-definition document enumerating plausible Top-1
   conventions a priori; compute each on existing frozen exports as
   diagnostics under the no-identification limits — artifact-only.
2. Quantify group/family crossing mass under the component-based split —
   measurable from the split file + ledger, no refit.
3. Provenance inventory labeling every feature on dimensions A and B
   before contract sealing.
4. Publish fold-assignment + component-ledger + prior-fitting-rules +
   refit-rule package for coordinator review BEFORE any fold run.
5. Defer HMDD v4 and author-contact separately.

## 4. Explicit non-claims

No reproduction percentage; no chosen-figure target; no claim the paper
is wrong; no claim any frozen feature is leak-free or that
unknown-provenance features are merely transductive; no claim R1/R2
bound anything — they are a sensitivity pair; no claim any regime is
deployable or optimistic absent a declared information set; no claim the
density control proves density irrelevant; no claim identifying the
authors' eval code; no claim v2.0 numerics "exceed" the paper given
unaligned metrics/folds. This contract authorizes no jobs.
