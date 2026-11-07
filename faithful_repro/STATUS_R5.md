# Faithful reproduction of DHGCMDA — status at pause (Round 5 partial)

Snapshot of the paper-faithful re-implementation + all results to date, pushed so work can continue locally.
Original working dir on the Devin VM: `~/dhg_runs/faithful` (this folder), `~/dhg_runs/lit` (= `lit/`), upstream clone `~/repos/DHGCMDA` (commit 06f9ba3).

## Layout
| Path | Content |
|---|---|
| `code/faithful.py` | paper-faithful model + CV_type / CV_triplet protocols (primary). Flags added in R5: `--predictor {diag,full}` (default diag), `--recon_reduction {sum,mean}` (default sum), `--cap_threads` (default 0 = off; on changes numerics, no speedup) |
| `code/faithful_r4.py`, `faithful_round2.py` | frozen earlier versions (R4 default output is bit-identical to `faithful.py` default) |
| `code/build_datasets.py`, `build_4view.py` | build `data/*.npz` from CUILab raw (`hmdd_v32_raw/`), upstream v2.0, fork `HMDD_data/MDAv3.2-3`; 4-view via miRBase NW + CTD/HumanNet-XC |
| `code/tdrc_baseline.py` | B7: TDRC (vendored `lit/TDRC`, unmodified) on the SAME splits/metrics as faithful.py |
| `code/compare_r4.py`, `compare_r5.py`, `aggregate.py` | aggregation; `compare_r5.py` writes `report/results_r5.md` |
| `code/ref_provenance.py`, `entity_rule*.py` | B1 provenance / entity-rule search (uses `lit/cuilab_old/`) |
| `data/` | processed datasets (npz + json metadata) |
| `hmdd_v32_raw/` | CUILab HMDD v3.2 `v3_*` downloads + SHA256SUMS; `lit/cuilab_old/` = unprefixed CUILab category files (source of the 411×271 artifact triplets) |
| `report/` | all reports: `REPRODUCE_DHGCMDA_FAITHFUL.md`, `GAP_LITERATURE.md`, `ROUND4.md`, `COUNCIL.md`, `B1_entity_rule.md`, `PAPER_INTERNAL.md`, `PREREG_R5.md`, `results_r5.md`, `email_authors.txt` (draft, NOT sent) |
| `results*/`, `logs*/` | per-run JSON (per-fold + summary) and logs for rounds 2–5 and B10 benchmark |
| `round1_upstream/` | round 1: unmodified upstream code results |
| `jobs_r5.txt` | pre-registered 165-job R5 matrix; `jobs_r5_remaining.txt` = 117 not yet run |

## Paths / environment
Scripts hardcode `~/dhg_runs/faithful`, `~/dhg_runs/lit`, `~/repos/DHGCMDA`, `~/dhg_runs/venv_np1/bin/python`. Locally either recreate that layout (symlink `~/dhg_runs/faithful` → this folder, `~/dhg_runs/lit` → `lit/`) or replace those paths. Env: Python 3.12, numpy 1.26.4, torch 2.5.1+cpu, torch-geometric, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, tensorly 0.10.0 (TDRC). numpy<2 is required by upstream code (`np.mat`).

External files NOT committed (size) — re-download and check against `ext/SHA256SUMS`, `ext/seqgene/SHA256SUMS`:
- `ext/d2025.bin` https://nlmpubs.nlm.nih.gov/projects/mesh/2025/asciimesh/d2025.bin
- `ext/misim_similarity.zip` (unzip to `ext/misim/`) http://www.lirmed.com/misim/similarity.zip
- `ext/seqgene/HumanNet-XC.tsv` https://www.inetbio.org/humannetv3/networks/HumanNet-XC.tsv
- `ext/seqgene/CTD_curated_genes_diseases.tsv.gz` http://ctdbase.org/reports/CTD_curated_genes_diseases.tsv.gz
- `ext/seqgene/mature.fa.gz`, `hairpin.fa` https://www.mirbase.org/download/
They are only needed to REBUILD `data/`; the committed npz files are enough to train.

## Round 5 progress (pre-registered in report/PREREG_R5.md; seeds 1234,1,2 × 5 folds)
Done 48/165 (all exit 0): B2 (ref 4-view), B7 (TDRC, all 3 datasets), B3 (recon mean, all 3 datasets), B4 full predictor on v2.0.
Stopped (killed mid-run, partial logs only, no JSON): B4 on v3.2_cuilab_deg2 / ref.
Not started: rest of B4, B9a ablations, B9b K sweep, B9c (t, λ2) grid. B8 (val-threshold F1) is computed inside every triplet job.

Resume (from this folder, after fixing paths):
```
xargs -P 8 -L 1 ./run_r5.sh < jobs_r5_remaining.txt
~/dhg_runs/venv_np1/bin/python code/compare_r5.py   # -> report/results_r5.md
```

## Interim R5 findings (see report/results_r5.md; seed-level paired deltas, pre-registered rule |Δ|>2·sd and same sign in 3 seeds)
- B7: TDRC on the same splits/metrics beats faithful DHGCMDA on v3.2 (cuilab Top-1 F1 0.340 vs 0.257; AUPR 0.921 vs 0.898; ref Top-1 F1 0.351 vs 0.273) and on v2.0 micro Top-1 F1 (0.468 vs 0.378); DHGCMDA is better on v2.0 AUPR (0.919 vs 0.905).
- B3: recon mean-reduction raises Top-1 F1 on all datasets (v2.0 0.388→0.518) — sensitivity only; still below paper 0.597 / 0.860.
- B4: full bilinear on v2.0 Top-1 F1 0.388→0.435 (+ all seeds, below 2·sd rule for macro F1).
- B2: ref 4-view slightly worse than 2-view (Top-1 F1 −0.014, AUPR −0.007).
- B8: test-optimised F1 exceeds validation-threshold F1 by +0.0065 on average (15 jobs).
- Primary results unchanged (faithful default config); none of the sensitivities reaches the paper's numbers.

`lit/TDRC/` is a vendored copy of https://github.com/BioMedicalBigDataMiningLab/TDRC at commit 1de1a87 (unmodified; compat shims live in `code/tdrc_baseline.py`).
