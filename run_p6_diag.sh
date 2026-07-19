#!/usr/bin/env bash
# Task A (P6): diag predictor under CANONICAL honest config (full_bilinear baseline already = gap_leakfree_s*).
# Canonical = full_bilinear|diag, K=2, --cv_scheme full, --leakage_free, exist_weight default 0.3, 650ep x 5fold.
# A/B: compare these diag runs vs existing full_bilinear leakage-free (leakage_gap.json / gap_leakfree_s*.log).
set -u
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-10}"
mkdir -p results/p6 logs
for SEED in 0 42 1234; do
  log="logs/p6_diag_lf_s${SEED}.log"
  echo ">>> [P6] diag leakage-free seed=${SEED} -> ${log}"
  DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
     --device cpu --predictor_mode diag --K_neigs 2 --cv_scheme full --leakage_free \
     --seed "${SEED}" --epoch "${EP}" --validation "${FOLD}" 2>&1 | tee "${log}"
  venv/bin/python parse_metrics.py "${log}" "results/p6/diag_lf_s${SEED}.json" 2>&1 | tail -2
done
echo "=== [P6] diag leakage-free DONE ==="
