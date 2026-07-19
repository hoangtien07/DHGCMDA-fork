#!/usr/bin/env bash
# Task C (P5) on v2.0: imbalance-aware type loss under CANONICAL honest config, with per-fold dumps.
# {ce (=baseline, re-run with dump for per-class), ldam} x 3 seeds. logit_adjust skipped on v2.0
# (v3.2 diagnostic showed it collapses the majority). Per-class recall via analyze_perclass_recall.py.
set -u
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-10}"
mkdir -p results/p5 logs
for TL in ce ldam; do
  for SEED in 0 42 1234; do
    log="logs/p5_v2_${TL}_lf_s${SEED}.log"
    dump="results/p5/dump_v2_${TL}_s${SEED}"
    mkdir -p "${dump}"
    echo ">>> [P5-v2] type_loss=${TL} leakage-free seed=${SEED} -> ${log}"
    DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
       --device cpu --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full --leakage_free \
       --type_loss "${TL}" --dump_scores "${dump}" \
       --seed "${SEED}" --epoch "${EP}" --validation "${FOLD}" 2>&1 | tee "${log}"
    venv/bin/python parse_metrics.py "${log}" "results/p5/v2_${TL}_lf_s${SEED}.json" 2>&1 | tail -2
  done
done
echo "=== [P5-v2] imbalance DONE ==="
