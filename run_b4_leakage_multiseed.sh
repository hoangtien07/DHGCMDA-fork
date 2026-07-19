#!/usr/bin/env bash
# B4-leakage robustness: leakage-free v2.0 K=2 full_bilinear qua seed {0,42} (bổ sung s1234 đã có).
# So từng seed với baseline leaked tương ứng để xác nhận mức tụt ~-12% robust.
set -u
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-12}"
mkdir -p results/b4 logs
for SEED in 0 42; do
  log="logs/b4_leakagefree_v2_K2_s${SEED}.log"
  echo ">>> leakage-free seed=${SEED} -> ${log}"
  DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
     --device cpu --predictor_mode full_bilinear --K_neigs 2 --exist_weight 0.1 --seed "${SEED}" \
     --leakage_free --epoch "${EP}" --validation "${FOLD}" 2>&1 | tee "${log}"
  venv/bin/python parse_metrics.py "${log}" "results/b4/leakagefree_v2_K2_s${SEED}.json" 2>&1 | tail -2
done
echo "=== B4 leakage multiseed DONE ==="
