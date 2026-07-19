#!/usr/bin/env bash
# B3 (branch breakthrough-imbalance-features): imbalance-aware type loss trên v3.2_wang.
# Đo bằng run_v32_correct_metric.py (monkey-patch → bypass metric-bug 4-type, in per-type suite).
# So sánh: ce (mapping-fix) vs logit_adjust vs ldam. Nhắm phục hồi minority type (Tissue=T5).
# Stage-1 = diagnostic nhanh (EP/FOLD nhỏ). Đổi EP/FOLD qua env để chạy full.
set -u
cd "$(dirname "$0")"

EP="${EP:-120}"
FOLD="${FOLD:-3}"
THREADS="${DHGCMDA_N_THREADS:-16}"
COMMON="--device cpu --dataset v3.2_wang --predictor_mode full_bilinear --exist_weight 0.1 --loss_mode two_head --epoch ${EP} --validation ${FOLD}"

mkdir -p logs results/b3

run_variant () {
  local name="$1"; shift
  local log="logs/b3_v32_${name}_ep${EP}f${FOLD}.log"
  echo "======================================================================"
  echo ">>> B3 variant=${name}  (EP=${EP} FOLD=${FOLD})  -> ${log}"
  echo "======================================================================"
  DHGCMDA_N_THREADS="${THREADS}" PYTHONUTF8=1 venv/bin/python run_v32_correct_metric.py \
      ${COMMON} "$@" 2>&1 | tee "${log}"
}

run_variant "ce"          --type_loss ce
run_variant "logitadjust" --type_loss logit_adjust --la_tau 1.0
run_variant "ldam"        --type_loss ldam --ldam_max_margin 0.5

echo "=== B3 stage-1 DONE. Per-type suite ở cuối mỗi log logs/b3_v32_*_ep${EP}f${FOLD}.log ==="
