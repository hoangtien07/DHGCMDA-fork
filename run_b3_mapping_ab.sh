#!/usr/bin/env bash
# B3 A/B: đo tác động của mapping-fix lên recall Tissue (type-5) trên v3.2_wang.
# legacy_type_map (BUG cũ: Tissue->lớp 0) vs fixed (k->k-1). Dump per-fold → per-class recall.
# Cùng mọi cấu hình khác → cô lập đúng tác động mapping.
set -u
cd "$(dirname "$0")"
EP="${EP:-200}"; FOLD="${FOLD:-3}"; TH="${DHGCMDA_N_THREADS:-12}"
COMMON="--device cpu --dataset v3.2_wang --predictor_mode full_bilinear --exist_weight 0.1 --loss_mode two_head --type_loss ce --epoch ${EP} --validation ${FOLD}"
mkdir -p results/b3 results/b3/dump_legacy results/b3/dump_fixed logs

echo ">>> A) legacy mapping (BUG cũ)"
DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
   ${COMMON} --legacy_type_map --dump_scores results/b3/dump_legacy/ 2>&1 | tee logs/b3_map_legacy_ep${EP}f${FOLD}.log

echo ">>> B) fixed mapping (k->k-1)"
DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
   ${COMMON} --dump_scores results/b3/dump_fixed/ 2>&1 | tee logs/b3_map_fixed_ep${EP}f${FOLD}.log

echo "=== per-class recall ==="
venv/bin/python analyze_perclass_recall.py --dump_dir results/b3/dump_legacy/ --tag "LEGACY(bug)" | tee logs/b3_perclass_legacy.log
venv/bin/python analyze_perclass_recall.py --dump_dir results/b3/dump_fixed/  --tag "FIXED"      | tee logs/b3_perclass_fixed.log
echo "=== B3 mapping A/B DONE ==="
