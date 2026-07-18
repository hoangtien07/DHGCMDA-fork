#!/usr/bin/env bash
# Đo leakage gap (P7/P9, docs/review/03): giữ MỌI thứ cố định, chỉ toggle --leakage_free.
# full_bilinear + K=2 + cv_scheme full (khớp default v2.0 mới), 3 seed × {leaky, leakage_free}.
# Tuần tự (mỗi run đã đa luồng). ETA nhiều giờ CPU.
set -u
cd "$(dirname "$0")"
export DHGCMDA_N_THREADS="${DHGCMDA_N_THREADS:-28}"
PY=./venv/bin/python
SEEDS=(0 42 1234)
COMMON="--device cpu --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full"
mkdir -p logs results

echo "[$(date '+%F %T')] START leakage-gap: seeds=${SEEDS[*]} threads=$DHGCMDA_N_THREADS"

for S in "${SEEDS[@]}"; do
  echo "[$(date '+%F %T')] === seed $S : LEAKY ==="
  $PY main_experiments_hetero1.py $COMMON --seed "$S" \
      > "logs/gap_leaky_s$S.log" 2>&1
  echo "[$(date '+%F %T')]     leaky s$S exit=$?"

  echo "[$(date '+%F %T')] === seed $S : LEAKAGE-FREE ==="
  $PY main_experiments_hetero1.py $COMMON --seed "$S" --leakage_free \
      > "logs/gap_leakfree_s$S.log" 2>&1
  echo "[$(date '+%F %T')]     leakfree s$S exit=$?"
done

echo "[$(date '+%F %T')] === Aggregate stats ==="
$PY summarize_stats.py \
  --logs logs/gap_leaky_s0.log logs/gap_leaky_s42.log logs/gap_leaky_s1234.log \
         logs/gap_leakfree_s0.log logs/gap_leakfree_s42.log logs/gap_leakfree_s1234.log \
  --labels leaky leaky leaky leakage_free leakage_free leakage_free \
  --out results/leakage_gap.json

echo "[$(date '+%F %T')] DONE. Xem results/leakage_gap.json"
