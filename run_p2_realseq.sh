#!/usr/bin/env bash
# Task B (P2): real miRBase k-mer sequence view (M_SEQ) replacing GIP, CANONICAL honest config.
# A/B vs GIP baseline = leakage_gap.json / gap_leakfree_s* (same config, no --mirna_seq_sim_path).
# NOTE: old results/b4/realseq_* used exist_weight 0.1 + legacy CV -> NOT comparable; re-run here.
set -u
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-10}"
SEQ="v2.0_495m383D/M_SEQ.txt"
mkdir -p results/p2 logs
for SEED in 0 42 1234; do
  log="logs/p2_realseq_lf_s${SEED}.log"
  echo ">>> [P2] real-seq leakage-free seed=${SEED} -> ${log}"
  DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
     --device cpu --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full --leakage_free \
     --mirna_seq_sim_path "${SEQ}" \
     --seed "${SEED}" --epoch "${EP}" --validation "${FOLD}" 2>&1 | tee "${log}"
  venv/bin/python parse_metrics.py "${log}" "results/p2/realseq_lf_s${SEED}.json" 2>&1 | tail -2
done
echo "=== [P2] real-seq leakage-free DONE ==="
