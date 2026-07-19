#!/usr/bin/env bash
# B4-features: feature miRNA SEQUENCE thật (M_SEQ) thay GIP, dưới {leaked, leakage_free}.
# So với (GIP,leaked)=0.694 và (GIP,leakage_free)=0.6075 đã có (v2.0 K=2 s1234).
set -u
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-12}"
SEQ="v2.0_495m383D/M_SEQ.txt"
BASE="--device cpu --predictor_mode full_bilinear --K_neigs 2 --exist_weight 0.1 --seed 1234 --epoch ${EP} --validation ${FOLD} --mirna_seq_sim_path ${SEQ}"
mkdir -p results/b4 logs

echo ">>> (real-seq, LEAKED)"
DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
   ${BASE} 2>&1 | tee logs/b4_realseq_leaked_v2_K2_s1234.log
venv/bin/python parse_metrics.py logs/b4_realseq_leaked_v2_K2_s1234.log results/b4/realseq_leaked_v2_K2_s1234.json 2>&1 | tail -2

echo ">>> (real-seq, LEAKAGE-FREE)"
DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
   ${BASE} --leakage_free 2>&1 | tee logs/b4_realseq_leakagefree_v2_K2_s1234.log
venv/bin/python parse_metrics.py logs/b4_realseq_leakagefree_v2_K2_s1234.log results/b4/realseq_leakagefree_v2_K2_s1234.json 2>&1 | tail -2

echo "=== B4-features exp DONE ==="
