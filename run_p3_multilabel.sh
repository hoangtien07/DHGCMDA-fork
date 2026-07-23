#!/usr/bin/env bash
# Task D (P3): keep multi-label signal. CANONICAL config + dumps for the multi-label metric.
# Arms: {softmax_5class, multilabel_bce} x 3 seeds. two_head baseline = reuse results/p5/dump_v2_ce_s*.
# multilabel_bce needs the multi-hot target from preprocess_v2_multilabel.py.
set -u
if [ "${ALLOW_UNSAFE_P3:-0}" != "1" ]; then
  echo "P3 is BLOCKED: current type outputs use softmax while genuine multi-label BCE requires independent logits/sigmoid channels." >&2
  echo "Do not run this legacy script until a dedicated multi-label logic-fix branch resolves the registry blocker." >&2
  exit 2
fi
cd "$(dirname "$0")"
EP="${EP:-650}"; FOLD="${FOLD:-5}"; TH="${DHGCMDA_N_THREADS:-10}"
ML="v2.0_495m383D/target_multilabel_v2.npy"
mkdir -p results/p3 logs
[ -f "${ML}" ] || { echo "❌ missing ${ML} — run preprocess_v2_multilabel.py first"; exit 1; }

run_one () { # $1=loss_mode $2=seed $3=extra
  local LM="$1" SEED="$2" EXTRA="$3"
  local log="logs/p3_${LM}_lf_s${SEED}.log"
  local dump="results/p3/dump_${LM}_s${SEED}"; mkdir -p "${dump}"
  echo ">>> [P3] loss_mode=${LM} leakage-free seed=${SEED} -> ${log}"
  DHGCMDA_N_THREADS="${TH}" PYTHONUTF8=1 PYTHONUNBUFFERED=1 venv/bin/python main_experiments_hetero1.py \
     --device cpu --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full --leakage_free \
     --loss_mode "${LM}" ${EXTRA} --dump_scores "${dump}" \
     --seed "${SEED}" --epoch "${EP}" --validation "${FOLD}" 2>&1 | tee "${log}"
  venv/bin/python parse_metrics.py "${log}" "results/p3/${LM}_lf_s${SEED}.json" 2>&1 | tail -2
}

for SEED in 0 42 1234; do
  run_one softmax_5class "${SEED}" ""
  run_one multilabel_bce "${SEED}" "--multilabel_target_path ${ML}"
done
echo "=== [P3] multilabel DONE ==="
