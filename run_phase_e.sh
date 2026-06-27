#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Plan E - True ablation rebuild full pipeline (3 modes x 5 fold).
#
# Verify hypothesis: Fig.4 ablation pattern bi dao co phai do additive switch
# implementation hay khong. Plan E rebuild 3 ablation dao (CL, HGCN, HGT) voi
# kien truc rut gon thuc su (HGCN plain, GCNConv, skip transformers).
#
# Cach dung:
#     ./run_phase_e.sh                          # full ~3.5h: 3 ablations sequential
#     ./run_phase_e.sh --only-mode no_cl_rebuild  # chi 1 mode (~50')

OnlyMode=""
SKIP_SUMMARIZE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only-mode) OnlyMode="$2"; shift 2 ;;
        --skip-summarize) SKIP_SUMMARIZE=1; shift ;;
        *) shift ;;
    esac
done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

# Modes to run
if [ -n "$OnlyMode" ]; then
    modes=("$OnlyMode")
else
    modes=(no_cl_rebuild no_hgcn_rebuild no_hgt_rebuild)
fi

echo "================================================================"
echo " PLAN E - TRUE ABLATION REBUILD (loss_mode=softmax_5class)"
echo " Modes: ${modes[*]}"
echo " ETA: ~50 phut/mode * ${#modes[@]} modes"
echo "================================================================"

for mode in "${modes[@]}"; do
    echo ""
    echo "[Plan E] $mode (softmax_5class) ~50 phut"
    echo "----------------------------------------------------------------"
    "$PY" main_experiments_hetero1.py \
        --device cpu --loss_mode softmax_5class --ablation "$mode" \
        2>&1 | tee "logs/phase_e_$mode.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] $mode exited nonzero"
    fi
done

if [ "$SKIP_SUMMARIZE" -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo " AGGREGATE PHASE E"
    echo "================================================================"
    "$PY" summarize_phase_e.py
    "$PY" generate_report.py
    echo ""
    echo "DONE Plan E. Xem BaoCao_DHGCMDA.docx + results/phase_e_summary.json"
fi
