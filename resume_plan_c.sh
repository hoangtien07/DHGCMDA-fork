#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Resume Plan C — verify w=0.1 fix Fig.4 + case study collapse.
#
# Cach dung:
#     ./resume_plan_c.sh                  # full sequential ~2.7h
#     ./resume_plan_c.sh --skip-casestudy # chi 5 ablation (~2.5h)
#     ./resume_plan_c.sh --only-casestudy # chi case study + rerank (~9')

SKIP_CASESTUDY=0
ONLY_CASESTUDY=0
for a in "$@"; do
    [ "$a" = "--skip-casestudy" ] && SKIP_CASESTUDY=1
    [ "$a" = "--only-casestudy" ] && ONLY_CASESTUDY=1
done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

# Backup snapshot Phase B-C neu chua co (idempotent)
if [ ! -d "results/snapshot_phaseBC_w0.3" ]; then
    mkdir -p "results/snapshot_phaseBC_w0.3"
    for f in case_study_score.npy case_study_breast.csv case_study_hcc.csv case_study_summary.json; do
        if [ -f "results/$f" ]; then
            cp "results/$f" "results/snapshot_phaseBC_w0.3/$f"
        fi
    done
    echo "[backup] Snapshot Phase B-C -> results/snapshot_phaseBC_w0.3/"
fi

# Step 1: Case study voi w=0.1
if [ "$SKIP_CASESTUDY" -eq 0 ]; then
    if [ -f "results/case_study_score.npy" ]; then
        rm -f "results/case_study_score.npy"
        echo "[clean] Deleted cached score.npy (se retrain voi w=0.1)"
    fi
    echo "================================================================"
    echo " 1/3 CASE STUDY (w=0.1) ~9 phut"
    echo "================================================================"
    "$PY" case_study.py --exist_weight 0.1 2>&1 | tee "logs/case_study_w0.1.log"

    echo "================================================================"
    echo " 2/3 RERANK (4 chien luoc, ~5 giay)"
    echo "================================================================"
    "$PY" rerank_case_study.py 2>&1 | tee "logs/rerank_w0.1.log"
fi

# Step 2: 5 ablation voi w=0.1
if [ "$ONLY_CASESTUDY" -eq 0 ]; then
    echo "================================================================"
    echo " 3/3 ABLATION (w=0.1) - 5 variants x ~30 phut = ~2.5h"
    echo "================================================================"
    for mode in no_cl no_hgcn no_avf no_hgt no_dv; do
        echo ""
        echo "[ABL] $mode (w=0.1)"
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --ablation "$mode" 2>&1 | tee "logs/abl_w0.1_$mode.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[WARN] ablation $mode exited nonzero"
        fi
    done
fi

# Step 3: aggregate
echo ""
echo "================================================================"
echo " AGGREGATE + REGEN BAO CAO"
echo "================================================================"
"$PY" summarize_plan_c_full.py
"$PY" generate_report.py
echo ""
echo "DONE. Xem BaoCao_DHGCMDA.docx + results/plan_c_full_summary.json"
