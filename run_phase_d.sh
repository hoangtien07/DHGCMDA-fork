#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Phase D — Fix A++ (5-class softmax CE) full pipeline.
#
# Verify Eq. 32 alignment hoan toan (single L_type, no L_existence).
# Compare voi Plan C-w0.1 (two_head) de xem fix Fig.4 + case study collapse hay khong.
#
# Cach dung:
#     ./run_phase_d.sh                   # full ~5-6h: baseline + 5 ablation + case study
#     ./run_phase_d.sh --skip-casestudy
#     ./run_phase_d.sh --only-baseline   # chi baseline (~50') de smoke kiem tra Top-1 F1

SKIP_CASESTUDY=0
ONLY_BASELINE=0
for a in "$@"; do
    [ "$a" = "--skip-casestudy" ] && SKIP_CASESTUDY=1
    [ "$a" = "--only-baseline" ] && ONLY_BASELINE=1
done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

# Backup Plan C w=0.1 case study results truoc khi rerun (idempotent)
if [ ! -d "results/snapshot_planC_w0.1" ]; then
    mkdir -p "results/snapshot_planC_w0.1"
    for f in case_study_score.npy case_study_breast.csv case_study_hcc.csv case_study_summary.json rerank_summary.json; do
        if [ -f "results/$f" ]; then
            cp "results/$f" "results/snapshot_planC_w0.1/$f"
        fi
    done
    echo "[backup] Snapshot Plan C-w0.1 -> results/snapshot_planC_w0.1/"
fi

# Step 1: Phase D baseline (~50')
echo "================================================================"
echo " 1/3 PHASE D BASELINE (loss_mode=softmax_5class) ~50 phut"
echo "================================================================"
"$PY" main_experiments_hetero1.py --device cpu --loss_mode softmax_5class 2>&1 | tee "logs/phase_d_baseline.log"

if [ "$ONLY_BASELINE" -eq 1 ]; then
    echo ""
    echo "DONE baseline only. Run summarize_phase_d.py to compare."
    exit 0
fi

# Step 2: Case study with softmax_5class (~9')
if [ "$SKIP_CASESTUDY" -eq 0 ]; then
    if [ -f "results/case_study_score.npy" ]; then
        rm -f "results/case_study_score.npy"
        echo "[clean] Deleted cached score.npy (se retrain voi softmax_5class)"
    fi
    echo "================================================================"
    echo " 2/3 CASE STUDY (softmax_5class) ~9 phut"
    echo "================================================================"
    "$PY" case_study.py --loss_mode softmax_5class 2>&1 | tee "logs/phase_d_casestudy.log"

    "$PY" rerank_case_study.py 2>&1 | tee "logs/phase_d_rerank.log"
fi

# Step 3: 5 ablation voi softmax_5class
echo "================================================================"
echo " 3/3 ABLATION (softmax_5class) - 5 variants x ~30 phut = ~2.5h"
echo "================================================================"
for mode in no_cl no_hgcn no_avf no_hgt no_dv; do
    echo ""
    echo "[ABL] $mode (softmax_5class)"
    "$PY" main_experiments_hetero1.py --device cpu --loss_mode softmax_5class --ablation "$mode" 2>&1 | tee "logs/phase_d_$mode.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] ablation $mode exited nonzero"
    fi
done

# Step 4: aggregate
echo ""
echo "================================================================"
echo " AGGREGATE + REGEN BAO CAO"
echo "================================================================"
"$PY" summarize_phase_d.py
"$PY" generate_report.py
echo ""
echo "DONE Phase D. Xem BaoCao_DHGCMDA.docx + results/phase_d_summary.json"
