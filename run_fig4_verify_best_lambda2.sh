#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Auto-chain script: rerun Fig.4 ablation voi best lambda2 from lambda2 sweep.
#
# Logic:
#   1. Read results/lambda2_sweep_summary.json
#   2. Identify best_lambda2 (closest to paper Top-1 F1=0.5970)
#   3. If best_lambda2 == 0.3, skip (da co tu fig4_verify_summary.json)
#   4. Else: rerun 5 ablation voi (seed=1, K=7, inter_view_weight=best_lambda2)
#
# Estimated: 0-5 ablation x ~50min = 0-4h CPU (depends on best lambda2).

export PYTHONUTF8=1
PY="venv/bin/python"

# Read best lambda2 from summary JSON
summary_path="results/lambda2_sweep_summary.json"
if [ ! -f "$summary_path" ]; then
    echo "[ERROR] $summary_path not found. Lambda2 sweep chua xong?"
    exit 1
fi

best_l2=$("$PY" -c "import json; print(json.load(open('$summary_path'))['best_lambda2'])")
best_t1=$("$PY" -c "import json; print(json.load(open('$summary_path'))['best_top1_f1'])")

echo "================================================================"
echo " AUTO-CHAIN: Fig.4 verify with BEST lambda2"
echo " Best lambda2 from sweep: $best_l2 (T1-F1 = $best_t1)"
echo " Paper T1-F1: 0.5970"
echo "================================================================"

is_default=$("$PY" -c "print('1' if float('$best_l2') == 0.3 else '0')")
if [ "$is_default" = "1" ]; then
    echo ""
    echo "[SKIP] Best lambda2 = 0.3 (default) — Fig.4 da co tu fig4_verify_summary.json"
    echo "       Khong can rerun. Verdict: lambda2 KHONG fix Fig.4."

    # Auto-trigger final summary
    "$PY" final_reproduce_report.py
    exit 0
fi

# Rerun Fig.4 ablation with best lambda2
seed=1
K=7
ablations=(no_cl no_hgcn no_avf no_hgt no_dv)

echo ""
echo "Running 5 ablations with lambda2 = $best_l2..."

idx=0
for abl in "${ablations[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "[$idx/${#ablations[@]}] ablation=$abl (lambda2=$best_l2)"
    echo "----------------------------------------------------------------"
    log="logs/fig4_l2_${best_l2}_${abl}.log"
    "$PY" main_experiments_hetero1.py \
        --device cpu --seed $seed --K_neigs $K --inter_view_weight "$best_l2" --ablation "$abl" \
        2>&1 | tee "$log"
done

# Aggregate Fig.4 with new lambda2
echo ""
echo "================================================================"
echo " FIG.4 VERIFY (lambda2=$best_l2) SUMMARY"
echo "================================================================"
"$PY" summarize_fig4_best_lambda2.py --lambda2 "$best_l2"

# Final report
echo ""
"$PY" final_reproduce_report.py

echo ""
echo "DONE chain. Xem results/final_reproduce_report.json"
