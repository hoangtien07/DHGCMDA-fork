#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Multi-seed full - 3 seeds x 3 baseline variants for MLRC paper.
#
# Bug fixed (2026-05-11): seed_torch + prepareData seed propagation.
# All previous Plan A/B/C/D/E results used seed=0 data split (bugged).
# Multi-seed v3 reruns 3 seeds fresh (1234, 42, 7).
#
# Usage:
#     ./run_multiseed_full.sh                       # full ~7.5h: 9 runs
#     ./run_multiseed_full.sh --only-variant plan_d # 3 runs ~2.5h

OnlyVariant=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only-variant) OnlyVariant="$2"; shift 2 ;;
        *) shift ;;
    esac
done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

# Variants: name -> args
variant_names=(plan_c_w01 plan_d plan_e)
variant_args_plan_c_w01=(--exist_weight 0.1)
variant_args_plan_d=(--loss_mode softmax_5class)
variant_args_plan_e=(--loss_mode softmax_5class --ablation no_cl_rebuild)

# 3 seeds fresh after bug fix
seeds=(1234 42 7)

# Filter variants
if [ -n "$OnlyVariant" ]; then
    case "$OnlyVariant" in
        plan_c_w01|plan_d|plan_e) selected_variants=("$OnlyVariant") ;;
        *) echo "[ERROR] Unknown variant: $OnlyVariant"; exit 1 ;;
    esac
else
    selected_variants=("${variant_names[@]}")
fi

total_runs=$(( ${#seeds[@]} * ${#selected_variants[@]} ))
echo "================================================================"
echo " MULTI-SEED FULL"
echo " Total runs: $total_runs (50 min each)"
echo " Seeds: ${seeds[*]}"
echo " Variants: ${selected_variants[*]}"
echo "================================================================"

get_args() {
    case "$1" in
        plan_c_w01) printf '%s\n' "${variant_args_plan_c_w01[@]}" ;;
        plan_d)     printf '%s\n' "${variant_args_plan_d[@]}" ;;
        plan_e)     printf '%s\n' "${variant_args_plan_e[@]}" ;;
    esac
}

run_idx=0
for seed in "${seeds[@]}"; do
    for vname in "${selected_variants[@]}"; do
        run_idx=$((run_idx+1))
        mapfile -t args_list < <(get_args "$vname")
        echo ""
        echo "[$run_idx/$total_runs] $vname seed=$seed"
        echo "----------------------------------------------------------------"

        log_path="logs/multiseed_${vname}_seed${seed}.log"
        "$PY" main_experiments_hetero1.py --device cpu --seed "$seed" "${args_list[@]}" 2>&1 | tee "$log_path"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[WARN] $vname seed=$seed exited nonzero"
        fi
    done
done

# Aggregate
echo ""
echo "================================================================"
echo " AGGREGATE MULTI-SEED FULL"
echo "================================================================"
"$PY" summarize_multiseed.py
echo ""
echo "DONE Multi-seed. See results/multiseed_full_summary.json"
