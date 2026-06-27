#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Overnight chain: v3.2 filtered baseline + 5 Fig.4 ablations.
#
# Goal: controlled experiment - thay v2.0 associations bang v3.2 associations
# (filter to v2.0 entities), KEEP v2.0 similarity. Test Fig.4 pattern voi
# new associations.
#
# Pre-requisite: build_v32_filtered.py da chay - tao folder v3.2_filtered_495m383D/
#
# Estimated: 1 baseline + 5 ablations = 6 runs x ~50min = ~5h CPU.

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

echo "================================================================"
echo " OVERNIGHT v3.2 FILTERED EXPERIMENT"
echo " Config: seed=1, K=7, default loss, dataset=v3.2_filtered_495m383D"
echo " Total: 6 runs x ~50min = ~5h CPU"
echo "================================================================"

# Step 1: Build v3.2 filtered (idempotent - skip if exists)
if [ ! -f "v3.2_filtered_495m383D/multi_all_mirna_disease_pairs_without_negative.csv" ]; then
    echo ""
    echo "[Prep] Build v3.2 filtered dataset..."
    "$PY" build_v32_filtered.py
else
    echo ""
    echo "[Prep] v3.2 filtered dataset exists - skip"
fi

# Step 2: Baseline v3.2 filtered (seed=1, K=7, no ablation)
echo ""
echo "================================================================"
echo " [1/6] BASELINE v3.2 filtered (seed=1, K=7)"
echo "================================================================"
log="logs/v32_baseline.log"
"$PY" main_experiments_hetero1.py --device cpu --seed 1 --K_neigs 7 --dataset v3.2_filtered_495m383D 2>&1 | tee "$log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[WARN] baseline exited nonzero"
fi

# Step 3: 5 ablations v3.2 filtered
ablations=(no_cl no_hgcn no_avf no_hgt no_dv)
idx=1
for abl in "${ablations[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "================================================================"
    echo " [$idx/6] ABLATION $abl (v3.2 filtered, seed=1, K=7)"
    echo "================================================================"
    log="logs/v32_$abl.log"
    "$PY" main_experiments_hetero1.py --device cpu --seed 1 --K_neigs 7 --dataset v3.2_filtered_495m383D --ablation "$abl" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] $abl exited nonzero"
    fi
done

# Step 4: Aggregate
echo ""
echo "================================================================"
echo " AGGREGATE v3.2 FILTERED RESULTS"
echo "================================================================"
"$PY" summarize_v32_filtered.py

# Step 5: Auto regen report + commit + push
echo ""
echo "[Final] Regen report + commit + push..."
"$PY" generate_report.py
"$PY" final_reproduce_report.py

# Git commit + push
git add -A
git commit -m "Overnight v3.2 filtered: baseline + 5 Fig.4 ablations (auto)"
git push origin main

echo ""
echo "DONE overnight v3.2. See results/v32_filtered_summary.json"
