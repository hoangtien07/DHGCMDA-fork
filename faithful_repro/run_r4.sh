#!/bin/bash
# round 4: same model, eval additions (micro recall, 10 negative draws) + 4-view cuilab dataset
d=$1; p=$2; s=$3; name=${d}_${p}_none_s${s}
cd ~/dhg_runs/faithful/code && PYTHONUNBUFFERED=1 ~/dhg_runs/venv_np1/bin/python faithful.py --data $d --protocol $p --seed $s --threads 1 --out ~/dhg_runs/faithful/results_r4/$name.json > ~/dhg_runs/faithful/logs_r4/$name.log 2>&1
echo "done $name $?" >> ~/dhg_runs/faithful/logs_r4/_done.txt
