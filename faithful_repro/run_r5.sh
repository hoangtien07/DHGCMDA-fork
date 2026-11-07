#!/bin/bash
# round 5: usage run_r5.sh NAME SCRIPT ARGS...  (jobs from jobs_r5.txt, pre-registered in report/PREREG_R5.md)
name=$1; shift; mkdir -p ~/dhg_runs/faithful/results_r5 ~/dhg_runs/faithful/logs_r5
cd ~/dhg_runs/faithful/code && PYTHONUNBUFFERED=1 ~/dhg_runs/venv_np1/bin/python "$@" --out ~/dhg_runs/faithful/results_r5/$name.json > ~/dhg_runs/faithful/logs_r5/$name.log 2>&1
echo "done $name $?" >> ~/dhg_runs/faithful/logs_r5/_done.txt
