#!/bin/bash
name=$(echo "$*" | sed 's/--//g; s/ /_/g')
cd ~/dhg_runs/faithful/code && PYTHONUNBUFFERED=1 ~/dhg_runs/venv_np1/bin/python faithful.py $* --threads 1 > ~/dhg_runs/faithful/logs/$name.log 2>&1
echo "done $name $?" >> ~/dhg_runs/faithful/logs/_done.txt
