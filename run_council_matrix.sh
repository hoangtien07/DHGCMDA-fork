#!/usr/bin/env bash
# Council experiment-matrix orchestrator (Plan L, Linux).
# Reads results/council_matrix.json -> runs each experiment, max 4 in parallel,
# 8 threads/job (32-core box). Each run: tee logs/council_<id>.log -> parse_metrics -> results/council_<id>.json
#
# Usage:
#   ./run_council_matrix.sh                 # run all experiments in the matrix
#   ./run_council_matrix.sh --matrix FILE   # custom matrix json
#   ./run_council_matrix.sh --lanes N       # concurrency (default 4)
#   ./run_council_matrix.sh --dry-run       # print the planned commands only
set -uo pipefail
cd "$(dirname "$0")"

PY="venv/bin/python"
MATRIX="results/council_matrix.json"
LANES=4
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --matrix) MATRIX="$2"; shift 2;;
    --lanes)  LANES="$2";  shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "[WARN] unknown arg: $1"; shift;;
  esac
done

mkdir -p logs results
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 DHGCMDA_N_THREADS=8

if [ ! -f "$MATRIX" ]; then echo "[ERR] matrix not found: $MATRIX"; exit 1; fi

# Emit "id<TAB>entrypoint<TAB>extra_flags" per experiment (priority order)
JOBSPEC="$(mktemp)"
"$PY" - "$MATRIX" > "$JOBSPEC" <<'PYEOF'
import sys, json
m = json.load(open(sys.argv[1]))
exps = m.get("matrix", m).get("experiments", []) if isinstance(m.get("matrix", m), dict) else []
if not exps and "experiments" in m: exps = m["experiments"]
exps = sorted(exps, key=lambda e: e.get("priority", 999))
for e in exps:
    print("\t".join([e["id"], e.get("entrypoint","main"), e.get("extra_flags","").strip()]))
PYEOF

NJOBS=$(wc -l < "$JOBSPEC")
echo "================================================================"
echo " COUNCIL MATRIX: $NJOBS experiments, $LANES lanes x 8 threads"
echo "================================================================"

run_one() {
  local id="$1" entry="$2" flags="$3"
  local log="logs/council_${id}.log" json="results/council_${id}.json"
  local script
  if [ "$entry" = "v32_correct" ]; then script="run_v32_correct_metric.py"; else script="main_experiments_hetero1.py"; fi
  echo "[START] $id ($script) flags: $flags"
  # shellcheck disable=SC2086
  "$PY" "$script" --device cpu $flags > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[WARN] $id exited rc=$rc (see $log)"; fi
  "$PY" parse_metrics.py "$log" "$json" >/dev/null 2>&1 && echo "[DONE] $id -> $json" || echo "[WARN] parse failed: $id"
}

if [ "$DRY" -eq 1 ]; then
  while IFS=$'\t' read -r id entry flags; do
    [ "$entry" = "v32_correct" ] && s="run_v32_correct_metric.py" || s="main_experiments_hetero1.py"
    echo "$id: $PY $s --device cpu $flags"
  done < "$JOBSPEC"
  rm -f "$JOBSPEC"; exit 0
fi

start=$SECONDS
running=0
while IFS=$'\t' read -r id entry flags; do
  run_one "$id" "$entry" "$flags" &
  running=$((running+1))
  if [ "$running" -ge "$LANES" ]; then wait -n; running=$((running-1)); fi
done < "$JOBSPEC"
wait
rm -f "$JOBSPEC"
echo "================================================================"
echo " MATRIX COMPLETE in $((SECONDS-start))s"
echo "================================================================"
ls -1 results/council_*.json 2>/dev/null | sed 's/^/  /'
