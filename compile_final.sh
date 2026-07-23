#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "--allow-legacy-report" ]; then
    echo "Refusing to regenerate the legacy report without --allow-legacy-report." >&2
    echo "See docs/status/PROJECT_STATUS.md for the leakage-controlled report policy." >&2
    exit 2
fi

# Compile final results sau khi tat ca ablation xong:
#   1. Re-parse tat ca ablation logs (handle UTF-16 tu Tee-Object)
#   2. Tong hop ablation_results.json dung format
#   3. Re-generate bao cao .docx voi data day du

export PYTHONUTF8=1
PY="venv/bin/python"

modes=(no_cl no_hgcn no_avf no_hgt no_dv)

echo "=== Step 1: Re-parse all ablation logs ==="
for mode in "${modes[@]}"; do
    log="logs/ablation_$mode.log"
    json="results/ablation_$mode.json"
    if [ -f "$log" ]; then
        "$PY" parse_metrics.py "$log" "$json" >/dev/null
        echo "  Parsed: $mode"
    else
        echo "  MISSING: $log"
    fi
done

echo ""
echo "=== Step 2: Compile ablation_results.json ==="
"$PY" - "${modes[@]}" <<'PYEOF'
import sys, json, os
modes = sys.argv[1:]
summary = {}
for m in modes:
    p = f"results/ablation_{m}.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            summary[m] = json.load(fh)
with open("results/ablation_results.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
print("  Saved: results/ablation_results.json")
PYEOF

echo ""
echo "=== Step 3: Re-generate report ==="
"$PY" generate_report.py

echo ""
echo "=== Step 4: Print summary table ==="
"$PY" - "${modes[@]}" <<'PYEOF'
import sys, json, os
modes = sys.argv[1:]
with open("results/baseline_v2.0_metrics.json", "r", encoding="utf-8") as fh:
    baseline = json.load(fh)
print("{0:<12} {1:<8} {2:<8} {3:<12} {4:<12} {5:<12}".format(
    'Variant', 'AUC', 'AUPR', 'Top1-Prec', 'Top1-Rec', 'Top1-F1'))
print("-" * 70)
def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
print("{0:<12} {1:<8} {2:<8} {3:<12} {4:<12} {5:<12}".format(
    'Full', fmt(baseline.get('AUC')), fmt(baseline.get('AUPR')),
    fmt(baseline.get('top1_precision')), fmt(baseline.get('top1_recall')),
    fmt(baseline.get('top1_f1'))))
for mode in modes:
    p = f"results/ablation_{mode}.json"
    if not os.path.exists(p):
        continue
    with open(p, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    if m and m.get('AUC'):
        name = "w/o " + mode[3:].upper()
        print("{0:<12} {1:<8} {2:<8} {3:<12} {4:<12} {5:<12}".format(
            name, fmt(m.get('AUC')), fmt(m.get('AUPR')),
            fmt(m.get('top1_precision')), fmt(m.get('top1_recall')),
            fmt(m.get('top1_f1'))))
PYEOF
