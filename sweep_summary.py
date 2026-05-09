"""Aggregate Plan C sweep results: parse logs/sweep_w*.log → JSON metrics
mỗi phase, build comparison table với paper + Phase A + Phase B-C.

Cách dùng:
    python sweep_summary.py

Output:
    results/sweep_w0.1.json, results/sweep_w0.05.json, results/sweep_w0.0.json
    results/plan_c_comparison.json — bảng tổng hợp 4 phase + paper
    Stdout: bảng comparison đẹp.
"""
import json
import os
import sys
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

# Reference values
PAPER = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}

PHASE_A = {  # code gốc, exist_weight=0.3, từ commit f2b2b15 era
    'AUC': 0.9738, 'AUPR': 0.9671, 'F1': 0.9295,
    'top1_precision': 0.5075, 'top1_recall': 0.5979, 'top1_f1': 0.5485,
    '_source': 'historical (Phase A — code gốc)',
}

# Phase B-C metrics đã có sẵn ở results/baseline_v2.0_metrics.json (exist_weight=0.3)

SWEEPS = [
    ('w0.1', 'sweep_w0.1', 0.1),
    ('w0.05', 'sweep_w0.05', 0.05),
    ('w0.0', 'sweep_w0.0', 0.0),
]


def parse_sweep(label, log_stem, exist_weight):
    log_path = LOGS_DIR / f'{log_stem}.log'
    out_path = RESULTS_DIR / f'{log_stem}.json'

    if not log_path.exists():
        return None
    metrics = parse_log(log_path)
    metrics['_source'] = str(log_path)
    metrics['exist_weight'] = exist_weight
    metrics['type_weight'] = round(1.0 - exist_weight, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'[parse] {log_path} -> {out_path}')
    return metrics


def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else '-'


def delta(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    sign = '+' if d > 0 else ''
    return f'{sign}{d:.1f}%'


def main():
    # Parse 3 sweep logs
    results = {}
    for label, log_stem, w in SWEEPS:
        m = parse_sweep(label, log_stem, w)
        if m is not None:
            results[label] = m

    # Load Phase B-C reference (đã có)
    bc_path = RESULTS_DIR / 'baseline_v2.0_metrics.json'
    if bc_path.exists():
        with open(bc_path, 'r', encoding='utf-8') as f:
            phase_bc = json.load(f)
    else:
        phase_bc = {}

    # Build comparison
    rows = [
        ('Paper baseline',     'paper', PAPER),
        ('Phase A (w=0.3 orig)', 'A', PHASE_A),
        ('Phase B-C (w=0.3 fix)', 'B-C', phase_bc),
    ]
    for label, log_stem, w in SWEEPS:
        if label in results:
            rows.append((f'Phase C-{label} (w={w})', log_stem, results[label]))
        else:
            rows.append((f'Phase C-{label} (w={w}) [pending]', log_stem, {}))

    # Print bảng
    keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
    print('\n' + '=' * 110)
    print(f'{"Run":<32} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}  {"Δ Top-1 F1 vs paper":<22}')
    print('=' * 110)
    for label, _, data in rows:
        vals = [fmt(data.get(k)) for k in keys]
        delta_str = delta(data.get('top1_f1'), PAPER['top1_f1'])
        print(f'{label:<32} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}  {delta_str:<22}')
    print('=' * 110)

    # Save full
    summary = {
        'paper': PAPER,
        'phase_A_orig': PHASE_A,
        'phase_B_C_fix3': phase_bc,
    }
    for label, _, data in rows:
        if label.startswith('Phase C-'):
            summary[label.split(' ')[1]] = data
    with open(RESULTS_DIR / 'plan_c_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/plan_c_comparison.json')

    # Recommend best
    completed = [(label, data) for label, _, data in rows
                 if label.startswith('Phase C-') and data.get('top1_f1') is not None]
    if completed:
        best = max(completed, key=lambda x: x[1].get('top1_f1', 0))
        print(f'\n[best] {best[0]} → Top-1 F1 = {best[1]["top1_f1"]:.4f}')
        if best[1].get('AUC', 1.0) < 0.90:
            print(f'  ⚠️  AUC binary {best[1]["AUC"]:.4f} < 0.90 — too low, không nên dùng')
        else:
            print(f'   AUC binary {best[1]["AUC"]:.4f} ≥ 0.90 — OK')


if __name__ == '__main__':
    main()
