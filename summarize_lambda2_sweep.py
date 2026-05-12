"""Aggregate lambda2 sweep — paper Fig.2a check.

Paper Fig.2a: sweep lambda2 (inter_view_weight) for best Top-1 metrics.
We test lambda2 ∈ {0.1, 0.3, 0.5} với (seed=1, K=7, default loss).
lambda2=0.3 đã có từ k_sweep_K7_seed1.json (Top-1 F1=0.5909).
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

PAPER = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}

# lambda2=0.3 baseline from k_sweep
BASELINE_L2_03 = RESULTS_DIR / 'k_sweep_K7_seed1.json'
LAMBDA2_VALUES = [0.1, 0.3, 0.5]


def parse_l2(l2):
    log = LOGS_DIR / f'lambda2_sweep_l2_{l2}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    m['lambda2'] = l2
    out = RESULTS_DIR / f'lambda2_sweep_l2_{l2}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '-'


def delta_pct(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    sign = '+' if d > 0 else ''
    return f'{sign}{d:.1f}%'


def main():
    results = {}

    # Load lambda2=0.3 from k_sweep baseline
    if BASELINE_L2_03.exists():
        with open(BASELINE_L2_03, 'r', encoding='utf-8') as f:
            results[0.3] = json.load(f)
        results[0.3]['lambda2'] = 0.3

    # Parse new lambda2 results
    for l2 in [0.1, 0.5]:
        m = parse_l2(l2)
        if m is not None:
            results[l2] = m

    print('\n' + '=' * 100)
    print('LAMBDA2 SWEEP — paper Fig.2a check (seed=1, K=7, default loss)')
    print('=' * 100)
    print(f'{"λ₂":<6} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}  {"Δ T1-F1 paper":<14}')
    print('-' * 100)
    print(f'{"PAPER":<6} {fmt(PAPER["AUC"]):>8} {fmt(PAPER["AUPR"]):>8} {fmt(PAPER["F1"]):>8} '
          f'{fmt(PAPER["top1_precision"]):>8} {fmt(PAPER["top1_recall"]):>8} {fmt(PAPER["top1_f1"]):>8}  {"---":<14}')
    print('-' * 100)

    best_t1 = -1
    best_l2 = None
    for l2 in LAMBDA2_VALUES:
        if l2 not in results:
            print(f'{l2:<6} (missing)')
            continue
        m = results[l2]
        keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
        vals = [fmt(m.get(k)) for k in keys]
        d = delta_pct(m.get('top1_f1'), PAPER['top1_f1'])
        print(f'{l2:<6} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}  {d:<14}')
        if m.get('top1_f1') and m['top1_f1'] > best_t1:
            best_t1 = m['top1_f1']
            best_l2 = l2

    if best_l2 is not None:
        paper_t1 = PAPER['top1_f1']
        delta = (best_t1 - paper_t1) / paper_t1 * 100
        sign = '+' if delta > 0 else ''
        print(f'\n[BEST] λ₂={best_l2}, Top-1 F1 = {best_t1:.4f} ({sign}{delta:.1f}% vs paper)')
        if abs(delta) <= 3.0:
            print(f'  ✅ Within 3% of paper — REPRODUCE CONFIRMED')
            print(f'  → Next: rerun Fig.4 ablation với λ₂={best_l2}')
        else:
            print(f'  ⚠️ Gap > 3% — λ₂ alone không fix Fig.4')

    summary = {
        'paper': PAPER,
        'config': 'seed=1, K=7, default loss',
        'results': {str(k): v for k, v in results.items()},
        'best_lambda2': best_l2,
        'best_top1_f1': float(best_t1) if best_t1 > 0 else None,
    }
    out = RESULTS_DIR / 'lambda2_sweep_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] {out}')


if __name__ == '__main__':
    main()
