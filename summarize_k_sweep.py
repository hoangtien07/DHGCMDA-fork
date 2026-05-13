"""Aggregate K_neigs sweep results — paper Fig.3 reproduce check.

Paper claim (Fig.3): Top-1 metrics max tại K=13. Binary metrics stable across K.

Usage:
    python summarize_k_sweep.py
    python summarize_k_sweep.py --seed 1
"""
import json
import argparse
from pathlib import Path
import numpy as np

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

PAPER = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}

K_VALUES = [3, 5, 7, 9, 11, 13, 15]  # paper Fig.3 tests K=1,3,5,7,9,11,13,15


def parse_k(K, seed):
    log = LOGS_DIR / f'k_sweep_K{K}_seed{seed}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    m['K_neigs'] = K
    m['seed'] = seed
    out = RESULTS_DIR / f'k_sweep_K{K}_seed{seed}.json'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    seed = args.seed

    results = {}
    for K in K_VALUES:
        m = parse_k(K, seed)
        if m is not None:
            results[K] = m

    print('\n' + '=' * 100)
    print(f'K SWEEP (seed={seed}) — Paper Fig.3 reproduce check')
    print(f'Paper claim: Top-1 max at K=13')
    print('=' * 100)
    print(f'{"K":<5} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}  {"Δ T1-F1 paper":<14}')
    print('-' * 100)
    print(f'{"PAPER":<5} {fmt(PAPER["AUC"]):>8} {fmt(PAPER["AUPR"]):>8} {fmt(PAPER["F1"]):>8} '
          f'{fmt(PAPER["top1_precision"]):>8} {fmt(PAPER["top1_recall"]):>8} {fmt(PAPER["top1_f1"]):>8}  {"---":<14}')
    print('-' * 100)

    best_t1_K = None
    best_t1 = -1
    for K in K_VALUES:
        if K not in results:
            print(f'{K:<5} (missing)')
            continue
        m = results[K]
        keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
        vals = [fmt(m.get(k)) for k in keys]
        delta = delta_pct(m.get('top1_f1'), PAPER['top1_f1'])
        print(f'{K:<5} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}  {delta:<14}')
        if m.get('top1_f1') and m['top1_f1'] > best_t1:
            best_t1 = m['top1_f1']
            best_t1_K = K

    if best_t1_K:
        paper_t1 = PAPER['top1_f1']
        delta = (best_t1 - paper_t1) / paper_t1 * 100
        sign = '+' if delta > 0 else ''
        print(f'\n[BEST T1-F1] K={best_t1_K}, Top-1 F1 = {best_t1:.4f} ({sign}{delta:.1f}% vs paper)')
        if best_t1_K == 13:
            print('  ✅ Paper Fig.3 claim CONFIRMED: K=13 optimal cho Top-1')
        else:
            print(f'  ⚠️ Paper Fig.3 NOT exactly reproduced: best là K={best_t1_K}, không phải 13')

    summary = {
        'paper': PAPER,
        'seed': seed,
        'results': {str(k): v for k, v in results.items()},
        'best_K_for_top1': best_t1_K,
        'best_top1_f1': float(best_t1) if best_t1 > 0 else None,
    }
    out = RESULTS_DIR / f'k_sweep_seed{seed}_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] {out}')


if __name__ == '__main__':
    main()
