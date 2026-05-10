"""Aggregate seed sweep results, identify seed closest to paper baseline.

Goal: tìm seed cho metrics gần paper nhất → fix seed đó cho hyperparam sweep.

Paper baseline (HMDD v2.0):
  AUC = 0.9669, AUPR = 0.9738, F1 = 0.9278
  Top-1 P = 0.5842, Top-1 R = 0.6341, Top-1 F1 = 0.5970
"""
import json
from pathlib import Path
import numpy as np

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

PAPER = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}

SEEDS = [0, 1, 42, 1234]


def parse_seed(seed):
    log = LOGS_DIR / f'seed_sweep_seed{seed}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    m['seed'] = seed
    out = RESULTS_DIR / f'seed_sweep_seed{seed}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '-'


def distance_to_paper(metrics):
    """L2 distance trên 6 metrics chính, normalize bằng paper value."""
    keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
    d = 0.0
    n = 0
    for k in keys:
        if metrics.get(k) is not None:
            ref = PAPER[k]
            d += ((metrics[k] - ref) / ref) ** 2
            n += 1
    return np.sqrt(d / n) if n > 0 else float('inf')


def main():
    results = {}
    for seed in SEEDS:
        m = parse_seed(seed)
        if m is not None:
            results[seed] = m

    print('\n' + '=' * 100)
    print('SEED SWEEP — find seed closest to paper baseline')
    print('=' * 100)
    print(f'{"Seed":<8} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}  {"Dist":>8}')
    print('-' * 100)
    print(f'{"PAPER":<8} {fmt(PAPER["AUC"]):>8} {fmt(PAPER["AUPR"]):>8} {fmt(PAPER["F1"]):>8} '
          f'{fmt(PAPER["top1_precision"]):>8} {fmt(PAPER["top1_recall"]):>8} {fmt(PAPER["top1_f1"]):>8}  {"---":>8}')
    print('-' * 100)

    distances = {}
    for seed in SEEDS:
        if seed not in results:
            print(f'{seed:<8} (missing)')
            continue
        m = results[seed]
        d = distance_to_paper(m)
        distances[seed] = d
        keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
        vals = [fmt(m.get(k)) for k in keys]
        print(f'{seed:<8} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}  {d:.4f}')

    if distances:
        best_seed = min(distances, key=distances.get)
        best_dist = distances[best_seed]
        print(f'\n[BEST] seed={best_seed}, normalized L2 distance to paper = {best_dist:.4f}')
        print(f'\n→ Use --seed {best_seed} cho hyperparam sweep tiếp theo.')

    summary = {
        'paper': PAPER,
        'results': {str(k): v for k, v in results.items()},
        'distances_to_paper': {str(k): float(v) for k, v in distances.items()},
        'best_seed': best_seed if distances else None,
    }
    out = RESULTS_DIR / 'seed_sweep_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] {out}')


if __name__ == '__main__':
    main()
