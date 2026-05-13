"""Aggregate multi-seed verify of best config (seed=1, K=7) → robustness check.

Goal: verify rằng reproduce gap -1.0% paper là robust qua seeds, không phải lucky seed.

Test: seeds {1 (existing), 42, 7} × K=7 → mean ± std.
1-sample t-test vs paper baseline.
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

SEEDS = [1, 42, 7]


def load_metrics(seed):
    """Seed=1 từ k_sweep, seed=42/7 từ multiseed_best logs."""
    if seed == 1:
        path = RESULTS_DIR / 'k_sweep_K7_seed1.json'
    else:
        path = RESULTS_DIR / f'multiseed_best_seed{seed}.json'

    # If JSON not exists, try to parse log
    if not path.exists():
        if seed == 1:
            log = LOGS_DIR / 'k_sweep_K7_seed1.log'
        else:
            log = LOGS_DIR / f'multiseed_best_seed{seed}.log'
        if not log.exists():
            return None
        m = parse_log(log)
        m['seed'] = seed
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        return m

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '-'


def main():
    results = {}
    for s in SEEDS:
        m = load_metrics(s)
        if m is not None:
            results[s] = m

    keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']

    print('\n' + '=' * 110)
    print('MULTI-SEED VERIFY — best config (K=7) × 3 seeds')
    print('=' * 110)
    print(f'{"Seed":<8} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}')
    print('-' * 110)
    print(f'{"PAPER":<8} {fmt(PAPER["AUC"]):>8} {fmt(PAPER["AUPR"]):>8} {fmt(PAPER["F1"]):>8} '
          f'{fmt(PAPER["top1_precision"]):>8} {fmt(PAPER["top1_recall"]):>8} {fmt(PAPER["top1_f1"]):>8}')
    print('-' * 110)

    for s in SEEDS:
        if s not in results:
            print(f'{s:<8} (missing)')
            continue
        m = results[s]
        vals = [fmt(m.get(k)) for k in keys]
        print(f'{s:<8} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}')

    # Compute stats
    print('\n' + '=' * 110)
    print('STATISTICS — Mean ± Std (n=3)')
    print('=' * 110)
    print(f'{"Metric":<18} {"Mean":>10} {"Std":>10} {"Paper":>10} {"Δ mean-paper":>14} {"t-stat":>10} {"Sig?":>10}')
    print('-' * 110)

    stats = {}
    for k in keys:
        vals = [results[s].get(k) for s in SEEDS if results.get(s, {}).get(k) is not None]
        if len(vals) >= 2:
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            ref = PAPER[k]
            delta = m - ref
            se = sd / np.sqrt(len(vals))
            t = delta / (se + 1e-9)
            sig = 'SIG' if abs(t) > 2.0 else 'marginal' if abs(t) > 1.0 else 'NS'
            stats[k] = {'mean': m, 'std': sd, 'paper': ref, 'delta': delta, 't': t, 'n': len(vals), 'sig': sig}
            print(f'{k:<18} {m:>10.4f} {sd:>10.4f} {ref:>10.4f} {delta:>+10.4f}      {t:>+10.3f} {sig:>10}')

    # Verdict
    print('\n' + '=' * 110)
    print('VERDICT')
    print('=' * 110)
    if 'top1_f1' in stats:
        s = stats['top1_f1']
        m, sd = s['mean'], s['std']
        gap_pct = abs(s['delta']) / PAPER['top1_f1'] * 100
        print(f'Top-1 F1: {m:.4f} ± {sd:.4f}  (paper {PAPER["top1_f1"]:.4f}, gap {gap_pct:.1f}%)')
        if gap_pct < 3.0:
            print(f'✅ REPRODUCE ROBUST: gap < 3% paper across 3 seeds')
        elif gap_pct < 5.0:
            print(f'⚠️ MARGINAL REPRODUCE: gap 3-5%, evidence less strong')
        else:
            print(f'❌ NOT REPRODUCE: gap >= 5%')

    out = {
        'paper': PAPER,
        'seeds': SEEDS,
        'config': {'K_neigs': 7, 'loss_mode': 'two_head', 'exist_weight': 0.3},
        'results': {str(s): results[s] for s in results},
        'stats': stats,
    }
    with open(RESULTS_DIR / 'multiseed_best_summary.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/multiseed_best_summary.json')


if __name__ == '__main__':
    main()
