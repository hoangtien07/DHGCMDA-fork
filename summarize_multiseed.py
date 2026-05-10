"""Aggregate multi-seed results: parse logs/multiseed_*.log + existing single-seed
JSONs, compute mean ± std cho 3 baseline variants × 3 seeds.

Pre-existing single-seed (seed=1234):
- Plan C-w0.1: results/sweep_w0.1.json
- Plan D baseline: results/phase_d_baseline.json
- Plan E baseline (no_cl_rebuild): results/phase_e_no_cl_rebuild.json

New seeds (42, 7):
- logs/multiseed_{plan_c_w01,plan_d,plan_e}_seed{42,7}.log

Output: results/multiseed_full_summary.json + bảng mean ± std stdout.
"""
import json
from pathlib import Path
import numpy as np

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

VARIANTS = ['plan_c_w01', 'plan_d', 'plan_e']
SEEDS = [1234, 42, 7]

# Map variant → existing seed=1234 file (đã có từ trước)
SEED_1234_MAP = {
    'plan_c_w01': RESULTS_DIR / 'sweep_w0.1.json',
    'plan_d': RESULTS_DIR / 'phase_d_baseline.json',
    'plan_e': RESULTS_DIR / 'phase_e_no_cl_rebuild.json',
}


def load_metrics(variant, seed):
    """Load metrics cho variant+seed. Seed 1234 dùng file pre-existing."""
    if seed == 1234 and variant in SEED_1234_MAP:
        path = SEED_1234_MAP[variant]
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    # New seeds: parse log
    log_path = LOGS_DIR / f'multiseed_{variant}_seed{seed}.log'
    if not log_path.exists():
        return None
    metrics = parse_log(log_path)
    metrics['_source'] = str(log_path)
    metrics['variant'] = variant
    metrics['seed'] = seed
    # Save individual JSON
    out_path = RESULTS_DIR / f'multiseed_{variant}_seed{seed}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '-'


def main():
    # Reference paper baseline
    paper = {'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
             'top1_f1': 0.5970}

    keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']

    summary = {'paper': paper, 'variants': {}}

    print('\n' + '=' * 110)
    print('MULTI-SEED FULL — 3 variants × 3 seeds (1234, 42, 7)')
    print('=' * 110)

    for variant in VARIANTS:
        all_runs = {}
        for seed in SEEDS:
            m = load_metrics(variant, seed)
            if m is not None:
                all_runs[seed] = m

        if not all_runs:
            print(f'\n{variant}: NO data found')
            continue

        stats = {}
        for k in keys:
            vals = [all_runs[s].get(k) for s in all_runs if all_runs[s].get(k) is not None]
            if vals:
                stats[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    'n': len(vals),
                    'values': [float(v) for v in vals],
                }

        summary['variants'][variant] = {'runs': all_runs, 'stats': stats}

        # Print
        print(f'\n{variant} (n={len(all_runs)} seeds):')
        for k in keys:
            if k in stats:
                s = stats[k]
                vs = ', '.join(f'{v:.4f}' for v in s['values'])
                print(f'  {k:<18}: {s["mean"]:.4f} ± {s["std"]:.4f}  [{vs}]')

    # Comparison vs paper
    print('\n' + '=' * 110)
    print('COMPARISON vs PAPER (Top-1 F1)')
    print('=' * 110)
    paper_t1 = paper['top1_f1']
    print(f'Paper baseline: {paper_t1:.4f}')
    print(f'{"Variant":<14} {"Mean ± Std":<22} {"Δ vs paper":<12} {"Significance":<25}')
    print('-' * 80)
    for variant in VARIANTS:
        v_data = summary['variants'].get(variant, {})
        t1_stats = v_data.get('stats', {}).get('top1_f1')
        if t1_stats:
            mean = t1_stats['mean']
            std = t1_stats['std']
            n = t1_stats['n']
            delta = (mean - paper_t1) / paper_t1 * 100
            # 1-sample t-test heuristic: |mean - paper| > 2*SE → significant
            se = std / np.sqrt(n) if n > 0 else float('inf')
            t_stat = abs(mean - paper_t1) / (se + 1e-9)
            sig = 'SIGNIFICANT' if t_stat > 2.0 else 'marginal' if t_stat > 1.0 else 'not significant'
            sign = '+' if delta > 0 else ''
            print(f'{variant:<14} {mean:.4f} ± {std:.4f}{"":>4} {sign}{delta:.1f}%{"":>5} {sig} (t={t_stat:.2f})')

    # Save
    with open(RESULTS_DIR / 'multiseed_full_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/multiseed_full_summary.json')


if __name__ == '__main__':
    main()
