"""Aggregate Fig.4 ablation rerun với best lambda2 từ sweep.

Compare 3-way: lambda2=0.3 (original Fig.4 verify) vs best lambda2 vs paper claim.
"""
import json
import argparse
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

ABL_MODES = ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']


def parse_abl(abl, l2):
    log = LOGS_DIR / f'fig4_l2_{l2}_{abl}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    m['ablation'] = abl
    m['lambda2'] = l2
    out = RESULTS_DIR / f'fig4_l2_{l2}_{abl}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m


def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else '-'


def delta_pct(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    sign = '+' if d > 0 else ''
    return f'{sign}{d:.1f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda2', type=float, required=True)
    args = parser.parse_args()
    l2 = args.lambda2

    # Load baseline (Full with this lambda2)
    baseline_path = RESULTS_DIR / f'lambda2_sweep_l2_{l2}.json'
    if not baseline_path.exists():
        print(f'[ERROR] Missing baseline {baseline_path}')
        return
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    full_t1 = baseline.get('top1_f1')

    # Parse 5 ablations
    results = {}
    for abl in ABL_MODES:
        m = parse_abl(abl, l2)
        if m is not None:
            results[abl] = m

    print('\n' + '=' * 100)
    print(f'FIG.4 VERIFY with lambda2={l2} (seed=1, K=7, default loss)')
    print('=' * 100)
    print(f'{"Variant":<22} {"AUC":>8} {"T1-F1":>8}  {"Δ Full":>10} {"Match":>10}')
    print('-' * 100)
    print(f'{"Full":<22} {fmt(baseline.get("AUC")):>8} {fmt(full_t1):>8}  {"---":>10} {"baseline":>10}')

    n_correct = 0
    for abl in ABL_MODES:
        if abl not in results:
            print(f'{"w/o " + abl:<22} (missing)')
            continue
        m = results[abl]
        t1 = m.get('top1_f1')
        d = delta_pct(t1, full_t1)
        match = '-'
        if t1 is not None and full_t1 is not None:
            if t1 < full_t1:
                match = 'YES ✅'
                n_correct += 1
            else:
                match = 'NO ❌'
        print(f'{"w/o " + abl:<22} {fmt(m.get("AUC")):>8} {fmt(t1):>8}  {d:>10} {match:>10}')

    print(f'\nFig.4 match: {n_correct}/5 (lambda2={l2})')

    # Compare with original Fig.4 verify (lambda2=0.3)
    orig_path = RESULTS_DIR / 'fig4_verify_summary.json'
    if orig_path.exists():
        with open(orig_path, 'r', encoding='utf-8') as f:
            orig = json.load(f)
        orig_match = orig.get('fig4_match', '?')
        print(f'\nCompare:')
        print(f'  Original (lambda2=0.3): {orig_match}')
        print(f'  New (lambda2={l2}):     {n_correct}/5')

    summary = {
        'baseline': baseline,
        'ablations': results,
        'lambda2': l2,
        'fig4_match': f'{n_correct}/5',
    }
    out = RESULTS_DIR / f'fig4_l2_{l2}_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] {out}')


if __name__ == '__main__':
    main()
