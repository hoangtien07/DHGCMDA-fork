"""Aggregate Fig.4 ablation verify với reproduce config (seed=1, K=7).

Paper Fig.4 claim: ALL ablation HURT baseline Top-1 F1.

Baseline: (seed=1, K=7) Top-1 F1 = 0.5909 (gap -1.0% paper, REPRODUCED).
Ablation must show LOWER T1-F1 than baseline để match paper.
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

ABL_MODES = ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']
BASELINE_PATH = RESULTS_DIR / 'k_sweep_K7_seed1.json'
PAPER_BASELINE_T1F1 = 0.5970


def parse_abl(abl):
    log = LOGS_DIR / f'fig4_verify_{abl}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    m['ablation'] = abl
    m['seed'] = 1
    m['K_neigs'] = 7
    out = RESULTS_DIR / f'fig4_verify_{abl}.json'
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
    # Load baseline (K=7 seed=1)
    if not BASELINE_PATH.exists():
        print(f'[ERROR] Missing baseline {BASELINE_PATH}')
        return
    with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    full_t1 = baseline.get('top1_f1')

    # Parse ablations
    results = {}
    for abl in ABL_MODES:
        m = parse_abl(abl)
        if m is not None:
            results[abl] = m

    print('\n' + '=' * 100)
    print('FIG.4 VERIFY — reproduce config (seed=1, K=7, default loss)')
    print('Paper claim: ALL ablation HURT baseline Top-1 F1')
    print('=' * 100)
    print(f'{"Variant":<22} {"AUC":>8} {"T1-F1":>8}  {"Δ Full":>10} {"Match paper":>15}')
    print('-' * 100)
    print(f'{"Full DHGCMDA (K=7)":<22} {fmt(baseline.get("AUC")):>8} {fmt(full_t1):>8}  {"---":>10} {"baseline":>15}')

    n_correct = 0
    n_total = 0
    for abl in ABL_MODES:
        if abl not in results:
            print(f'{"w/o " + abl:<22} (missing)')
            continue
        m = results[abl]
        t1 = m.get('top1_f1')
        d = delta_pct(t1, full_t1)
        match = '-'
        if t1 is not None and full_t1 is not None:
            n_total += 1
            if t1 < full_t1:
                match = 'YES ✅'
                n_correct += 1
            else:
                match = 'NO ❌'
        print(f'{"w/o " + abl:<22} {fmt(m.get("AUC")):>8} {fmt(t1):>8}  {d:>10} {match:>15}')

    print('-' * 100)
    print(f'\nFig.4 match: {n_correct}/{n_total} ablations hurt baseline')

    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)
    if n_correct == n_total and n_total > 0:
        print(f'🏆 FULL MATCH: paper Fig.4 REPRODUCED voi (seed=1, K=7).')
        print(f'   All {n_total} ablation hurt baseline as paper claims.')
        print(f'   Reproduce goal: ACHIEVED for binary + Top-1 + Fig.4.')
    elif n_correct >= 3:
        print(f'✅ PARTIAL MATCH: {n_correct}/{n_total} ablations match paper.')
        print(f'   Stronger than Plan E (0/3 rebuild) but not full.')
    else:
        print(f'❌ STILL NOT REPRODUCE: only {n_correct}/{n_total} match.')
        print(f'   Need fallback: λ₂ sweep, class weighting, or HMDD v3.2.')

    # Save
    summary = {
        'baseline_seed1_K7': baseline,
        'ablations': results,
        'paper_baseline_t1f1': PAPER_BASELINE_T1F1,
        'fig4_match': f'{n_correct}/{n_total}',
    }
    with open(RESULTS_DIR / 'fig4_verify_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/fig4_verify_summary.json')


if __name__ == '__main__':
    main()
