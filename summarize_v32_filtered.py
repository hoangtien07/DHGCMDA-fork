"""Aggregate v3.2 filtered experiment — baseline + 5 ablations.

Compare with:
- v2.0 baseline + 5 ablations (existing fig4_verify with seed=1, K=7)
- Paper v3.2 numbers (if available)
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

PAPER_V20 = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}

# Paper v3.2 numbers — TBD from paper Table 3 v3.2 row (placeholder)
PAPER_V32 = {
    'AUC': None, 'AUPR': None, 'F1': None,
    'top1_f1': None,
    'note': 'Paper Table 3 v3.2 — need to extract from paper text',
}

ABL_MODES = ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']


def parse_run(name):
    log = LOGS_DIR / f'{name}.log'
    if not log.exists():
        return None
    m = parse_log(log)
    out = RESULTS_DIR / f'{name}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return m


def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else '-'


def delta_pct(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    s = '+' if d > 0 else ''
    return f'{s}{d:.1f}%'


def main():
    # Baseline
    baseline = parse_run('v32_baseline')
    if baseline is None:
        print('[ERROR] v32_baseline log not found')
        return

    # Ablations
    abls = {}
    for abl in ABL_MODES:
        m = parse_run(f'v32_{abl}')
        if m is not None:
            abls[abl] = m

    # v2.0 comparison: fig4_verify ablations
    v20_baseline_path = RESULTS_DIR / 'k_sweep_K7_seed1.json'
    v20_baseline = json.load(open(v20_baseline_path)) if v20_baseline_path.exists() else None
    v20_abls = {}
    for abl in ABL_MODES:
        p = RESULTS_DIR / f'fig4_verify_{abl}.json'
        if p.exists():
            v20_abls[abl] = json.load(open(p))

    full_t1 = baseline.get('top1_f1')

    print('\n' + '=' * 110)
    print('v3.2 FILTERED EXPERIMENT — Baseline + 5 ablations')
    print('Config: seed=1, K=7, default loss, dataset=v3.2_filtered_495m383D')
    print('=' * 110)

    print('\n## BASELINE COMPARISON ##')
    print(f'{"":<24} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}')
    print('-' * 90)
    print(f'{"Paper v2.0":<24} {fmt(PAPER_V20["AUC"]):>8} {fmt(PAPER_V20["AUPR"]):>8} '
          f'{fmt(PAPER_V20["F1"]):>8} {fmt(PAPER_V20["top1_precision"]):>8} '
          f'{fmt(PAPER_V20["top1_recall"]):>8} {fmt(PAPER_V20["top1_f1"]):>8}')
    if v20_baseline:
        keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
        print(f'{"v2.0 reproduce (K=7)":<24} ' +
              ' '.join(f'{fmt(v20_baseline.get(k)):>8}' for k in keys))
    print(f'{"v3.2 filtered (this)":<24} ' +
          ' '.join(f'{fmt(baseline.get(k)):>8}' for k in
                   ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']))

    print('\n## FIG.4 ABLATION COMPARISON — v2.0 vs v3.2 filtered ##')
    print('(Paper claim: ALL 5 ablations hurt baseline)')
    print(f'{"Variant":<14} {"v2.0 T1-F1":>10} {"v2.0 Δ":>10} {"v3.2 T1-F1":>10} {"v3.2 Δ":>10}  {"v2.0":>6} {"v3.2":>6}')
    print('-' * 90)
    v20_full = v20_baseline.get('top1_f1') if v20_baseline else None
    print(f'{"Full":<14} {fmt(v20_full):>10} {"---":>10} {fmt(full_t1):>10} {"---":>10}  '
          f'{"base":>6} {"base":>6}')

    n_v20_correct = 0
    n_v32_correct = 0
    n_total = 0
    for abl in ABL_MODES:
        v20 = v20_abls.get(abl, {}).get('top1_f1')
        v32 = abls.get(abl, {}).get('top1_f1')
        v20_d = delta_pct(v20, v20_full)
        v32_d = delta_pct(v32, full_t1)
        v20_m = '-'
        v32_m = '-'
        if v20 is not None and v20_full is not None:
            v20_m = '✅' if v20 < v20_full else '❌'
            n_total += 1
            if v20 < v20_full:
                n_v20_correct += 1
        if v32 is not None and full_t1 is not None:
            v32_m = '✅' if v32 < full_t1 else '❌'
            if v32 < full_t1:
                n_v32_correct += 1
        print(f'{"w/o " + abl:<14} {fmt(v20):>10} {v20_d:>10} {fmt(v32):>10} {v32_d:>10}  '
              f'{v20_m:>6} {v32_m:>6}')

    print('-' * 90)
    print(f'\nFig.4 match: v2.0 = {n_v20_correct}/{n_total}, v3.2 filtered = {n_v32_correct}/{n_total}')

    print('\n' + '=' * 110)
    print('VERDICT')
    print('=' * 110)
    if n_v32_correct > n_v20_correct:
        print(f'🎉 v3.2 dataset IMPROVES Fig.4 match ({n_v32_correct}/5 vs v2.0 {n_v20_correct}/5)')
        print('   Hypothesis "dataset is key" SUPPORTED — paper Fig.4 reproducible với v3.2 data')
    elif n_v32_correct == n_v20_correct:
        print(f'⚠️ Same Fig.4 match across v2.0 and v3.2 filtered ({n_v20_correct}/5)')
        print('   Dataset alone does NOT explain Fig.4 pattern')
    else:
        print(f'❌ v3.2 worse than v2.0 ({n_v32_correct}/5 vs {n_v20_correct}/5)')
        print('   Strong evidence: paper Fig.4 claim does not generalize')

    # Save
    summary = {
        'config': {'seed': 1, 'K_neigs': 7, 'dataset': 'v3.2_filtered_495m383D'},
        'baseline_v32': baseline,
        'baseline_v20': v20_baseline,
        'paper_v20': PAPER_V20,
        'ablations_v32': abls,
        'ablations_v20': v20_abls,
        'fig4_match_v32': f'{n_v32_correct}/{n_total}',
        'fig4_match_v20': f'{n_v20_correct}/{n_total}',
    }
    with open(RESULTS_DIR / 'v32_filtered_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/v32_filtered_summary.json')


if __name__ == '__main__':
    main()
