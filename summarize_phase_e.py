"""Aggregate Phase E (Plan E true ablation rebuild) results.

Compare 3 rebuild modes vs Phase D baseline (cùng loss_mode softmax_5class)
+ Plan C-w0.1 baseline (two_head). Verify Fig.4 pattern có được khôi phục
khi rebuild ablation đúng cách hay không.

Cách dùng:
    python summarize_phase_e.py

Output:
    results/phase_e_no_*_rebuild.json (3 files)
    results/phase_e_summary.json
    Stdout: bảng so sánh + verdict 4 scenarios (A/B/C/D)
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

REBUILD_MODES = ['no_cl_rebuild', 'no_hgcn_rebuild', 'no_hgt_rebuild']

# Reference: paper baseline + Phase D baseline (đã có)
PAPER = {'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
         'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970}


def parse_log_to_json(log_stem, json_stem, extra=None):
    log_path = LOGS_DIR / f'{log_stem}.log'
    out_path = RESULTS_DIR / f'{json_stem}.json'
    if not log_path.exists():
        return None
    metrics = parse_log(log_path)
    metrics['_source'] = str(log_path)
    if extra:
        metrics.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '-'


def delta_pct(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    sign = '+' if d > 0 else ''
    return f'{sign}{d:.1f}%'


def main():
    # Parse 3 rebuild logs
    rebuild_results = {}
    for mode in REBUILD_MODES:
        m = parse_log_to_json(f'phase_e_{mode}', f'phase_e_{mode}',
                              {'loss_mode': 'softmax_5class', 'ablation': mode})
        if m is not None:
            rebuild_results[mode] = m

    # Reference: Phase D baseline (đã có)
    phase_d_baseline = {}
    pd_path = RESULTS_DIR / 'phase_d_baseline.json'
    if pd_path.exists():
        with open(pd_path, 'r', encoding='utf-8') as f:
            phase_d_baseline = json.load(f)

    # Reference: Phase D additive ablations (đã có) — so sánh với rebuild
    phase_d_ablations = {}
    for legacy_mode in ['no_cl', 'no_hgcn', 'no_hgt']:
        ap = RESULTS_DIR / f'phase_d_{legacy_mode}.json'
        if ap.exists():
            with open(ap, 'r', encoding='utf-8') as f:
                phase_d_ablations[legacy_mode] = json.load(f)

    paper_t1 = PAPER['top1_f1']
    full_t1 = phase_d_baseline.get('top1_f1') if phase_d_baseline else None

    # =================================================================
    # A. PLAN E REBUILD vs PHASE D ADDITIVE — CORE COMPARISON
    # =================================================================
    print('\n' + '=' * 110)
    print('A. PLAN E REBUILD vs PHASE D ADDITIVE — verify Fig.4 hypothesis')
    print(f'   Phase D baseline Top-1 F1 = {fmt(full_t1)} (paper {paper_t1:.4f}, +{(full_t1-paper_t1)/paper_t1*100:+.1f}% vs paper)' if full_t1 else '   (Phase D baseline not loaded)')
    print('=' * 110)
    print(f'{"Variant":<28} {"AUC":>8} {"T1-F1":>8} {"Δ Full":>10}  {"Match paper expect":<22}')
    print('-' * 110)

    rebuild_correct = 0
    additive_correct = 0
    for legacy, rebuild in zip(['no_cl', 'no_hgcn', 'no_hgt'], REBUILD_MODES):
        # Additive (Phase D)
        d_t1 = phase_d_ablations.get(legacy, {}).get('top1_f1')
        d_auc = phase_d_ablations.get(legacy, {}).get('AUC')
        d_delta = delta_pct(d_t1, full_t1)
        d_match = '-'
        if d_t1 is not None and full_t1 is not None:
            if d_t1 < full_t1:
                d_match = '✅ YES (lower)'
                additive_correct += 1
            else:
                d_match = '❌ NO (higher)'
        print(f'{legacy + " (additive)":<28} {fmt(d_auc):>8} {fmt(d_t1):>8} {d_delta:>10}  {d_match:<22}')

        # Rebuild (Plan E)
        r_t1 = rebuild_results.get(rebuild, {}).get('top1_f1')
        r_auc = rebuild_results.get(rebuild, {}).get('AUC')
        r_delta = delta_pct(r_t1, full_t1)
        r_match = '-'
        if r_t1 is not None and full_t1 is not None:
            if r_t1 < full_t1:
                r_match = '✅ YES (lower)'
                rebuild_correct += 1
            else:
                r_match = '❌ NO (higher)'
        print(f'{rebuild + " (rebuild)":<28} {fmt(r_auc):>8} {fmt(r_t1):>8} {r_delta:>10}  {r_match:<22}')
        print()  # separator
    print('-' * 110)
    print(f'\nFig.4 match: Phase D additive = {additive_correct}/3  vs  Plan E rebuild = {rebuild_correct}/3')

    # =================================================================
    # B. VERDICT 4 SCENARIOS
    # =================================================================
    print('\n' + '=' * 110)
    print('B. VERDICT — Plan E hypothesis test')
    print('=' * 110)
    if rebuild_correct == 3:
        print('🏆 SCENARIO A: All 3 rebuild ablation HURT baseline (match paper Fig.4)')
        print('   → Paper claim VALID khi rebuild ablation correctly. Khuyến nghị upstream fix.')
        print('   → STRONG positive replication result.')
    elif rebuild_correct == 2:
        print('✅ SCENARIO B: 2/3 rebuild match paper')
        print('   → Replication: paper partially correct')
    elif rebuild_correct == 1:
        print('⚠️ SCENARIO C: 1/3 rebuild match paper')
        print('   → Replication: paper Fig.4 limited — components not universally critical')
    else:
        print('❌ SCENARIO D: 0/3 rebuild hurt baseline (rebuild ALSO inverts)')
        print('   → STRONG negative replication: paper Fig.4 likely seed/hyperparam artifact')
        print('   → Hypothesis "ablation impl khác paper" REJECTED.')
        print('   → Paper angle shift: fundamental robustness concern.')

    # Compare additive vs rebuild
    if rebuild_correct > additive_correct:
        print(f'\n→ Plan E rebuild matches paper BETTER than additive ({rebuild_correct} vs {additive_correct})')
        print('  Confirms hypothesis: additive switch ablation impl khác paper.')
    elif rebuild_correct < additive_correct:
        print(f'\n→ Plan E rebuild matches paper WORSE than additive ({rebuild_correct} vs {additive_correct})')
        print('  Hypothesis "additive impl khác paper" REJECTED. Issue elsewhere.')
    else:
        print(f'\n→ Same match count ({rebuild_correct} = {additive_correct}). No significant difference.')

    # SAVE SUMMARY
    summary = {
        'paper': PAPER,
        'phase_d_baseline': phase_d_baseline,
        'phase_d_additive_ablations': phase_d_ablations,
        'phase_e_rebuild_ablations': rebuild_results,
        'fig4_match_additive': f'{additive_correct}/3',
        'fig4_match_rebuild': f'{rebuild_correct}/3',
        'verdict': (
            'A_strong_positive' if rebuild_correct == 3 else
            'B_partial' if rebuild_correct == 2 else
            'C_weak' if rebuild_correct == 1 else
            'D_strong_negative'
        ),
    }
    with open(RESULTS_DIR / 'phase_e_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/phase_e_summary.json')


if __name__ == '__main__':
    main()
