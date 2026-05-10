"""Aggregate Phase D (Fix A++ softmax_5class) results: parse logs/phase_d_*.log,
build comparison table với paper, Plan C-w0.1, Phase D.

Cách dùng:
    python summarize_phase_d.py

Output:
    results/phase_d_baseline.json
    results/phase_d_no_*.json (5 ablation)
    results/phase_d_summary.json
    Stdout: 3 bảng (baseline / Fig.4 verify / case study verify)
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

ABL_MODES = ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']


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
    # Parse baseline + 5 ablation
    baseline = parse_log_to_json('phase_d_baseline', 'phase_d_baseline',
                                  {'loss_mode': 'softmax_5class', 'variant': 'baseline'})
    abl_results = {}
    for mode in ABL_MODES:
        m = parse_log_to_json(f'phase_d_{mode}', f'phase_d_{mode}',
                              {'loss_mode': 'softmax_5class', 'ablation': mode})
        if m is not None:
            abl_results[mode] = m

    # Reference data (paper + Plan C-w0.1)
    paper = {'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
             'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970}

    # Plan C-w0.1 baseline + ablations
    plan_c_baseline = {}
    p1 = RESULTS_DIR / 'plan_c_comparison.json'
    if p1.exists():
        with open(p1, 'r', encoding='utf-8') as f:
            d = json.load(f)
        plan_c_baseline = d.get('C-w0.1') or d.get('w0.1') or {}

    plan_c_ablations = {}
    for mode in ABL_MODES:
        ap = RESULTS_DIR / f'ablation_w0.1_{mode}.json'
        if ap.exists():
            with open(ap, 'r', encoding='utf-8') as f:
                plan_c_ablations[mode] = json.load(f)

    paper_t1 = paper['top1_f1']

    # =================================================================
    # A. BASELINE COMPARISON
    # =================================================================
    print('\n' + '=' * 100)
    print('A. BASELINE — Phase D Fix A++ (softmax_5class) vs Plan C-w0.1 (two_head) vs Paper')
    print('=' * 100)
    print(f'{"Run":<32} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-P":>8} {"T1-R":>8} {"T1-F1":>8}  {"Δ vs paper":<14}')
    print('-' * 100)
    rows = [
        ('Paper', paper),
        ('Plan C-w0.1 (two_head)', plan_c_baseline),
        ('Phase D Fix A++ (softmax_5class)', baseline or {}),
    ]
    for label, data in rows:
        keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
        vals = [fmt(data.get(k)) for k in keys]
        d = delta_pct(data.get('top1_f1'), paper_t1)
        print(f'{label:<32} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8}  {d:<14}')

    # =================================================================
    # B. FIG.4 ABLATION VERIFY — Phase D vs Plan C-w0.1
    # =================================================================
    print('\n' + '=' * 100)
    print('B. FIG.4 ABLATION VERIFY — paper bảo TẤT CẢ ablation phải HURT (lower than baseline)')
    print('=' * 100)
    full_t1 = baseline.get('top1_f1') if baseline else None
    plan_c_full_t1 = plan_c_baseline.get('top1_f1')
    print(f'{"Variant":<22} {"D T1-F1":>10} {"D Δ Full":>10} {"D Match":>10}  {"C T1-F1":>10} {"C Δ Full":>10} {"C Match":>10}')
    print('-' * 100)
    print(f'{"Full":<22} {fmt(full_t1):>10} {"":>10} {"":>10}  {fmt(plan_c_full_t1):>10} {"":>10} {"":>10}')
    n_d_correct = 0
    n_c_correct = 0
    for mode in ABL_MODES:
        d_t1 = abl_results.get(mode, {}).get('top1_f1')
        c_t1 = plan_c_ablations.get(mode, {}).get('top1_f1')
        d_delta = delta_pct(d_t1, full_t1)
        c_delta = delta_pct(c_t1, plan_c_full_t1)
        d_match = '-'
        c_match = '-'
        if d_t1 is not None and full_t1 is not None:
            if d_t1 < full_t1:
                d_match = 'YES'
                n_d_correct += 1
            else:
                d_match = 'NO'
        if c_t1 is not None and plan_c_full_t1 is not None:
            if c_t1 < plan_c_full_t1:
                c_match = 'YES'
                n_c_correct += 1
            else:
                c_match = 'NO'
        print(f'{"w/o " + mode:<22} {fmt(d_t1):>10} {d_delta:>10} {d_match:>10}  {fmt(c_t1):>10} {c_delta:>10} {c_match:>10}')
    print('-' * 100)
    print(f'\nFig.4 match: Phase D = {n_d_correct}/5  vs  Plan C-w0.1 = {n_c_correct}/5')

    # =================================================================
    # C. CASE STUDY VERIFY
    # =================================================================
    print('\n' + '=' * 100)
    print('C. CASE STUDY VERIFY — paper xác nhận 13/15 breast + 12/15 HCC')
    print('=' * 100)
    cs_path = RESULTS_DIR / 'case_study_summary.json'
    cs_old_path = RESULTS_DIR / 'snapshot_planC_w0.1' / 'case_study_summary.json'
    cs_new = json.load(open(cs_path, 'r', encoding='utf-8')) if cs_path.exists() else {}
    cs_plan_c = json.load(open(cs_old_path, 'r', encoding='utf-8')) if cs_old_path.exists() else {}

    print(f'{"":<28} {"Plan C-w0.1":>20} {"Phase D Fix A++":>20}')
    print('-' * 100)
    for disease, paper_count in [('breast', 13), ('hcc', 12)]:
        old = cs_plan_c.get(disease, {})
        new = cs_new.get(disease, {})
        old_str = f'{old.get("overlap_count", "-")}/15 (type:{old.get("type_match_count", "-")}/15)'
        new_str = f'{new.get("overlap_count", "-")}/15 (type:{new.get("type_match_count", "-")}/15)'
        print(f'{disease + " (paper " + str(paper_count) + "/15 PMID)":<28} {old_str:>20} {new_str:>20}')

    print('\nType diversity check (paper has 4 types per disease):')
    for tag, cs in [('Plan C-w0.1', cs_plan_c), ('Phase D Fix A++', cs_new)]:
        if not cs:
            continue
        print(f'  {tag}:')
        for disease in ('breast', 'hcc'):
            top15 = cs.get(disease, {}).get('top15', [])
            types = {}
            for r in top15:
                t = r.get('predicted_type', '?')
                types[t] = types.get(t, 0) + 1
            type_str = ', '.join(f'{k}:{v}' for k, v in sorted(types.items(), key=lambda x: -x[1]))
            print(f'    {disease}: {type_str}')

    # SAVE SUMMARY
    full = {
        'paper': paper,
        'plan_c_baseline': plan_c_baseline,
        'plan_c_ablations': plan_c_ablations,
        'phase_d_baseline': baseline,
        'phase_d_ablations': abl_results,
        'phase_d_fig4_match': f'{n_d_correct}/5',
        'plan_c_fig4_match': f'{n_c_correct}/5',
        'case_study_plan_c': cs_plan_c,
        'case_study_phase_d': cs_new,
    }
    with open(RESULTS_DIR / 'phase_d_summary.json', 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/phase_d_summary.json')

    # Verdict
    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)
    if full_t1 is not None and full_t1 >= paper_t1:
        print(f'✅ Top-1 F1 {full_t1:.4f} ≥ paper {paper_t1:.4f} — baseline match/exceed paper.')
    elif full_t1 is not None:
        print(f'⚠️ Top-1 F1 {full_t1:.4f} < paper {paper_t1:.4f} — baseline below paper.')
    if n_d_correct >= 4:
        print(f'✅ Fig.4 pattern: {n_d_correct}/5 ablation match paper — RESTORED.')
    elif n_d_correct >= 2:
        print(f'⚠️ Fig.4 pattern: {n_d_correct}/5 ablation match paper — PARTIAL.')
    else:
        print(f'❌ Fig.4 pattern: {n_d_correct}/5 ablation match paper — STILL INVERTED.')


if __name__ == '__main__':
    main()
