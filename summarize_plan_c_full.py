"""Aggregate ALL Plan C results: sweep (loss), case study (w=0.1), ablation
(w=0.1) → bảng tổng so sánh paper Fig. 4 và Tables 5/6.

Cách dùng (sau khi background bjhe6g7gd xong):
    python summarize_plan_c_full.py

Output:
    results/ablation_w0.1_*.json (5 files)
    results/plan_c_full_summary.json
    Stdout: 3 bảng (loss sweep / ablation Fig.4 verify / case study verify)
"""
import json
from pathlib import Path

from parse_metrics import parse_log

RESULTS_DIR = Path('results')
LOGS_DIR = Path('logs')

ABL_MODES = ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']

# Paper Fig. 4 — Top-1 F1 expected behavior (paper bảo TẤT CẢ ablation hurt)
# Numbers approx từ paper Fig.4 đọc qua chart
PAPER_FIG4_TOP1F1 = {
    'baseline': 0.5970,
    # Paper bảo all ablation < baseline (hurt). Approx:
    'no_cl':    0.5750,  # ~3.7% drop trong paper
    'no_hgcn':  0.5800,
    'no_avf':   0.5850,
    'no_hgt':   0.5780,
    'no_dv':    0.5820,
}


def parse_ablation(mode):
    log_path = LOGS_DIR / f'abl_w0.1_{mode}.log'
    out_path = RESULTS_DIR / f'ablation_w0.1_{mode}.json'

    if not log_path.exists():
        return None
    metrics = parse_log(log_path)
    metrics['_source'] = str(log_path)
    metrics['ablation'] = mode
    metrics['exist_weight'] = 0.1

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
    # =================================================================
    # 1. LOSS SWEEP — đọc lại từ plan_c_comparison.json (đã có)
    # =================================================================
    plan_c = {}
    p1 = RESULTS_DIR / 'plan_c_comparison.json'
    if p1.exists():
        with open(p1, 'r', encoding='utf-8') as f:
            plan_c = json.load(f)

    print('\n' + '=' * 90)
    print('A. LOSS SWEEP (exist_weight) — Plan C-1/2/3/4')
    print('=' * 90)
    print(f'{"Run":<28} {"AUC":>8} {"AUPR":>8} {"F1":>8} {"T1-F1":>8}  {"Δ vs paper":<12}')
    print('-' * 90)
    paper_t1 = plan_c.get('paper', {}).get('top1_f1', 0.5970)
    for label, key in [('Paper', 'paper'),
                       ('Phase A (w=0.3 orig)', 'phase_A_orig'),
                       ('Phase B-C (w=0.3 fix)', 'phase_B_C_fix3'),
                       ('Phase C-w0.1', 'C-w0.1'),
                       ('Phase C-w0.05', 'C-w0.05'),
                       ('Phase C-w0.0', 'C-w0.0')]:
        d = plan_c.get(key, {})
        # Try fallback for new keys without 'C-' prefix
        if not d:
            d = plan_c.get(key.replace('C-', ''), {})
        print(f'{label:<28} {fmt(d.get("AUC")):>8} {fmt(d.get("AUPR")):>8} '
              f'{fmt(d.get("F1")):>8} {fmt(d.get("top1_f1")):>8}  '
              f'{delta_pct(d.get("top1_f1"), paper_t1):<12}')

    # =================================================================
    # 2. ABLATION VERIFY (Fig. 4) — w=0.1 trên 5 variant
    # =================================================================
    abl_results = {}
    for mode in ABL_MODES:
        m = parse_ablation(mode)
        if m is not None:
            abl_results[mode] = m

    # baseline ở w=0.1 đã có trong plan_c['C-w0.1']
    baseline_w01 = plan_c.get('C-w0.1') or plan_c.get('w0.1') or {}

    print('\n' + '=' * 100)
    print('B. ABLATION VERIFY Fig.4 (exist_weight=0.1) — paper bảo TẤT CẢ ablation phải HURT')
    print('=' * 100)
    print(f'{"Variant":<22} {"AUC":>8} {"T1-F1":>8} {"Δ vs Full(w=0.1)":>20}  {"Paper expects":<22}')
    print('-' * 100)
    full_t1 = baseline_w01.get('top1_f1')
    print(f'{"Full DHGCMDA (w=0.1)":<22} {fmt(baseline_w01.get("AUC")):>8} '
          f'{fmt(full_t1):>8} {"":>20}  {"baseline":<22}')
    n_correct = 0
    n_total = 0
    for mode in ABL_MODES:
        d = abl_results.get(mode, {})
        t1 = d.get('top1_f1')
        delta_str = delta_pct(t1, full_t1)
        # Paper expectation: ablation should be LOWER than baseline
        paper_t1_abl = PAPER_FIG4_TOP1F1.get(mode, full_t1)
        paper_expects = f'~{paper_t1_abl:.4f} (lower)'
        if t1 is not None and full_t1 is not None:
            n_total += 1
            if t1 < full_t1:  # match paper expectation
                marker = '✅'
                n_correct += 1
            else:
                marker = '❌'
        else:
            marker = '-'
        print(f'{"w/o " + mode:<22} {fmt(d.get("AUC")):>8} {fmt(t1):>8} '
              f'{delta_str:>20}  {paper_expects:<22} {marker}')
    print('-' * 100)
    if n_total > 0:
        print(f'\nFig.4 pattern verify: {n_correct}/{n_total} ablation match paper '
              f'(lower than baseline). '
              f'{"FULL MATCH" if n_correct == n_total else "PARTIAL" if n_correct > 0 else "STILL INVERTED"}')

    # =================================================================
    # 3. CASE STUDY VERIFY (Tables 5/6) — w=0.1
    # =================================================================
    cs_path = RESULTS_DIR / 'case_study_summary.json'
    cs_old_path = RESULTS_DIR / 'snapshot_phaseBC_w0.3' / 'case_study_summary.json'
    cs_new = json.load(open(cs_path, 'r', encoding='utf-8')) if cs_path.exists() else {}
    cs_old = json.load(open(cs_old_path, 'r', encoding='utf-8')) if cs_old_path.exists() else {}

    print('\n' + '=' * 90)
    print('C. CASE STUDY VERIFY (Tables 5/6) — paper xác nhận 13/15 breast + 12/15 HCC')
    print('=' * 90)
    print(f'{"":<28} {"Phase B-C (w=0.3)":>20} {"Phase C-w0.1":>20}')
    print('-' * 90)
    for disease, paper_count in [('breast', 13), ('hcc', 12)]:
        old = cs_old.get(disease, {})
        new = cs_new.get(disease, {})
        old_str = f'{old.get("overlap_count", "-")}/15 (type:{old.get("type_match_count", "-")}/15)'
        new_str = f'{new.get("overlap_count", "-")}/15 (type:{new.get("type_match_count", "-")}/15)'
        print(f'{disease + " (paper " + str(paper_count) + "/15 PMID)":<28} {old_str:>20} {new_str:>20}')

    # Type diversity check
    if cs_new:
        print('\nType diversity check (should be 4 types per disease for full reproduce):')
        for disease in ['breast', 'hcc']:
            top15 = cs_new.get(disease, {}).get('top15', [])
            types = {}
            for r in top15:
                t = r.get('predicted_type', '?')
                types[t] = types.get(t, 0) + 1
            type_str = ', '.join(f'{k}:{v}' for k, v in types.items())
            print(f'  {disease}: {type_str}')

    # =================================================================
    # SAVE FULL SUMMARY
    # =================================================================
    full = {
        'loss_sweep': plan_c,
        'ablation_w0.1': abl_results,
        'case_study_old': cs_old,
        'case_study_new': cs_new,
        'fig4_match_count': f'{n_correct}/{n_total}' if n_total else 'N/A',
    }
    with open(RESULTS_DIR / 'plan_c_full_summary.json', 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    print(f'\n[save] results/plan_c_full_summary.json')


if __name__ == '__main__':
    main()
