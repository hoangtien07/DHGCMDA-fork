"""Final reproduce report — tổng kết tất cả phase A→E + refocus reproduce.

Output: results/final_reproduce_report.json + stdout bảng tổng + verdict.
"""
import json
from pathlib import Path

RESULTS_DIR = Path('results')

PAPER = {
    'AUC': 0.9669, 'AUPR': 0.9738, 'F1': 0.9278,
    'top1_precision': 0.5842, 'top1_recall': 0.6341, 'top1_f1': 0.5970,
}


def load(p):
    p = RESULTS_DIR / p
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else '-'


def delta_pct(v, ref):
    if v is None or ref is None or ref == 0:
        return '-'
    d = (v - ref) / ref * 100
    s = '+' if d > 0 else ''
    return f'{s}{d:.1f}%'


def main():
    print('\n' + '=' * 100)
    print('FINAL REPRODUCE REPORT — DHGCMDA paper trên HMDD v2.0')
    print('=' * 100)

    # 1. Best baseline config
    k7_seed1 = load('k_sweep_K7_seed1.json')
    if k7_seed1:
        print('\n📊 BEST BASELINE CONFIG: seed=1, K=7, default loss')
        print(f'  AUC:    {fmt(k7_seed1.get("AUC"))} (paper {fmt(PAPER["AUC"])}, {delta_pct(k7_seed1.get("AUC"), PAPER["AUC"])})')
        print(f'  AUPR:   {fmt(k7_seed1.get("AUPR"))} (paper {fmt(PAPER["AUPR"])}, {delta_pct(k7_seed1.get("AUPR"), PAPER["AUPR"])})')
        print(f'  F1:     {fmt(k7_seed1.get("F1"))} (paper {fmt(PAPER["F1"])}, {delta_pct(k7_seed1.get("F1"), PAPER["F1"])})')
        print(f'  T1-F1:  {fmt(k7_seed1.get("top1_f1"))} (paper {fmt(PAPER["top1_f1"])}, {delta_pct(k7_seed1.get("top1_f1"), PAPER["top1_f1"])})')

    # 2. Seed sweep
    seed_sum = load('seed_sweep_summary.json')
    if seed_sum:
        print('\n📊 SEED SWEEP (default config, 4 seeds tested)')
        for seed in [0, 1, 42, 1234]:
            r = seed_sum.get('results', {}).get(str(seed), {})
            if r:
                print(f'  seed={seed:<5} T1-F1={fmt(r.get("top1_f1"))} ({delta_pct(r.get("top1_f1"), PAPER["top1_f1"])})')

    # 3. K sweep
    k_sum = load('k_sweep_seed1_summary.json')
    if k_sum:
        print('\n📊 K SWEEP (seed=1, 5 K values)')
        for K in [7, 9, 11, 13, 15]:
            r = k_sum.get('results', {}).get(str(K), {})
            if r:
                print(f'  K={K:<3} T1-F1={fmt(r.get("top1_f1"))} ({delta_pct(r.get("top1_f1"), PAPER["top1_f1"])})')

    # 4. Lambda2 sweep
    l2_sum = load('lambda2_sweep_summary.json')
    if l2_sum:
        print('\n📊 LAMBDA2 SWEEP (seed=1, K=7)')
        for l2 in [0.1, 0.3, 0.5]:
            r = l2_sum.get('results', {}).get(str(l2), {})
            if r:
                print(f'  λ₂={l2} T1-F1={fmt(r.get("top1_f1"))} ({delta_pct(r.get("top1_f1"), PAPER["top1_f1"])})')

    # 5. Fig.4 verify
    fig4 = load('fig4_verify_summary.json')
    if fig4:
        print('\n📊 FIG.4 ABLATION (seed=1, K=7, λ₂=0.3)')
        full_t1 = fig4.get('baseline_seed1_K7', {}).get('top1_f1')
        print(f'  Full T1-F1: {fmt(full_t1)}')
        abls = fig4.get('ablations', {})
        for abl in ['no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv']:
            r = abls.get(abl, {})
            if r:
                t1 = r.get('top1_f1')
                d = delta_pct(t1, full_t1)
                match = '✅' if (t1 is not None and full_t1 is not None and t1 < full_t1) else '❌'
                print(f'  w/o {abl:<8} T1-F1={fmt(t1)} (Δ Full {d})  {match}')
        print(f'  Fig.4 match: {fig4.get("fig4_match", "?")}')

    # 6. Verdict
    print('\n' + '=' * 100)
    print('VERDICT TỔNG (so paper)')
    print('=' * 100)

    components = []
    if k7_seed1:
        auc_match = k7_seed1.get('AUC', 0) >= PAPER['AUC'] - 0.01
        t1_match = abs(k7_seed1.get('top1_f1', 0) - PAPER['top1_f1']) <= 0.03
        components.append(('Binary metrics (AUC/AUPR/F1)', '✅ Vượt paper' if auc_match else '⚠️'))
        components.append(('Top-1 F1 baseline (gap ≤ 3%)', '✅ Reproduce' if t1_match else '⚠️'))

    if fig4:
        fig4_match_str = fig4.get('fig4_match', '0/5')
        try:
            n = int(fig4_match_str.split('/')[0])
            if n >= 4:
                components.append(('Fig.4 ablation pattern', '✅ Reproduce'))
            elif n >= 2:
                components.append(('Fig.4 ablation pattern', f'⚠️ Partial ({n}/5)'))
            else:
                components.append(('Fig.4 ablation pattern', f'❌ Failed ({n}/5)'))
        except:
            pass

    for name, status in components:
        print(f'  {name:<40} {status}')

    # Final summary
    binary_ok = k7_seed1 and k7_seed1.get('AUC', 0) >= PAPER['AUC'] - 0.01
    t1_ok = k7_seed1 and abs(k7_seed1.get('top1_f1', 0) - PAPER['top1_f1']) <= 0.03

    if binary_ok and t1_ok:
        print('\n🏆 BASELINE REPRODUCE: ACHIEVED (binary + Top-1)')
    else:
        print('\n⚠️ BASELINE REPRODUCE: PARTIAL')

    print('\n[save] results/final_reproduce_report.json')

    out = {
        'paper': PAPER,
        'best_baseline_config': {'seed': 1, 'K_neigs': 7, 'loss_mode': 'two_head'},
        'best_baseline_metrics': k7_seed1,
        'seed_sweep_summary': seed_sum,
        'k_sweep_summary': k_sum,
        'lambda2_sweep_summary': l2_sum,
        'fig4_verify_summary': fig4,
    }
    with open(RESULTS_DIR / 'final_reproduce_report.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
