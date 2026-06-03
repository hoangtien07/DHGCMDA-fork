"""Plan H-2: Aggregate multi-seed ablation verification → answer: reversal REAL or NOISE?

Reads logs/ms_{variant}_seed{s}.log for variant in {baseline, no_cl, no_hgt}, seed in {0,1,42,1234}.
seed=1 reuses Plan C w=0.1 data (baseline 0.5996, no_cl 0.6286, no_hgt 0.6452).
Computes per-seed delta (ablation - baseline), then mean ± std across seeds.
Verdict: if mean delta > 2*std → REAL reversal; else within noise.
"""
import os
import re
import json
import glob
import numpy as np


def read_log_top1(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-16-le' if raw[:2] == b'\xff\xfe' else 'utf-8', errors='replace')
    if 'COMPREHENSIVE RESULTS' not in text:
        return None
    final = text[text.rfind('COMPREHENSIVE RESULTS'):]
    m = re.search(r'Top-1 F1\s*:\s*([\d.]+)', final)
    return float(m.group(1)) if m else None


def main():
    # seed=1 from Plan C w=0.1 (already have)
    seed1 = {'baseline': 0.5996, 'no_cl': 0.6286, 'no_hgt': 0.6452}

    seeds = [0, 1, 42, 1234]
    variants = ['baseline', 'no_cl', 'no_hgt']
    data = {v: {} for v in variants}
    data['baseline'][1] = seed1['baseline']
    data['no_cl'][1] = seed1['no_cl']
    data['no_hgt'][1] = seed1['no_hgt']

    for v in variants:
        for s in [0, 42, 1234]:
            val = read_log_top1(f'logs/ms_{v}_seed{s}.log')
            if val is not None:
                data[v][s] = val

    print('=' * 70)
    print('Plan H-2: Multi-seed ablation verification (exist_weight=0.1)')
    print('=' * 70)
    print(f"{'seed':>6} {'baseline':>10} {'no_cl':>10} {'no_hgt':>10} {'Δcl':>8} {'Δhgt':>8}")
    print('-' * 70)

    deltas_cl, deltas_hgt = [], []
    for s in seeds:
        b = data['baseline'].get(s)
        cl = data['no_cl'].get(s)
        hgt = data['no_hgt'].get(s)
        if b is None:
            continue
        dcl = (cl - b) if cl is not None else None
        dhgt = (hgt - b) if hgt is not None else None
        if dcl is not None:
            deltas_cl.append(dcl)
        if dhgt is not None:
            deltas_hgt.append(dhgt)
        print(f"{s:>6} {b:>10.4f} "
              f"{cl if cl else 0:>10.4f} {hgt if hgt else 0:>10.4f} "
              f"{dcl*100 if dcl is not None else 0:>+7.1f}% {dhgt*100 if dhgt is not None else 0:>+7.1f}%")

    print('=' * 70)

    def verdict(deltas, name):
        if len(deltas) < 2:
            print(f'{name}: insufficient seeds ({len(deltas)})')
            return
        arr = np.array(deltas)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f'\n{name} ablation delta (vs baseline) across {len(arr)} seeds:')
        print(f'  Mean Δ: {mean*100:+.2f}%')
        print(f'  Std Δ:  {std*100:.2f}%')
        print(f'  Range:  [{arr.min()*100:+.1f}%, {arr.max()*100:+.1f}%]')
        if std > 0 and abs(mean) > 2 * std:
            print(f'  ✅ VERDICT: REAL reversal (|mean| {abs(mean)*100:.1f}% > 2σ {2*std*100:.1f}%)')
            print(f'     → Ablation improvement is robust, NOT noise. Finding defensible.')
        else:
            print(f'  ⚠ VERDICT: WITHIN NOISE (|mean| {abs(mean)*100:.1f}% <= 2σ {2*std*100:.1f}%)')
            print(f'     → Reversal may be noise. Cannot strongly claim "paper Fig.4 wrong".')

    verdict(deltas_cl, 'no_cl')
    verdict(deltas_hgt, 'no_hgt')

    out = {
        'config': 'exist_weight=0.1',
        'seeds': seeds,
        'data': data,
        'delta_no_cl': deltas_cl,
        'delta_no_hgt': deltas_hgt,
    }
    with open('results/multiseed_ablation_verify.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: results/multiseed_ablation_verify.json')


if __name__ == '__main__':
    main()
