#!/usr/bin/env python3
"""summarize_stats.py — P9 (docs/review/02-improvement-proposals.md): đánh giá thống kê trung thực.

Gộp per-fold metric từ NHIỀU điều kiện (vd. leaky vs leakage-free, method A vs B) và báo cáo:
  - mô tả: n, mean, std;
  - khoảng tin cậy bootstrap 95% (percentile) cho mean;
  - so sánh THEO CẶP (paired) per-fold giữa 2 điều kiện: paired t-test + Wilcoxon signed-rank,
    kèm hiệu chỉnh đa so sánh Holm khi có nhiều metric/nhiều cặp;
  - "gap" = mean(A) - mean(B) (vd. leakage gap = leaky - leakage_free).

KHÔNG suy diễn sinh học/lâm sàng — chỉ chỉ số tính toán.

Nguồn dữ liệu (chọn 1):
  A) --logs L1 L2 ... [--labels A B ...]
     Parse dòng "Fold k completed - AUC: x, Top-1 F1: y" (main_experiments_hetero1.py in ra).
     Mỗi log = 1 điều kiện; per-fold AUC & Top-1 F1 rút ra tự động.
  B) --json results.json
     Schema: {"label1": {"auc": [..per-fold..], "top1_f1": [..]}, "label2": {...}, ...}

Ví dụ:
  # Leakage gap: chạy 1 lần leaky + 1 lần leakage-free (cùng seed), rồi so sánh
  python summarize_stats.py --logs logs/leaky_s42.log logs/leakfree_s42.log \
         --labels leaky leakage_free --out results/leakage_gap_s42.json

  # Nhiều seed: nối per-fold của nhiều log cho mỗi điều kiện bằng cách lặp --logs theo nhóm
  python summarize_stats.py --json results/my_perfold.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    from parse_metrics import _read_log  # encoding-safe reader (UTF-8/UTF-16 BOM)
except Exception:
    def _read_log(p):
        return Path(p).read_text(encoding='utf-8', errors='replace')

# "Fold 1 completed - AUC: 0.9821, Top-1 F1: 0.6614"
_FOLD_RE = re.compile(
    r"Fold\s+(\d+)\s+completed\s*-\s*AUC:\s*([0-9.]+),\s*Top-1\s*F1:\s*([0-9.]+)")

_BOOT_SEED = 20260712  # cố định để bootstrap tái lập (không dùng Date/random động)


def parse_log_perfold(log_path):
    """Trả về dict {'auc': [...], 'top1_f1': [...]} theo thứ tự fold."""
    text = _read_log(Path(log_path))
    auc, top1 = [], []
    for m in _FOLD_RE.finditer(text):
        auc.append(float(m.group(2)))
        top1.append(float(m.group(3)))
    return {'auc': auc, 'top1_f1': top1}


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    """Percentile bootstrap CI cho mean. Trả (lo, hi)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(_BOOT_SEED)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def describe(values):
    v = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(v)
    return {
        'n': int(len(v)),
        'mean': float(v.mean()) if len(v) else float('nan'),
        'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        'ci95_boot': [lo, hi],
        'values': [float(x) for x in v],
    }


def paired_tests(a, b):
    """Paired tests trên per-fold (a, b phải cùng độ dài, ghép theo fold)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < len(a) or n < len(b):
        print(f"  [WARN] số fold khác nhau ({len(a)} vs {len(b)}) — cắt về {n} để ghép cặp.")
    a, b = a[:n], b[:n]
    diff = a - b
    out = {
        'n_pairs': int(n),
        'mean_diff': float(diff.mean()) if n else float('nan'),
        'ci95_boot_diff': list(bootstrap_ci(diff)),
    }
    if _HAVE_SCIPY and n >= 2:
        try:
            t_stat, t_p = _sps.ttest_rel(a, b)
            out['paired_t'] = {'stat': float(t_stat), 'p': float(t_p)}
        except Exception as e:
            out['paired_t'] = {'error': str(e)}
        try:
            if np.any(diff != 0):
                w_stat, w_p = _sps.wilcoxon(a, b)
                out['wilcoxon'] = {'stat': float(w_stat), 'p': float(w_p)}
            else:
                out['wilcoxon'] = {'note': 'all diffs zero'}
        except Exception as e:
            out['wilcoxon'] = {'error': str(e)}
    else:
        out['note'] = 'scipy không có hoặc n<2 → bỏ p-value'
    return out


def holm_correct(pvals):
    """Holm–Bonferroni. Trả list p đã hiệu chỉnh theo thứ tự gốc."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description="P9 honest stats: bootstrap CI + paired tests.")
    ap.add_argument('--logs', nargs='+', help='Log files, mỗi file = 1 điều kiện.')
    ap.add_argument('--labels', nargs='+', help='Nhãn cho mỗi log/điều kiện.')
    ap.add_argument('--json', type=str, help='JSON {label:{metric:[vals]}}.')
    ap.add_argument('--metrics', nargs='+', default=['auc', 'top1_f1'],
                    help='Metric để phân tích (mặc định: auc top1_f1).')
    ap.add_argument('--out', type=str, default='', help='Ghi kết quả ra JSON.')
    args = ap.parse_args()

    conditions = {}  # label -> {metric: [per-fold vals]}

    if args.json:
        with open(args.json) as f:
            conditions = json.load(f)
    elif args.logs:
        labels = args.labels or [Path(p).stem for p in args.logs]
        if len(labels) != len(args.logs):
            ap.error("số --labels phải khớp số --logs")
        # Nhãn LẶP LẠI → GỘP per-fold (vd gộp nhiều seed vào 1 điều kiện 'leaky').
        for lbl, log in zip(labels, args.logs):
            d = parse_log_perfold(log)
            if lbl in conditions:
                for k, v in d.items():
                    conditions[lbl].setdefault(k, []).extend(v)
            else:
                conditions[lbl] = d
    else:
        ap.error("cần --logs hoặc --json")

    report = {'conditions': {}, 'comparisons': {}}

    print("=" * 78)
    print("P9 — THỐNG KÊ TRUNG THỰC (bootstrap CI + paired tests)")
    print("=" * 78)

    for lbl, data in conditions.items():
        report['conditions'][lbl] = {}
        print(f"\n### Điều kiện: {lbl}")
        for met in args.metrics:
            vals = data.get(met, [])
            if not vals:
                print(f"  {met:10s}: (không có dữ liệu)")
                continue
            d = describe(vals)
            report['conditions'][lbl][met] = d
            print(f"  {met:10s}: n={d['n']}  mean={d['mean']:.4f}  std={d['std']:.4f}  "
                  f"CI95[{d['ci95_boot'][0]:.4f}, {d['ci95_boot'][1]:.4f}]")

    labels = list(conditions.keys())
    if len(labels) >= 2:
        # So sánh từng cặp; gom p-value để hiệu chỉnh Holm.
        pairs = [(labels[i], labels[j])
                 for i in range(len(labels)) for j in range(i + 1, len(labels))]
        pvec, pkeys = [], []
        for (A, B) in pairs:
            for met in args.metrics:
                a = conditions[A].get(met, [])
                b = conditions[B].get(met, [])
                if not a or not b:
                    continue
                res = paired_tests(a, b)
                key = f"{A}_vs_{B}::{met}"
                report['comparisons'][key] = res
                # gap A - B (vd leakage gap = leaky - leakage_free)
                print(f"\n### {A} vs {B} — {met}")
                print(f"  gap(mean {A}-{B}) = {res['mean_diff']:+.4f}  "
                      f"CI95[{res['ci95_boot_diff'][0]:+.4f}, {res['ci95_boot_diff'][1]:+.4f}]")
                if 'paired_t' in res and 'p' in res['paired_t']:
                    print(f"  paired t: t={res['paired_t']['stat']:+.3f}  p={res['paired_t']['p']:.4g}")
                    pvec.append(res['paired_t']['p'])
                    pkeys.append(key)
                if 'wilcoxon' in res and 'p' in res['wilcoxon']:
                    print(f"  Wilcoxon: W={res['wilcoxon']['stat']:.3f}  p={res['wilcoxon']['p']:.4g}")
        # Holm trên các paired-t p-value
        if pvec:
            adj = holm_correct(pvec)
            print("\n### Hiệu chỉnh đa so sánh (Holm, trên paired-t):")
            for k, p, pa in zip(pkeys, pvec, adj):
                sig = "  *SIG(α=.05)" if pa < 0.05 else ""
                report['comparisons'][k]['paired_t']['p_holm'] = float(pa)
                print(f"  {k:40s} p={p:.4g} → Holm={pa:.4g}{sig}")

    print("\n" + "=" * 78)
    print("Lưu ý: KHÔNG suy diễn sinh học/lâm sàng. Ưu tiên AUC/AUPR/Top-1 F1 (không phụ thuộc")
    print("ngưỡng); F1 nhị phân trong pipeline tune ngưỡng trên test (audit F4) nên đọc thận trọng.")
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Đã ghi {args.out}")


if __name__ == '__main__':
    main()
