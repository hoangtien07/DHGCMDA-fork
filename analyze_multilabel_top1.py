"""Task D (P3): metric multi-label "predicted type ∈ known types" từ dump npz (--dump_scores).

Bypass Calculate_Metrics (KHÔNG sửa). Trả lời câu hỏi P3: multilabel_bce có tăng khả năng đoán
đúng MỘT trong các type đã biết của cặp đa-type không, mà không giảm trên cặp single-type?

Metric:
  - hit@1 = (argmax type_probs) ∈ tập type ĐÃ BIẾT của cặp (từ multilabel_pairs_meta.npz).
  - Báo cáo tách: MULTI-type subset (161 cặp) vs SINGLE-type subset, và overall.
  - So thêm strict-top1 = argmax == collapsed-true_type (giống metric cũ) để đối chiếu.

Dump per-fold: fold*.npz {mirna_idx, disease_idx, type_probs[N,K], true_type[N]} (test positives).
Union các fold = toàn bộ positive (proper k-fold).
"""
import argparse
import glob
import os
import numpy as np


def load_meta(meta_path):
    m = np.load(meta_path)
    pair2types = {}
    for r in range(m['mi_idx'].shape[0]):
        ts = set(int(t) for t in m['type_set'][r] if t > 0)
        pair2types[(int(m['mi_idx'][r]), int(m['dis_idx'][r]))] = ts
    return pair2types


def eval_dump_dir(dump_dir, pair2types):
    files = sorted(glob.glob(os.path.join(dump_dir, 'fold*.npz')))
    if not files:
        return None
    rows = []  # (is_multi, hit_known, hit_strict)
    for f in files:
        d = np.load(f)
        probs = d['type_probs']          # [N, K]
        mi = d['mirna_idx'].astype(int)
        di = d['disease_idx'].astype(int)
        true_c = d['true_type'].astype(int)   # collapsed single-label (1..K)
        pred = probs.argmax(axis=1) + 1       # 1..K
        for n in range(probs.shape[0]):
            known = pair2types.get((int(mi[n]), int(di[n])), None)
            if known is None:
                known = {int(true_c[n])}       # fallback: collapsed label
            is_multi = len(known) > 1
            rows.append((is_multi, int(pred[n] in known), int(pred[n] == true_c[n])))
    return np.array(rows, dtype=int)


def summarize(rows, tag):
    n = len(rows)
    multi = rows[rows[:, 0] == 1]
    single = rows[rows[:, 0] == 0]
    def acc(a, col):
        return float(a[:, col].mean()) if len(a) else float('nan')
    print(f"\n===== MULTI-LABEL TOP-1 [{tag}] (N={n}, multi={len(multi)}, single={len(single)}) =====")
    print(f"{'subset':<14}{'n':>6}{'hit@known':>12}{'strict-top1':>13}")
    print(f"{'overall':<14}{n:>6}{acc(rows,1):>12.4f}{acc(rows,2):>13.4f}")
    print(f"{'multi-type':<14}{len(multi):>6}{acc(multi,1):>12.4f}{acc(multi,2):>13.4f}")
    print(f"{'single-type':<14}{len(single):>6}{acc(single,1):>12.4f}{acc(single,2):>13.4f}")
    return {'n': n, 'overall_hit': acc(rows, 1), 'multi_hit': acc(multi, 1),
            'single_hit': acc(single, 1), 'overall_strict': acc(rows, 2),
            'multi_strict': acc(multi, 2), 'single_strict': acc(single, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir', required=True, help='thư mục chứa fold*.npz (1 seed), hoặc dùng nhiều lần')
    ap.add_argument('--meta', default='v2.0_495m383D/multilabel_pairs_meta.npz')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    pair2types = load_meta(args.meta)
    rows = eval_dump_dir(args.dump_dir, pair2types)
    if rows is None:
        print(f"❌ không thấy fold*.npz trong {args.dump_dir}")
        return
    summarize(rows, args.tag or os.path.basename(args.dump_dir.rstrip('/')))


if __name__ == '__main__':
    main()
