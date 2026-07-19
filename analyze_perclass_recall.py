#!/usr/bin/env python3
"""Per-class top-1 recall/precision từ dump npz (--dump_scores).

Trả lời trực tiếp câu hỏi B3: mapping-fix có phục hồi recall của Tissue (type-5) không?
Bypass Calculate_Metrics (metric-bug). Chỉ đọc raw scores held-out CV.
"""
import argparse
import glob
import os
import numpy as np

TYPE_NAMES_5 = ["circulation", "epigenetics", "target", "genetics", "tissue"]
TYPE_NAMES_4 = ["circulation", "epigenetics", "target", "genetics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "fold*.npz")))
    if not files:
        print(f"❌ không thấy fold*.npz trong {args.dump_dir}")
        return

    all_true, all_pred = [], []
    K = None
    for f in files:
        d = np.load(f)
        probs = d["type_probs"]          # [N, K]
        true = d["true_type"].astype(int)  # 1..K
        K = probs.shape[1]
        pred = probs.argmax(axis=1)      # 0..K-1
        all_true.append(true - 1)        # -> 0..K-1
        all_pred.append(pred)
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    names = TYPE_NAMES_5 if K == 5 else TYPE_NAMES_4 if K == 4 else [f"t{i}" for i in range(K)]

    print(f"\n===== PER-CLASS TOP-1 {('['+args.tag+']') if args.tag else ''}  (K={K}, N={len(yt)}, {len(files)} folds) =====")
    print(f"{'type':<14}{'n_true':>8}{'recall':>9}{'prec':>9}{'f1':>9}")
    recalls, f1s = [], []
    for c in range(K):
        tp = int(np.sum((yt == c) & (yp == c)))
        n_true = int(np.sum(yt == c))
        n_pred = int(np.sum(yp == c))
        rec = tp / n_true if n_true else 0.0
        prec = tp / n_pred if n_pred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec); f1s.append(f1)
        flag = "  <-- Tissue" if (K == 5 and c == 4) else ""
        print(f"{names[c]:<14}{n_true:>8}{rec:>9.3f}{prec:>9.3f}{f1:>9.3f}{flag}")
    acc = float(np.mean(yt == yp))
    print(f"{'-'*49}")
    print(f"accuracy={acc:.4f}  macro-recall={np.mean(recalls):.4f}  macro-F1={np.mean(f1s):.4f}")
    print(f"minority classes recall: " +
          ", ".join(f"{names[c]}={recalls[c]:.3f}" for c in range(K)))


if __name__ == "__main__":
    main()
