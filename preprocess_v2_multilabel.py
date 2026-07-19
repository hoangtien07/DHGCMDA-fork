"""Task D (P3): build multi-hot target tensor cho v2.0 (giữ tín hiệu multi-label, audit F2).

Mặc định pipeline (read_association_csv) đè "last-wins" → 161 cặp đa-type (20.4% dòng) mất nhãn phụ.
Script này đọc CSV GỐC, dựng target multi-hot [495, 383, 4] (KHÔNG collapse) → nạp qua
--multilabel_target_path cho --loss_mode multilabel_bce. Additive, không đụng source/model.

Output:
  v2.0_495m383D/target_multilabel_v2.npy      — [495,383,4] float multi-hot (channel k-1 = type k)
  v2.0_495m383D/multilabel_pairs_meta.npz     — mi_idx/dis_idx/n_types của 161 cặp đa-type (0-based)
                                                 + tất cả cặp positive → cho analyze_multilabel_top1.py
"""
import os
import numpy as np
import pandas as pd

CSV = 'v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv'
N_MI, N_DIS, N_TYPES = 495, 383, 4
OUT_TARGET = 'v2.0_495m383D/target_multilabel_v2.npy'
OUT_META = 'v2.0_495m383D/multilabel_pairs_meta.npz'


def main():
    df = pd.read_csv(CSV, header=None)
    mi = df.iloc[:, 0].values - 1     # 0-based
    dis = df.iloc[:, 1].values - 1
    typ = df.iloc[:, 2].values.astype(int)   # 1..4

    target = np.zeros((N_MI, N_DIS, N_TYPES), dtype=np.float32)
    from collections import defaultdict
    pair_types = defaultdict(set)
    for i, j, t in zip(mi, dis, typ):
        if 0 <= i < N_MI and 0 <= j < N_DIS and 1 <= t <= N_TYPES:
            target[i, j, t - 1] = 1.0
            pair_types[(int(i), int(j))].add(int(t))

    # Multi-hot sanity
    counts = target.sum(axis=-1)              # số type mỗi cặp
    n_pairs = int((counts > 0).sum())
    n_multi = int((counts > 1).sum())
    print(f"[P3-v2] rows={len(df)} unique_pairs={n_pairs} multi_type_pairs={n_multi} "
          f"({100*n_multi/n_pairs:.1f}%)")
    print(f"[P3-v2] per-type positive counts (multi-hot): "
          f"{{ {', '.join(f'{k+1}:{int(target[:,:,k].sum())}' for k in range(N_TYPES))} }}")

    np.save(OUT_TARGET, target)
    print(f"[P3-v2] saved target {target.shape} -> {OUT_TARGET}")

    # Meta cho analysis: mọi cặp positive + cờ multi-type + tập type (padded -1)
    all_pairs = sorted(pair_types.keys())
    mi_idx = np.array([p[0] for p in all_pairs], dtype=np.int64)
    dis_idx = np.array([p[1] for p in all_pairs], dtype=np.int64)
    n_types_per = np.array([len(pair_types[p]) for p in all_pairs], dtype=np.int64)
    max_t = max(n_types_per)
    type_set = -np.ones((len(all_pairs), max_t), dtype=np.int64)  # 1-based types, -1 pad
    for r, p in enumerate(all_pairs):
        for c, t in enumerate(sorted(pair_types[p])):
            type_set[r, c] = t
    np.savez(OUT_META, mi_idx=mi_idx, dis_idx=dis_idx,
             n_types=n_types_per, type_set=type_set)
    print(f"[P3-v2] saved meta ({len(all_pairs)} pairs, {n_multi} multi-type) -> {OUT_META}")


if __name__ == '__main__':
    main()
