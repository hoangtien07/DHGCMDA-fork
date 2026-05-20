"""Preprocess HMDD v3.2 từ raw cuilab data → format DHGCMDA tương đương v2.0.

Pragmatic approach: GIP (Gaussian Interaction Profile) similarity duy nhất
(không có sẵn MeSH semantic / miRNA functional / gene-based similarity). Dùng GIP
cho cả 4 view → degenerate dual-view nhưng vẫn chạy được DHGCMDA pipeline.

Outputs (lưu vào v3.2_processed/):
- D_SSM1.txt: dis_gip similarity (proxy cho semantic)
- D_SSM2.txt: dis_gip similarity (proxy cho gene-based) — same as SSM1
- M_FSM.txt:  mi_gip similarity (proxy cho functional)
- M_GSM.txt:  mi_gip similarity (proxy cho sequence) — same as FSM
- multi_all_mirna_disease_pairs_without_negative.csv: (mi_idx, dis_idx, type 1-5)
- miRNA name.xls: index → name mapping
- disease name.xls: index → name mapping
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

RAW_DIR = 'HMDD_data/MDAv3.2'
OUT_DIR = 'v3.2_processed'

# Type mapping cho v3.2 (5 types theo paper Table 2)
TYPE_FILES = {
    1: 'v3_circulation.txt',
    2: 'v3_epigenetics.txt',
    3: 'v3_target.txt',
    4: 'v3_genetics.txt',
    5: 'v3_tissue.txt',
}
TYPE_NAMES = {1: 'circulation', 2: 'epigenetics', 3: 'target', 4: 'genetics', 5: 'tissue'}

MIN_ASSOC_PER_MI = 2
MIN_ASSOC_PER_DIS = 2


def load_type_file(path, type_id):
    """Load 1 type file → list of (mi_name, dis_name, type_id)."""
    df = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8')
    df = df[['mir', 'disease']].dropna()
    df['mir'] = df['mir'].str.strip().str.lower()
    df['disease'] = df['disease'].str.strip()  # giữ case cho disease (matching paper Table 5/6)
    df = df.drop_duplicates()
    return [(m, d, type_id) for m, d in zip(df['mir'], df['disease'])]


def main():
    print('[v3.2] Loading 5 type files...')
    all_triplets = []
    for tid, fname in TYPE_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        triplets = load_type_file(path, tid)
        print(f'  {fname}: {len(triplets):,} triplets')
        all_triplets.extend(triplets)

    print(f'\n[v3.2] Total raw triplets: {len(all_triplets):,}')

    # Dedupe — nếu cùng (mi, dis) có nhiều type, chọn type FIRST (theo order paper: circulation prior)
    # Hoặc tốt hơn: giữ type majority hoặc tất cả.
    # Paper Table 2 cho v3.2: 1155+403+2293+3997+3900 = 11748 → distinct (mi,dis,type) chứ không phải distinct (mi,dis).
    # Vậy ta giữ tất cả nhưng dedupe duplicates trong cùng type.
    triplets_unique = list(set(all_triplets))
    print(f'[v3.2] Unique (mi, dis, type) triplets: {len(triplets_unique):,}')

    # Đếm miRNA/disease counts
    mi_counts = defaultdict(int)
    dis_counts = defaultdict(int)
    for m, d, t in triplets_unique:
        mi_counts[m] += 1
        dis_counts[d] += 1

    # Filter ≥2 (theo paper)
    kept_mis = {m for m, c in mi_counts.items() if c >= MIN_ASSOC_PER_MI}
    kept_dis = {d for d, c in dis_counts.items() if c >= MIN_ASSOC_PER_DIS}
    print(f'[v3.2] Filtered miRNAs (≥{MIN_ASSOC_PER_MI} assoc): {len(kept_mis):,} / {len(mi_counts):,}')
    print(f'[v3.2] Filtered diseases (≥{MIN_ASSOC_PER_DIS} assoc): {len(kept_dis):,} / {len(dis_counts):,}')

    triplets_filt = [(m, d, t) for m, d, t in triplets_unique if m in kept_mis and d in kept_dis]
    print(f'[v3.2] Final triplets after filter: {len(triplets_filt):,}')

    # Build sorted miRNA/disease index
    mi_list = sorted(kept_mis)
    dis_list = sorted(kept_dis)
    mi_to_idx = {m: i for i, m in enumerate(mi_list)}
    dis_to_idx = {d: i for i, d in enumerate(dis_list)}

    n_mi = len(mi_list)
    n_dis = len(dis_list)
    print(f'\n[v3.2] Matrix shape: [{n_mi}, {n_dis}]')

    # Build association matrix
    # Nếu 1 (mi,dis) có nhiều type, chọn type có ít sample nhất trong overall (cho minority class)
    # Hoặc đơn giản: chọn type đầu tiên gặp.
    # Paper: 11748 distinct triplets có 5 types. Nếu collapse về matrix duy nhất với 1 type/cell, sẽ mất info.
    # Approach: build matrix với value = type, nếu duplicate thì giữ LAST (arbitrary). DHGCMDA chấp nhận single type per cell.
    assoc_matrix = np.zeros((n_mi, n_dis), dtype=np.int8)
    type_counts = defaultdict(int)
    for m, d, t in triplets_filt:
        mi_idx = mi_to_idx[m]
        dis_idx = dis_to_idx[d]
        assoc_matrix[mi_idx, dis_idx] = t
        type_counts[t] += 1

    print(f'\n[v3.2] Final association distribution:')
    for t in sorted(type_counts.keys()):
        n = type_counts[t]
        print(f'  Type {t} ({TYPE_NAMES[t]}): {n:,} ({n / (n_mi*n_dis) * 100:.3f}%)')
    total_assoc = int((assoc_matrix > 0).sum())
    print(f'  Total non-zero: {total_assoc:,} ({total_assoc / (n_mi*n_dis) * 100:.2f}%)')

    # Build GIP similarity (using existing helpers from prepareData.py)
    print(f'\n[v3.2] Computing GIP similarity matrices...')
    binary_matrix = (assoc_matrix > 0).astype(np.float32)

    # GIP_M: shape [n_mi, n_mi], từ row vectors
    A = binary_matrix
    A_sq = A * A
    row_sum = A_sq.sum(axis=1, keepdims=True)
    diff_mi = row_sum + row_sum.T - 2 * np.dot(A, A.T)
    rm = n_dis * 1.0 / max(A_sq.sum(), 1e-7)
    GIP_M = np.exp(-rm * diff_mi).astype(np.float32)

    # GIP_D: shape [n_dis, n_dis], từ col vectors
    AT = binary_matrix.T
    AT_sq = AT * AT
    col_sum = AT_sq.sum(axis=1, keepdims=True)
    diff_dis = col_sum + col_sum.T - 2 * np.dot(AT, AT.T)
    rd = n_mi * 1.0 / max(AT_sq.sum(), 1e-7)
    GIP_D = np.exp(-rd * diff_dis).astype(np.float32)

    print(f'  GIP_M shape: {GIP_M.shape}, range [{GIP_M.min():.4f}, {GIP_M.max():.4f}]')
    print(f'  GIP_D shape: {GIP_D.shape}, range [{GIP_D.min():.4f}, {GIP_D.max():.4f}]')

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'\n[v3.2] Saving outputs to {OUT_DIR}/...')

    # Similarity matrices — space-separated txt giống v2.0 format
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM1.txt'), GIP_D, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM2.txt'), GIP_D, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_FSM.txt'),  GIP_M, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_GSM.txt'),  GIP_M, fmt='%.6f', delimiter=' ')
    print(f'  Saved D_SSM1/SSM2.txt, M_FSM/GSM.txt')

    # Association CSV: mi_idx (1-based), dis_idx (1-based), type (1-5)
    csv_rows = []
    for i in range(n_mi):
        for j in range(n_dis):
            t = int(assoc_matrix[i, j])
            if t > 0:
                csv_rows.append([i + 1, j + 1, t])
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(os.path.join(OUT_DIR, 'multi_all_mirna_disease_pairs_without_negative.csv'),
                  index=False, header=False)
    print(f'  Saved multi_all_mirna_disease_pairs_without_negative.csv ({len(csv_rows):,} rows)')

    # Name mappings
    pd.DataFrame({'idx': range(1, n_mi + 1), 'name': mi_list}).to_excel(
        os.path.join(OUT_DIR, 'miRNA name.xlsx'), index=False, header=False)
    pd.DataFrame({'idx': range(1, n_dis + 1), 'name': dis_list}).to_excel(
        os.path.join(OUT_DIR, 'disease name.xlsx'), index=False, header=False)
    print(f'  Saved miRNA name.xlsx + disease name.xlsx')

    # Sanity stats
    print(f'\n[v3.2] DONE.')
    print(f'  miRNAs: {n_mi} (paper: 411)')
    print(f'  Diseases: {n_dis} (paper: 271)')
    print(f'  Associations: {total_assoc} (paper: 11,748)')


if __name__ == '__main__':
    main()
