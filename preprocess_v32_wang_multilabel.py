"""Plan G-1: Multi-label version của preprocess_v32_wang.py.

Khác biệt key: KHÔNG collapse multi-label về single-label per cell. Mỗi (mi, dis)
có thể appear ở multiple rows trong CSV nếu có multiple types. Loss BCE sẽ handle.

Discovery (Pre-G verification): Sum per-type = 16,341 triplets nhưng unique
(mi, dis) pairs chỉ 12,534 → 23.3% signal mất khi collapse về single-label.
"""
import os
import sys
import csv
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines', 'TDRC'))
from data import GetData

TDRC_DATA = 'baselines/TDRC/data_v32/HMDD3.2_processed'
OUT_DIR = 'v3.2_wang_multilabel'

TDRC_TYPE_FILES = {
    'circu.csv':   1,
    'epic.csv':    2,
    'target.csv':  3,
    'genetic.csv': 4,
    'tissue.csv':  5,
}


def read_binary_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32)


def main():
    print('=' * 60)
    print('[v3.2_wang_multilabel] Multi-label preserved version')
    print('=' * 60)

    mi_name_df = pd.read_csv(os.path.join(TDRC_DATA, 'mi_name.csv'), index_col=0)
    di_name_df = pd.read_csv(os.path.join(TDRC_DATA, 'di_name.csv'), index_col=0)
    n_mi = len(mi_name_df)
    n_dis = len(di_name_df)
    print(f'miRNAs: {n_mi}, Diseases: {n_dis}')

    # Load 5 type matrices
    type_matrices = {}
    for fname, type_id in TDRC_TYPE_FILES.items():
        path = os.path.join(TDRC_DATA, fname)
        mat = read_binary_matrix(path)
        type_matrices[type_id] = mat

    # ===== MULTI-LABEL: KHÔNG collapse cell =====
    # Build list-of-triplets thay vì matrix [n_mi, n_dis] single value
    print('\n[v3.2_wang_multilabel] Building list-of-triplets (NO collapse)...')
    triplets = []
    for type_id, mat in type_matrices.items():
        positions = np.argwhere(mat > 0)  # shape (k, 2)
        for i, j in positions:
            triplets.append((int(i), int(j), type_id))

    print(f'  Total triplets: {len(triplets):,}')

    # Build binary association matrix (for similarity + view construction)
    binary_assoc = np.zeros((n_mi, n_dis), dtype=np.float32)
    for i, j, _ in triplets:
        binary_assoc[i, j] = 1.0
    n_unique_pairs = int(binary_assoc.sum())
    print(f'  Unique (mi, dis) pairs: {n_unique_pairs:,}')
    print(f'  Multi-label ratio: {len(triplets) / n_unique_pairs:.2f}x')
    print(f'  Signal recovered: {(len(triplets) - n_unique_pairs) / len(triplets) * 100:.1f}%')

    # Build multi-label target tensor [n_mi, n_dis, 5] (one-hot multi)
    target_multilabel = np.zeros((n_mi, n_dis, 5), dtype=np.float32)
    for i, j, t in triplets:
        target_multilabel[i, j, t - 1] = 1.0  # type 1..5 → channel 0..4

    # Save multi-label target tensor
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, 'target_multilabel.npy'), target_multilabel)
    print(f'  Saved multi-label target tensor: {target_multilabel.shape}')

    # Wang MeSH disease similarity (copy từ TDRC)
    print('\n[v3.2_wang_multilabel] Loading Wang MeSH disease similarity...')
    dis_sim = pd.read_csv(os.path.join(TDRC_DATA, 'Dis_sim.csv'), index_col=0).values.astype(np.float32)
    dis_sim = (dis_sim + dis_sim.T) / 2.0
    np.fill_diagonal(dis_sim, 1.0)
    print(f'  Disease sim: {dis_sim.shape}, range [{dis_sim.min():.4f}, {dis_sim.max():.4f}]')

    # miRNA functional similarity (Wang method, vectorized từ TDRC)
    print('\n[v3.2_wang_multilabel] Computing miRNA functional similarity...')
    t0 = time.time()
    tdrc_data = GetData(root='baselines/TDRC/data_v32')
    mi_func_sim = tdrc_data.get_functional_sim(binary_assoc).astype(np.float32)
    np.fill_diagonal(mi_func_sim, 1.0)
    print(f'  Done in {time.time() - t0:.1f}s')

    # Save outputs
    print(f'\n[v3.2_wang_multilabel] Saving to {OUT_DIR}/...')
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM1.txt'), dis_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM2.txt'), dis_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_FSM.txt'), mi_func_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_GSM.txt'), mi_func_sim, fmt='%.6f', delimiter=' ')

    # CSV: cho phép duplicate (mi, dis) với types khác nhau
    csv_path = os.path.join(OUT_DIR, 'multi_all_mirna_disease_pairs_without_negative.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, t in triplets:
            writer.writerow([i + 1, j + 1, t])  # 1-based
    print(f'  Saved triplet CSV: {len(triplets):,} rows (duplicates allowed)')

    # Name mappings
    pd.DataFrame({'idx': range(1, n_mi + 1),
                  'name': mi_name_df.iloc[:, 0].tolist()}).to_excel(
        os.path.join(OUT_DIR, 'miRNA name.xlsx'), index=False, header=False)
    pd.DataFrame({'idx': range(1, n_dis + 1),
                  'name': di_name_df.iloc[:, 0].tolist()}).to_excel(
        os.path.join(OUT_DIR, 'disease name.xlsx'), index=False, header=False)

    print(f'\n[v3.2_wang_multilabel] DONE.')
    print(f'  Output: {OUT_DIR}/')
    print(f'  Triplets: {len(triplets):,}')
    print(f'  Unique pairs: {n_unique_pairs:,}')
    print(f'  Multi-label channels (5): saved as target_multilabel.npy')


if __name__ == '__main__':
    main()
