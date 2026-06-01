"""Build v3.2_wang/ — DHGCMDA-format dataset từ TDRC's v3.2 preprocessing với Wang's MeSH semantic similarity.

Plan E-1a: tận dụng TDRC's `Dis_sim.csv` (447×447, Wang MeSH) + compute miRNA functional
similarity bằng Wang's method (TDRC vectorized) → replace GIP-only approach từ preprocess_v32.py.

Outputs (lưu vào v3.2_wang/):
- D_SSM1.txt: Wang's MeSH disease semantic similarity (447×447)
- D_SSM2.txt: same as SSM1 (DHGCMDA dual-view dùng 2 disease views, paper distinguishes SSM1 method 1 vs SSM2 method 2 — tạm reuse same)
- M_FSM.txt: Wang's miRNA functional similarity computed từ disease sim + associations
- M_GSM.txt: same as FSM (placeholder cho miRNA View 1 — paper distinguishes sequence vs functional, reuse functional)
- multi_all_mirna_disease_pairs_without_negative.csv: (mi_idx, dis_idx, type 1-5), 1-based
- miRNA name.xlsx, disease name.xlsx
"""
import os
import sys
import csv
import time
import numpy as np
import pandas as pd

# Import TDRC's vectorized get_functional_sim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines', 'TDRC'))
from data import GetData

TDRC_DATA = 'baselines/TDRC/data_v32/HMDD3.2_processed'
OUT_DIR = 'v3.2_wang'

# TDRC type order: ['target', 'circu', 'epic', 'genetic', 'tissue'] (theo data.py:16)
# Paper DHGCMDA type order: 1=Circulation 2=Epigenetics 3=Target 4=Genetics 5=Tissue
# Mapping TDRC index → DHGCMDA type_id
TDRC_TYPE_FILES = {
    'circu.csv':   1,  # circulation
    'epic.csv':    2,  # epigenetics
    'target.csv':  3,  # target
    'genetic.csv': 4,  # genetics
    'tissue.csv':  5,  # tissue
}


def read_binary_matrix(path):
    """Đọc CSV binary matrix với header row + index col (TDRC format)."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32)


def main():
    print('=' * 60)
    print('[v3.2_wang] Building DHGCMDA dataset từ TDRC Wang preprocessing')
    print('=' * 60)

    # 1. Load name mappings
    mi_name_df = pd.read_csv(os.path.join(TDRC_DATA, 'mi_name.csv'), index_col=0)
    di_name_df = pd.read_csv(os.path.join(TDRC_DATA, 'di_name.csv'), index_col=0)
    n_mi = len(mi_name_df)
    n_dis = len(di_name_df)
    print(f'[v3.2_wang] miRNAs: {n_mi}')
    print(f'[v3.2_wang] Diseases: {n_dis}')

    # 2. Load 5 type matrices, build multi-type association matrix
    print('\n[v3.2_wang] Loading 5 type matrices...')
    type_matrices = {}
    for fname, type_id in TDRC_TYPE_FILES.items():
        path = os.path.join(TDRC_DATA, fname)
        mat = read_binary_matrix(path)
        assert mat.shape == (n_mi, n_dis), f'{fname} shape {mat.shape} != ({n_mi}, {n_dis})'
        type_matrices[type_id] = mat
        print(f'  Type {type_id} ({fname}): {int(mat.sum()):,} associations')

    # Build multi-type association matrix: cell = type_id (priority order 1→5)
    print('\n[v3.2_wang] Building multi-type association matrix...')
    assoc_matrix = np.zeros((n_mi, n_dis), dtype=np.int8)
    for type_id in [5, 4, 3, 2, 1]:  # đặt later types first, lower types overwrite (priority lower wins)
        mask = type_matrices[type_id] > 0
        assoc_matrix[mask] = type_id

    # Stats
    binary_assoc = (assoc_matrix > 0).astype(np.float32)
    total_assoc = int(binary_assoc.sum())
    print(f'  Binary associations: {total_assoc:,}')
    for t in range(1, 6):
        n = int((assoc_matrix == t).sum())
        print(f'  Type {t}: {n:,} ({n / (n_mi * n_dis) * 100:.3f}%)')

    # 3. Disease semantic similarity (Wang MeSH) — copy từ TDRC
    print('\n[v3.2_wang] Loading Wang MeSH disease semantic similarity...')
    dis_sim_df = pd.read_csv(os.path.join(TDRC_DATA, 'Dis_sim.csv'), index_col=0)
    dis_sim = dis_sim_df.values.astype(np.float32)
    # Make symmetric + diagonal=1 cho cosine-like
    dis_sim = (dis_sim + dis_sim.T) / 2.0
    np.fill_diagonal(dis_sim, 1.0)
    print(f'  Disease similarity: {dis_sim.shape}, range [{dis_sim.min():.4f}, {dis_sim.max():.4f}]')
    nonzero_frac = (dis_sim > 0).sum() / dis_sim.size
    print(f'  Nonzero fraction: {nonzero_frac * 100:.1f}%')

    # 4. miRNA functional similarity (Wang's method dùng disease_sim + associations)
    print('\n[v3.2_wang] Computing miRNA functional similarity (Wang method, vectorized)...')
    print('  (Sử dụng TDRC GetData.get_functional_sim — vectorized, ~20 phút CPU)')
    t0 = time.time()
    # Khởi tạo TDRC GetData để dùng method get_functional_sim
    tdrc_data = GetData(root='baselines/TDRC/data_v32')
    # tdrc_data đã có dis_sim và type_tensor, nhưng ta dùng phương thức để compute cho mạng association mới
    mi_func_sim = tdrc_data.get_functional_sim(binary_assoc)
    mi_func_sim = mi_func_sim.astype(np.float32)
    np.fill_diagonal(mi_func_sim, 1.0)
    print(f'  Done in {time.time() - t0:.1f}s')
    print(f'  miRNA functional sim: {mi_func_sim.shape}, range [{mi_func_sim.min():.4f}, {mi_func_sim.max():.4f}]')

    # 5. Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'\n[v3.2_wang] Saving to {OUT_DIR}/...')

    # Similarity matrices (space-separated txt — DHGCMDA format)
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM1.txt'), dis_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'D_SSM2.txt'), dis_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_FSM.txt'), mi_func_sim, fmt='%.6f', delimiter=' ')
    np.savetxt(os.path.join(OUT_DIR, 'M_GSM.txt'), mi_func_sim, fmt='%.6f', delimiter=' ')
    print('  Saved 4 similarity files')

    # Association CSV
    csv_rows = []
    for i in range(n_mi):
        for j in range(n_dis):
            t = int(assoc_matrix[i, j])
            if t > 0:
                csv_rows.append([i + 1, j + 1, t])  # 1-based indices
    with open(os.path.join(OUT_DIR, 'multi_all_mirna_disease_pairs_without_negative.csv'),
              'w', newline='') as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)
    print(f'  Saved association CSV: {len(csv_rows):,} rows')

    # Name mappings (1-based index, name)
    pd.DataFrame({'idx': range(1, n_mi + 1),
                  'name': mi_name_df.iloc[:, 0].tolist()}).to_excel(
        os.path.join(OUT_DIR, 'miRNA name.xlsx'), index=False, header=False)
    pd.DataFrame({'idx': range(1, n_dis + 1),
                  'name': di_name_df.iloc[:, 0].tolist()}).to_excel(
        os.path.join(OUT_DIR, 'disease name.xlsx'), index=False, header=False)
    print('  Saved name mappings xlsx')

    print(f'\n[v3.2_wang] DONE.')
    print(f'  Output: {OUT_DIR}/')
    print(f'  Dimensions: {n_mi} miRNAs × {n_dis} diseases')
    print(f'  Total associations: {total_assoc:,}')
    print(f'  Types: 5 (Circulation, Epigenetics, Target, Genetics, Tissue)')


if __name__ == '__main__':
    main()
