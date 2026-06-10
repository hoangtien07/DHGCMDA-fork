"""
build_v32_dense.py — build a paper-density v3.2 dataset by filtering v3.2_wang.

WHY: paper v3.2 is 411x271 @ 10.5% density (curated, NOT public). Our v3.2_wang
(TDRC preprocessing) is 713x447 @ 3.9% density. Iterative min-assoc>=7 filter on
v3.2_wang yields 385x275 @ 10.3% — matches paper SHAPE+DENSITY almost exactly,
while REUSING the correct Wang MeSH similarity. Tests whether data density (not
model) drives the 0.33->0.86 gap.

Reuses v3.2_wang's similarity matrices by SLICING to the kept entities (no new
similarity computation). Output folder runs on UNMODIFIED model code via CLI
flags (--dataset v3.2_wang_dense --mi_num M --dis_num D --num_association_types 5).
"""
import os
import numpy as np
import pandas as pd

SRC = 'v3.2_wang'
DST = 'v3.2_wang_dense'
THR = 7

def iterative_filter(df, thr):
    cur = df.copy()
    for _ in range(30):
        mc = cur.groupby('mi').size(); dc = cur.groupby('dis').size()
        keepm = set(mc[mc >= thr].index); keepd = set(dc[dc >= thr].index)
        nxt = cur[cur.mi.isin(keepm) & cur.dis.isin(keepd)]
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur

def main():
    os.makedirs(DST, exist_ok=True)
    df = pd.read_csv(f'{SRC}/multi_all_mirna_disease_pairs_without_negative.csv',
                     header=None, names=['mi', 'dis', 'type'])
    n_mi_src = df.mi.max(); n_dis_src = df.dis.max()
    print(f"source v3.2_wang: {df.mi.nunique()} miRNA x {df.dis.nunique()} disease x {len(df)} assoc")

    kept = iterative_filter(df, THR)
    keep_mi = sorted(kept.mi.unique())      # original 1-based indices, sorted
    keep_dis = sorted(kept.dis.unique())
    M, D = len(keep_mi), len(keep_dis)
    dens = len(kept) / (M * D) * 100
    print(f"filtered (>= {THR}): {M} miRNA x {D} disease x {len(kept)} assoc, density {dens:.1f}%")
    print(f"paper target: 411 x 271 x 11748, density 10.5%")

    # remap old 1-based idx -> new 1-based idx
    mi_map = {old: new + 1 for new, old in enumerate(keep_mi)}
    dis_map = {old: new + 1 for new, old in enumerate(keep_dis)}

    # remap association csv
    out = kept.copy()
    out['mi'] = out['mi'].map(mi_map)
    out['dis'] = out['dis'].map(dis_map)
    out = out.sort_values(['mi', 'dis']).reset_index(drop=True)
    out.to_csv(f'{DST}/multi_all_mirna_disease_pairs_without_negative.csv',
               header=False, index=False)
    # verify per-type
    names = {1: 'Circ', 2: 'Epi', 3: 'Target', 4: 'Genetics', 5: 'Tissue'}
    print("per-type:", {names[t]: int(c) for t, c in out['type'].value_counts().sort_index().items()})

    # slice similarity matrices (rows/cols are 0-based for original 1-based idx-1)
    mi_rows = [m - 1 for m in keep_mi]      # 0-based
    dis_rows = [d - 1 for d in keep_dis]

    def slice_mat(fname, idx, expected_n):
        mat = np.loadtxt(f'{SRC}/{fname}')
        assert mat.shape[0] == expected_n, f"{fname}: got {mat.shape}, expected {expected_n}"
        sub = mat[np.ix_(idx, idx)]
        np.savetxt(f'{DST}/{fname}', sub, fmt='%.6f')
        print(f"  {fname}: {mat.shape} -> {sub.shape}")

    slice_mat('M_FSM.txt', mi_rows, n_mi_src)
    slice_mat('M_GSM.txt', mi_rows, n_mi_src)
    slice_mat('D_SSM1.txt', dis_rows, n_dis_src)
    slice_mat('D_SSM2.txt', dis_rows, n_dis_src)

    print(f"\nDONE. Run with:")
    print(f"  python run_v32_correct_metric.py --device cpu --dataset {DST} "
          f"--mi_num {M} --dis_num {D} --num_association_types 5 "
          f"--loss_mode two_head --exist_weight 0.1 --predictor_mode full_bilinear "
          f"--epoch 300 --validation 5")

if __name__ == '__main__':
    main()
