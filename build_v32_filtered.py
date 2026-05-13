"""Build v3.2 filtered association matrix — controlled experiment.

Goal: replace v2.0 associations with v3.2 associations, KEEP v2.0 entities + similarity.
This isolates the effect of dataset (associations only) on Fig.4 reproduce.

Input:
  HMDD_data/v3_alldata.txt — full v3.2 dataset
  v2.0_495m383D/miRNA name.xls, disease name.xls — v2.0 entity lists

Output:
  v3.2_filtered_495m383D/multi_all_mirna_disease_pairs_without_negative.csv
  v3.2_filtered_495m383D/<copies of v2.0 similarity files>

Subcategory → 4 paper types mapping:
  circulation_biomarker_* → 1 (circulation)
  epigenetics → 2 (epigenetics)
  target gene, therapeutic target → 3 (target)
  genetics_* → 4 (genetics)
  others (tissue, lncRNA, other, transcription factor): EXCLUDED
"""
import os
import shutil
import pandas as pd
import numpy as np
from collections import Counter

V32_FILE = 'HMDD_data/v3_alldata.txt'
V20_DIR = 'v2.0_495m383D'
OUT_DIR = 'v3.2_filtered_495m383D'

# Subcategory → 4-type mapping (1=circulation, 2=epigenetics, 3=target, 4=genetics)
CATEGORY_MAP = {}
# Circulation
for sub in ['circulation_biomarker_diagnosis_down', 'circulation_biomarker_diagnosis_ns',
            'circulation_biomarker_diagnosis_up', 'circulation_biomarker_prognosis_down',
            'circulation_biomarker_prognosis_ns', 'circulation_biomarker_prognosis_up']:
    CATEGORY_MAP[sub] = 1
# Epigenetics
CATEGORY_MAP['epigenetics'] = 2
# Target
CATEGORY_MAP['target gene'] = 3
CATEGORY_MAP['therapeutic target'] = 3
# Genetics
for sub in ['genetics_GWAS', 'genetics_knock down_promote', 'genetics_knock down_suppress',
            'genetics_overexpression_promote', 'genetics_overexpression_suppress']:
    CATEGORY_MAP[sub] = 4

EXCLUDED_CATS = ['tissue_expression_down', 'tissue_expression_ns', 'tissue_expression_up',
                 'lncRNA target', 'other', 'transcription factor target']


def load_v20_entities():
    """Load v2.0 miRNA + disease name lists with their indices."""
    mi_df = pd.read_excel(os.path.join(V20_DIR, 'miRNA name.xls'), header=None, engine='xlrd')
    dis_df = pd.read_excel(os.path.join(V20_DIR, 'disease name.xls'), header=None, engine='xlrd')

    mi_name_to_idx = {}
    for _, row in mi_df.iterrows():
        idx = int(row[0]) - 1  # 1-based → 0-based
        name = str(row[1]).strip().lower()
        mi_name_to_idx[name] = idx

    dis_name_to_idx = {}
    for _, row in dis_df.iterrows():
        idx = int(row[0]) - 1
        name = str(row[1]).strip().lower()
        dis_name_to_idx[name] = idx

    return mi_name_to_idx, dis_name_to_idx


def main():
    print('=' * 80)
    print('Build v3.2 filtered dataset — v3.2 associations + v2.0 entities')
    print('=' * 80)

    # Step 1: Load v2.0 entity lists
    mi_idx, dis_idx = load_v20_entities()
    print(f'\nv2.0 entities: {len(mi_idx)} miRNAs, {len(dis_idx)} diseases')

    # Step 2: Load v3.2 raw data
    print(f'\nLoading {V32_FILE}...')
    df = pd.read_csv(V32_FILE, sep='\t', dtype=str, keep_default_na=False)
    print(f'  Total rows: {len(df)}')
    print(f'  Unique categories: {df["category"].nunique()}')

    # Step 3: Filter to mapped categories
    df['type_id'] = df['category'].map(CATEGORY_MAP)
    df_filtered = df[df['type_id'].notna()].copy()
    df_filtered['type_id'] = df_filtered['type_id'].astype(int)
    print(f'\nAfter category filter: {len(df_filtered)} rows ({len(df_filtered)/len(df)*100:.1f}%)')
    print(f'  Categories kept: {df_filtered["category"].value_counts().to_dict()}')
    print(f'  Excluded categories: {EXCLUDED_CATS}')

    # Step 4: Filter to v2.0 entities (intersection)
    df_filtered['mir_lower'] = df_filtered['mir'].str.strip().str.lower()
    df_filtered['dis_lower'] = df_filtered['disease'].str.strip().str.lower()
    df_filtered['mi_idx'] = df_filtered['mir_lower'].map(mi_idx)
    df_filtered['dis_idx'] = df_filtered['dis_lower'].map(dis_idx)
    df_intersect = df_filtered[df_filtered['mi_idx'].notna() & df_filtered['dis_idx'].notna()].copy()
    print(f'\nAfter v2.0 entity intersection: {len(df_intersect)} rows')
    print(f'  miRNAs covered: {df_intersect["mir_lower"].nunique()}/{len(mi_idx)}')
    print(f'  diseases covered: {df_intersect["dis_lower"].nunique()}/{len(dis_idx)}')

    # Step 5: Deduplicate (same miRNA-disease-type may have multiple PMIDs)
    df_unique = df_intersect.drop_duplicates(subset=['mi_idx', 'dis_idx', 'type_id'])
    print(f'\nAfter dedup (unique miRNA-disease-type): {len(df_unique)} associations')

    # Step 6: Handle multi-type per (miRNA, disease) — keep the FIRST type seen
    # (paper handles multi-type per pair as separate labels; here we follow v2.0 schema
    # which has one type per pair. Could be refined later.)
    df_pair = df_unique.drop_duplicates(subset=['mi_idx', 'dis_idx'])
    print(f'After multi-type collapse (1 type per pair): {len(df_pair)} pairs')
    type_counts = df_pair['type_id'].value_counts().sort_index()
    print(f'  Type distribution:')
    type_names = {1: 'circulation', 2: 'epigenetics', 3: 'target', 4: 'genetics'}
    for t, count in type_counts.items():
        pct = count / len(df_pair) * 100
        print(f'    {type_names[t]:<14} ({t}): {count:5} pairs ({pct:.1f}%)')

    # Step 7: Build output folder, copy v2.0 similarity files
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    print(f'\nCreating {OUT_DIR}/...')

    # Copy similarity matrices + entity name files from v2.0
    files_to_copy = ['D_SSM1.txt', 'D_SSM2.txt', 'M_FSM.txt', 'M_GSM.txt',
                     'miRNA name.xls', 'disease name.xls',
                     'd_gs.xlsx', 'm_ss.xlsx',
                     'dis_sem_sim_2.0.csv', 'mi_fun_sim_2.0.csv', 'mi_dis_mat_2.0.csv',
                     'D_GSM.txt']
    for f in files_to_copy:
        src = os.path.join(V20_DIR, f)
        dst = os.path.join(OUT_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f'  Copied {f}')

    # Step 8: Write v3.2 association CSV in v2.0 format
    out_csv = os.path.join(OUT_DIR, 'multi_all_mirna_disease_pairs_without_negative.csv')
    # Format: miRNA_idx, disease_idx, type (matching v2.0 CSV format)
    # v2.0 CSV is 1-based index, type ∈ {1, 2, 3, 4}
    out_df = pd.DataFrame({
        'miRNA': (df_pair['mi_idx'].astype(int) + 1).values,
        'disease': (df_pair['dis_idx'].astype(int) + 1).values,
        'type': df_pair['type_id'].astype(int).values,
    })
    out_df.to_csv(out_csv, index=False, header=False)
    print(f'\n[save] {out_csv} — {len(out_df)} associations')

    # Also write case study version (same content)
    out_csv_cs = os.path.join(OUT_DIR, 'multi_all_mirna_disease_pairs_without_negative_forcasestudy.csv')
    out_df.to_csv(out_csv_cs, index=False, header=False)

    # Step 9: Summary stats
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'  v3.2 raw:                   32,282 associations')
    print(f'  After category filter:      {len(df_filtered):,} ({len(df_filtered)/32282*100:.1f}%)')
    print(f'  After v2.0 intersect:       {len(df_intersect):,}')
    print(f'  After dedup:                {len(df_unique):,}')
    print(f'  Final (1 type/pair):        {len(df_pair):,}')
    print(f'')
    print(f'  v2.0 baseline:              1,498 unique pairs (paper)')
    print(f'  v3.2 filtered:              {len(df_pair):,} unique pairs')
    print(f'  Ratio:                      {len(df_pair)/1498:.2f}× v2.0')
    print(f'')
    print(f'Output: {OUT_DIR}/')
    print(f'  → Use --dataset v3.2_filtered_495m383D to train')


if __name__ == '__main__':
    main()
