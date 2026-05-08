"""Case study: train DHGCMDA trên 100% data (no CV split), predict top-15 miRNA cho
breast neoplasms + hepatocellular carcinoma → cross-check với paper Table 5/6.

Reuse infrastructure từ main_experiments_hetero1.py — chỉ thay CV split bằng full-data
training. Output: results/case_study_breast.csv + results/case_study_hcc.csv.
"""
import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

from param import parameter_parser
from prepareData import prepare_data_optimized
from hetero_model import HeterogenousGraphCLAMIR
from main_experiments_hetero1 import (
    SimplifiedMultiTypeAssociationLoss,
    constructHW_knn,
    constructHW_kmean,
    create_hetero_data_optimized,
    get_L2reg,
    seed_torch,
    device,
)

RESULTS_DIR = 'results'
DATA_DIR = 'v2.0_495m383D'

# Paper Table 5: top-15 breast neoplasms (15 miRNAs)
PAPER_BREAST_TOP15 = {
    'hsa-mir-148a': 'epigenetics',
    'hsa-mir-1290': 'target',
    'hsa-mir-449b': 'genetics',
    'hsa-mir-195':  'circulation',
    'hsa-mir-130a': 'target',
    'hsa-mir-632':  'target',
    'hsa-mir-148b': 'circulation',
    'hsa-mir-335':  'epigenetics',
    'hsa-mir-4521': 'target',
    'hsa-mir-30c':  'target',
    'hsa-mir-593':  'epigenetics',
    'hsa-mir-125b': 'genetics',
    'hsa-mir-488':  'genetics',
    'hsa-mir-1323': 'circulation',
    'hsa-mir-24':   'genetics',
}

# Paper Table 6: top-15 hepatocellular carcinoma
PAPER_HCC_TOP15 = {
    'hsa-mir-196a': 'epigenetics',
    'hsa-mir-217':  'target',
    'hsa-mir-429':  'epigenetics',
    'hsa-mir-26b':  'genetics',
    'hsa-mir-411':  'target',
    'hsa-mir-1469': 'circulation',
    'hsa-mir-30d':  'genetics',
    'hsa-mir-204':  'target',
    'hsa-mir-487b': 'circulation',
    'hsa-mir-766':  'target',
    'hsa-mir-153':  'genetics',
    'hsa-mir-657':  'target',
    'hsa-mir-19b':  'genetics',
    'hsa-mir-133a': 'target',
    'hsa-let-7d':   'target',
}

TYPE_MAP = {1: 'circulation', 2: 'epigenetics', 3: 'target', 4: 'genetics'}


def load_name_mappings():
    """Read miRNA name.xls + disease name.xls → 2 dict idx → name."""
    mi_path = os.path.join(DATA_DIR, 'miRNA name.xls')
    dis_path = os.path.join(DATA_DIR, 'disease name.xls')

    # File .xls dùng xlrd. Format: 2 columns, no header. Col 0 = name, Col 1 = index (1-based).
    mi_df = pd.read_excel(mi_path, header=None, engine='xlrd')
    dis_df = pd.read_excel(dis_path, header=None, engine='xlrd')

    # Build idx (0-based) → name dict
    mi_idx2name = {}
    for _, row in mi_df.iterrows():
        name = str(row[0]).strip().lower()  # hsa-mir-XXX
        idx = int(row[1]) - 1
        mi_idx2name[idx] = name

    dis_idx2name = {}
    for _, row in dis_df.iterrows():
        name = str(row[0]).strip().lower()
        idx = int(row[1]) - 1
        dis_idx2name[idx] = name

    return mi_idx2name, dis_idx2name


def find_disease_idx(dis_idx2name, query_substr):
    """Tìm index của disease có tên chứa substring."""
    matches = [(i, n) for i, n in dis_idx2name.items() if query_substr.lower() in n]
    if not matches:
        raise ValueError(f"No disease found matching '{query_substr}'")
    if len(matches) > 1:
        print(f"[INFO] Multiple matches for '{query_substr}': {matches[:3]}... — picking first")
    return matches[0][0], matches[0][1]


def train_full_data(args):
    """Train DHGCMDA trên 100% associations. Return final score tensor [495, 383, 5]."""
    seed_torch(args.seed)

    print(f"[case_study] Loading data...")
    dataset = prepare_data_optimized(args)
    if dataset is None:
        raise RuntimeError("prepare_data_optimized failed")

    # Lấy từ fold 0 nhưng OVERRIDE: dùng tất cả nonzero làm one_index
    association_matrix = dataset['md_p'].to(device).float()  # [495, 383] với value = 1..4
    binary_target = (association_matrix != 0).float()

    # Build all_one_index = mọi (i,j) có association
    nonzero_idx = torch.nonzero(binary_target, as_tuple=False)  # [N, 2]
    print(f"[case_study] Total nonzero associations: {nonzero_idx.shape[0]}")

    # Build all_zero_index, then sample negative 10x
    zero_idx = torch.nonzero(binary_target == 0, as_tuple=False)  # [M, 2]
    perm = torch.randperm(zero_idx.shape[0], device=zero_idx.device)
    sampled_zero = zero_idx[perm[:nonzero_idx.shape[0] * 10]]
    print(f"[case_study] Sampled negative: {sampled_zero.shape[0]}")

    one_index_tensor = nonzero_idx.to(device)
    zero_index_tensor = sampled_zero.to(device)

    # Build dual-view features (giống main_experiments_hetero1.py:705-787)
    dis_sem_data = dataset['dis_sem'].to(device).float()
    mi_fun_data = dataset['mi_fun'].to(device).float()
    d_gs_data = dataset['d_gs'].to(device).float()
    m_ss_data = dataset['m_ss'].to(device).float()

    concat_mi_view1 = torch.cat([association_matrix, m_ss_data], dim=1).to(device).float()
    concat_mi_view2 = torch.cat([association_matrix, mi_fun_data], dim=1).to(device).float()
    concat_dis_view1 = torch.cat([association_matrix.t(), d_gs_data], dim=1).to(device).float()
    concat_dis_view2 = torch.cat([association_matrix.t(), dis_sem_data], dim=1).to(device).float()

    print(f"[case_study] Building hypergraphs (KNN K={args.K_neigs})...")
    G_mi_view1 = constructHW_knn(concat_mi_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
    G_mi_view2 = constructHW_knn(concat_mi_view2.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
    G_dis_view1 = constructHW_knn(concat_dis_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
    G_dis_view2 = constructHW_knn(concat_dis_view2.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)

    G_mi_view1 = G_mi_view1.to(device).float()
    G_mi_view2 = G_mi_view2.to(device).float()
    G_dis_view1 = G_dis_view1.to(device).float()
    G_dis_view2 = G_dis_view2.to(device).float()

    train_data_list = [dis_sem_data, mi_fun_data, None, None, association_matrix]
    hetero_data = create_hetero_data_optimized(train_data_list)

    # Init model
    model = HeterogenousGraphCLAMIR(args).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    regression_crit = SimplifiedMultiTypeAssociationLoss(args, model)

    print(f"[case_study] Training {args.epoch} epochs on full data...")
    start = time.time()
    for epoch in range(1, args.epoch + 1):
        score, mi_cl_loss, dis_cl_loss, mi_sim_recon, dis_sim_recon = model(
            concat_mi_view1, concat_dis_view1,
            G_mi_view1, G_mi_view2, G_dis_view1, G_dis_view2, hetero_data
        )
        mi_sim_recon = mi_sim_recon.to(device)
        dis_sim_recon = dis_sim_recon.to(device)

        # Dynamic graph update
        if epoch > 0 and epoch % args.update_graph_frequency == 0:
            hetero_data = create_hetero_data_optimized(train_data_list, mi_sim_recon, dis_sim_recon)

        mi_recon_loss = F.mse_loss(mi_sim_recon, mi_fun_data)
        dis_recon_loss = F.mse_loss(dis_sim_recon, dis_sem_data)
        recover_loss = regression_crit(one_index_tensor, zero_index_tensor, score, association_matrix)
        reg_loss = get_L2reg(model.parameters())

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 1.0 * (
            mi_recon_loss + dis_recon_loss) + 0.0001 * reg_loss

        optimizer.zero_grad()
        tol_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={tol_loss.item():.4f}, recover={recover_loss.item():.4f}, "
                  f"recon={(mi_recon_loss + dis_recon_loss).item():.4f}, t={time.time() - start:.1f}s")

    elapsed = time.time() - start
    print(f"[case_study] Training done in {elapsed:.1f}s")

    # Final forward — get score tensor [495, 383, 5]
    model.eval()
    with torch.no_grad():
        score, _, _, _, _ = model(
            concat_mi_view1, concat_dis_view1,
            G_mi_view1, G_mi_view2, G_dis_view1, G_dis_view2, hetero_data
        )

    return score.detach().cpu().numpy()


def rank_top15_for_disease(score, disease_idx, mi_idx2name, paper_top15_dict):
    """Rank top-15 miRNAs cho 1 disease theo max prob across 4 type channels.

    Args:
        score: ndarray [495, 383, 5] — channel 0 = existence, channels 1..4 = type probs
        disease_idx: int — disease column index (0-based)
        mi_idx2name: dict idx → "hsa-mir-XXX"
        paper_top15_dict: dict miRNA name → type (paper Table 5 hoặc 6)

    Returns:
        list of dicts: rank, miRNA_name, predicted_type, score, in_paper_top15, paper_type
    """
    # Lấy slice cho disease cụ thể: shape [495, 5]
    disease_slice = score[:, disease_idx, :]  # [495, 5]

    # Bỏ existence channel (0), lấy max prob trên 4 type channels (1..4)
    type_probs = disease_slice[:, 1:5]  # [495, 4]
    max_score = type_probs.max(axis=1)  # [495]
    max_type_idx = type_probs.argmax(axis=1)  # [495] — 0..3

    # Sort descending
    sorted_idx = np.argsort(-max_score)[:15]

    results = []
    for rank, mi_idx in enumerate(sorted_idx, start=1):
        name = mi_idx2name.get(int(mi_idx), f'mir_idx_{int(mi_idx)}')
        pred_type = TYPE_MAP[int(max_type_idx[mi_idx]) + 1]  # +1 vì TYPE_MAP key = 1..4
        in_paper = name in paper_top15_dict
        paper_type = paper_top15_dict.get(name, '')
        results.append({
            'rank': rank,
            'miRNA_name': name,
            'predicted_type': pred_type,
            'score': float(max_score[mi_idx]),
            'in_paper_top15': in_paper,
            'paper_type': paper_type,
            'type_match': bool(in_paper and pred_type == paper_type),
        })
    return results


def main():
    args = parameter_parser()
    args.epoch = 650
    args.validation = 1  # not used, but set for safety
    print(f"[case_study] device={device}, epoch={args.epoch}, n_head={args.n_head}, "
          f"update_freq={args.update_graph_frequency}")

    # Step 1: train trên full data
    score = train_full_data(args)
    print(f"[case_study] Score tensor shape: {score.shape}")

    # Step 2: load mapping files
    print(f"[case_study] Loading name mappings...")
    mi_idx2name, dis_idx2name = load_name_mappings()
    print(f"  miRNAs: {len(mi_idx2name)}")
    print(f"  Diseases: {len(dis_idx2name)}")

    # Step 3: tìm disease indices
    breast_idx, breast_name = find_disease_idx(dis_idx2name, 'breast neoplasms')
    hcc_idx, hcc_name = find_disease_idx(dis_idx2name, 'carcinoma, hepatocellular')
    print(f"  Breast: idx={breast_idx}, name='{breast_name}'")
    print(f"  HCC:    idx={hcc_idx}, name='{hcc_name}'")

    # Step 4: rank top-15
    breast_top15 = rank_top15_for_disease(score, breast_idx, mi_idx2name, PAPER_BREAST_TOP15)
    hcc_top15 = rank_top15_for_disease(score, hcc_idx, mi_idx2name, PAPER_HCC_TOP15)

    # Step 5: save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    breast_df = pd.DataFrame(breast_top15)
    hcc_df = pd.DataFrame(hcc_top15)
    breast_path = os.path.join(RESULTS_DIR, 'case_study_breast.csv')
    hcc_path = os.path.join(RESULTS_DIR, 'case_study_hcc.csv')
    breast_df.to_csv(breast_path, index=False)
    hcc_df.to_csv(hcc_path, index=False)
    print(f"\n[case_study] Saved: {breast_path}")
    print(f"[case_study] Saved: {hcc_path}")

    # Step 6: print summary
    breast_overlap = sum(1 for r in breast_top15 if r['in_paper_top15'])
    breast_type_match = sum(1 for r in breast_top15 if r['type_match'])
    hcc_overlap = sum(1 for r in hcc_top15 if r['in_paper_top15'])
    hcc_type_match = sum(1 for r in hcc_top15 if r['type_match'])

    print(f"\n=== SUMMARY ===")
    print(f"Breast neoplasms: {breast_overlap}/15 miRNA trùng paper, {breast_type_match}/15 type cũng khớp")
    print(f"HCC:              {hcc_overlap}/15 miRNA trùng paper, {hcc_type_match}/15 type cũng khớp")

    # Save summary JSON
    summary = {
        'breast': {
            'overlap_count': breast_overlap,
            'type_match_count': breast_type_match,
            'top15': breast_top15,
        },
        'hcc': {
            'overlap_count': hcc_overlap,
            'type_match_count': hcc_type_match,
            'top15': hcc_top15,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'case_study_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[case_study] Saved: {RESULTS_DIR}/case_study_summary.json")


if __name__ == '__main__':
    main()
