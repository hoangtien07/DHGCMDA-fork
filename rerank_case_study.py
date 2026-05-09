"""Rerank case study top-15 với 4 chiến lược khác nhau, dùng cached score
tensor [495, 383, 5] tại results/case_study_score.npy. Mục tiêu: kiểm tra
xem class collapse top-15 (15/15 cùng type) có fix được ở mức rank không
mà không cần retrain.

Strategies:
  (i)   max_type   = score[:, j, 1:5].max()         — hiện tại
  (ii)  sum_type   = score[:, j, 1:5].sum()         — tổng strength 4 type
  (iii) exist_only = score[:, j, 0]                  — pure binary head
  (iv)  softmax_t  = softmax(score[:, j, 1:5]).max() — type prob đã normalize

In ra bảng so sánh cho mỗi disease: overlap với paper top-15 + type match.
"""
import os
import json
import numpy as np
import pandas as pd

from case_study import (
    PAPER_BREAST_TOP15, PAPER_HCC_TOP15, TYPE_MAP,
    load_name_mappings, find_disease_idx,
)

RESULTS_DIR = 'results'
SCORE_PATH = os.path.join(RESULTS_DIR, 'case_study_score.npy')


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def rank_with_strategy(score, disease_idx, strategy, mi_idx2name, paper_top15_dict):
    """score: [495, 383, 5]. Returns (top15_list, type_distribution_dict)."""
    disease_slice = score[:, disease_idx, :]  # [495, 5]
    type_probs = disease_slice[:, 1:5]        # [495, 4]
    type_argmax = type_probs.argmax(axis=1)   # [495]

    if strategy == 'max_type':
        rank_score = type_probs.max(axis=1)
    elif strategy == 'sum_type':
        rank_score = type_probs.sum(axis=1)
    elif strategy == 'exist_only':
        rank_score = disease_slice[:, 0]
    elif strategy == 'softmax_t':
        rank_score = softmax(type_probs, axis=1).max(axis=1)
    else:
        raise ValueError(strategy)

    sorted_idx = np.argsort(-rank_score)[:15]

    top15 = []
    type_count = {'circulation': 0, 'epigenetics': 0, 'target': 0, 'genetics': 0}
    for rank, mi_idx in enumerate(sorted_idx, start=1):
        name = mi_idx2name.get(int(mi_idx), f'mir_idx_{int(mi_idx)}')
        pred_type = TYPE_MAP[int(type_argmax[mi_idx]) + 1]
        type_count[pred_type] += 1
        in_paper = name in paper_top15_dict
        paper_type = paper_top15_dict.get(name, '')
        top15.append({
            'rank': rank,
            'miRNA_name': name,
            'predicted_type': pred_type,
            'score': float(rank_score[mi_idx]),
            'in_paper_top15': in_paper,
            'paper_type': paper_type,
            'type_match': bool(in_paper and pred_type == paper_type),
        })
    return top15, type_count


def summarize(top15, label):
    overlap = sum(1 for r in top15 if r['in_paper_top15'])
    type_match = sum(1 for r in top15 if r['type_match'])
    return {'label': label, 'overlap': overlap, 'type_match': type_match}


def main():
    if not os.path.exists(SCORE_PATH):
        raise FileNotFoundError(f"{SCORE_PATH} not found — run case_study.py first")
    score = np.load(SCORE_PATH)
    print(f"[rerank] Loaded score tensor shape={score.shape}")
    print(f"[rerank] score channel ranges:")
    for c in range(score.shape[2]):
        s = score[:, :, c]
        print(f"  ch{c}: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}")

    mi_idx2name, dis_idx2name = load_name_mappings()
    breast_idx, _ = find_disease_idx(dis_idx2name, 'breast neoplasms')
    hcc_idx, _ = find_disease_idx(dis_idx2name, 'hepatocellular')

    strategies = ['max_type', 'sum_type', 'exist_only', 'softmax_t']
    all_results = {}

    for strategy in strategies:
        breast_top15, breast_dist = rank_with_strategy(
            score, breast_idx, strategy, mi_idx2name, PAPER_BREAST_TOP15)
        hcc_top15, hcc_dist = rank_with_strategy(
            score, hcc_idx, strategy, mi_idx2name, PAPER_HCC_TOP15)

        all_results[strategy] = {
            'breast': {'top15': breast_top15, 'type_dist': breast_dist,
                       'overlap': sum(1 for r in breast_top15 if r['in_paper_top15']),
                       'type_match': sum(1 for r in breast_top15 if r['type_match'])},
            'hcc': {'top15': hcc_top15, 'type_dist': hcc_dist,
                    'overlap': sum(1 for r in hcc_top15 if r['in_paper_top15']),
                    'type_match': sum(1 for r in hcc_top15 if r['type_match'])},
        }

    # In bảng so sánh
    print("\n" + "=" * 88)
    print(f"{'Strategy':<14} {'Disease':<8} {'Overlap':<10} {'TypeMatch':<10} {'Type distribution (15 miRNAs)':<40}")
    print("=" * 88)
    for strategy in strategies:
        for disease in ('breast', 'hcc'):
            r = all_results[strategy][disease]
            dist_str = ', '.join(f"{k}:{v}" for k, v in r['type_dist'].items() if v > 0)
            print(f"{strategy:<14} {disease:<8} {r['overlap']}/15{'':<5} {r['type_match']}/15{'':<5} {dist_str}")
    print("=" * 88)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'rerank_summary.json')
    serializable = {}
    for strategy in strategies:
        serializable[strategy] = {
            disease: {k: v for k, v in all_results[strategy][disease].items()}
            for disease in ('breast', 'hcc')
        }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[rerank] Saved {summary_path}")


if __name__ == '__main__':
    main()
