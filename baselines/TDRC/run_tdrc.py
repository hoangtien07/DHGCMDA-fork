"""Wrapper để chạy TDRC trên HMDD v3.2 + xuất metrics JSON.

Tận dụng experiments.py có sẵn. Lưu metrics CV_type + CV_triplet vào file JSON cho báo cáo.
"""
import os
import sys
import json
import time
import numpy as np

# Make modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import GetData
from experiments import Experiments


def run_v32():
    print('[TDRC] Loading HMDD v3.2 data...')
    # data.py expects: GetData(root, miRNA_num=713, dis_num=447), root='' → looks for HMDD3.2_processed/
    # Sau khi extract rar có thư mục data_v32/HMDD3.2_processed/. data.py hardcode path → cần symlink/move.
    # Easier: pass root='data_v32' để khớp HMDD3.2_processed subfolder.
    data = GetData(root='data_v32')

    print(f'[TDRC] Tensor shape: {data.type_tensor.shape}')  # (m, d, 5)
    print(f'[TDRC] Total non-zero: {int(data.type_tensor.sum())}')

    # CV_type — Top-1 metrics
    print('\n[TDRC] CV_type (5-fold)... — ~3-5 min')
    t0 = time.time()
    exp_type = Experiments(data, model_name='TDRC', r=4, alpha=0.125, beta=0.25, lam=0.001, tol=1e-6, max_iter=500)
    type_metrics = exp_type.CV_type()
    print(f'[TDRC] CV_type done in {time.time() - t0:.1f}s')
    print(f'  Top-1 Precision: {type_metrics[0]:.4f}')
    print(f'  Top-1 Recall:    {type_metrics[1]:.4f}')
    top1_p = float(type_metrics[0])
    top1_r = float(type_metrics[1])
    top1_f1 = 2 * top1_p * top1_r / max(top1_p + top1_r, 1e-7)

    # CV_triplet — Binary AUPR/AUC/F1
    print('\n[TDRC] CV_triplet (5-fold × 10 negative samples)... — ~10-15 min')
    t0 = time.time()
    exp_trip = Experiments(data, model_name='TDRC', r=4, alpha=0.125, beta=0.25, lam=0.001, tol=1e-6, max_iter=500)
    trip_metrics = exp_trip.CV_triplet()
    print(f'[TDRC] CV_triplet done in {time.time() - t0:.1f}s')
    print(f'  AUPR:  {trip_metrics[0, 0]:.4f}')
    print(f'  AUC:   {trip_metrics[0, 1]:.4f}')
    print(f'  F1:    {trip_metrics[0, 2]:.4f}')

    out = {
        'method': 'TDRC',
        'dataset': 'HMDD v3.2',
        'CV_type': {
            'top1_precision': top1_p,
            'top1_recall': top1_r,
            'top1_f1': float(top1_f1),
        },
        'CV_triplet': {
            'AUPR':       float(trip_metrics[0, 0]),
            'AUC':        float(trip_metrics[0, 1]),
            'F1':         float(trip_metrics[0, 2]),
            'Accuracy':   float(trip_metrics[0, 3]),
            'Recall':     float(trip_metrics[0, 4]),
            'Specificity':float(trip_metrics[0, 5]),
            'Precision':  float(trip_metrics[0, 6]),
        },
        'hyperparams': {'r': 4, 'alpha': 0.125, 'beta': 0.25, 'lam': 0.001, 'max_iter': 500},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'baseline_TDRC_v32.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[TDRC] Saved: {out_path}')


if __name__ == '__main__':
    run_v32()
