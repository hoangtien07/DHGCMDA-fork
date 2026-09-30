"""Ensemble evaluation: average per-fold predictions across seeds on an
IDENTICAL CV split (all runs used --load_folds), then recompute metrics with
the repo's own evaluate_optimized_with_comprehensive_metrics — same code path
as the training runs, no reimplementation.

Usage: .venv/bin/python /home/ubuntu/eval_ensemble.py /home/ubuntu/preds_s0 /home/ubuntu/preds_s1 ...
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/repos/DHGCMDA-fork')
from main_experiments_hetero1 import evaluate_optimized_with_comprehensive_metrics

dirs = sys.argv[1:]
NFOLDS = 5

top1_sum = {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}
binary_sum = np.zeros(7)
cv_type_sum = np.zeros(9)

for fold in range(1, NFOLDS + 1):
    pres_one, pres_zero = [], []
    true_one = true_zero = None
    for d in dirs:
        p = torch.load(f'{d}/fold{fold}.pt', weights_only=False)
        pres_one.append(p['pre_one'].cpu())
        pres_zero.append(p['pre_zero'].cpu())
        if true_one is None:
            true_one, true_zero = p['true_one'].cpu(), p['true_zero'].cpu()
        else:
            assert torch.equal(true_one, p['true_one'].cpu()), f'fold{fold}: true_one differs across seeds (split not fixed!)'
            assert torch.equal(true_zero, p['true_zero'].cpu()), f'fold{fold}: true_zero differs across seeds'

    pre_one = torch.stack(pres_one).mean(0)
    pre_zero = torch.stack(pres_zero).mean(0)

    res = evaluate_optimized_with_comprehensive_metrics(true_one, true_zero, pre_one, pre_zero)
    binary, cv_type, top1 = res[:3]
    binary_sum += np.asarray(binary).ravel()[:7]
    cv_type_sum += np.asarray(cv_type).ravel()[:9]
    for k in top1_sum:
        top1_sum[k] += top1[k]
    print(f'fold{fold}: Top-1 F1 {top1["top1_f1"]:.4f}  AUC {binary[0][0]:.4f}')

print('\n=== ENSEMBLE ({} seeds x {} folds) ==='.format(len(dirs), NFOLDS))
b = binary_sum / NFOLDS
print(f'AUC {b[0]:.4f}  AUPR {b[1]:.4f}  F1 {b[2]:.4f}')
for k, v in top1_sum.items():
    print(f'{k} {v / NFOLDS:.4f}')
