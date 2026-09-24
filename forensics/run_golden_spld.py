"""Golden SPLD run: replay recovered SPLDHyperAWNTF CV_type on MDAv3.2-3,
score each fold with forensics/eval_top1.py, and cross-check against SPLD's
internal metric per fold. Writes forensics/golden_spld_eval.json.

Run: cd /tmp/spld_model/SPLDHyperAWNTF && python3 -u <this> [max_iter]
"""
import sys, json, math, time
import numpy as np

sys.path.insert(0, '/home/ubuntu/repos/DHGCMDA-fork/forensics/spld_work/SPLDHyperAWNTF')
sys.path.insert(0, '/home/ubuntu/repos/DHGCMDA-fork/forensics/spld_work/SPLDHyperAWNTF/method')
sys.path.insert(0, '/home/ubuntu/repos/DHGCMDA-fork/forensics/spld_work/SPLDHyperAWNTF/data')
sys.path.insert(0, '/home/ubuntu/repos/DHGCMDA-fork/forensics')
import tensorly as tl
from data import MDAv3_3_GetData
from method import model
from eval_top1 import build_pair_folds, top1_metrics, multilabel_metrics
from fold_misim import functional_sim

MAX_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MISIM_MODE = sys.argv[2] if len(sys.argv) > 2 else 'static'  # static|fold|none
OUT = ('/home/ubuntu/repos/DHGCMDA-fork/forensics/golden_spld_eval.json'
       if MISIM_MODE == 'static' else
       f'/home/ubuntu/repos/DHGCMDA-fork/forensics/golden_spld_eval_{MISIM_MODE}.json')

root = '/home/ubuntu/repos/DHGCMDA-fork/forensics/spld_work/HMDD_data'
d = MDAv3_3_GetData.MDAv3_3_GetData(root)
mdl = model.Hyper_Model('SPLDHyperAWNTF')
A = d.type_tensor.sum(2)
folds = build_pair_folds((A > 0).astype(int), 5, 0)

res = {'folds': [], 'aggregate': {}, 'params': {'r': 4, 'alpha': 2, 'beta': 2,
       'lam_t': 0.001, 'lam_c': 0.3, 'tol': 1e-5, 'max_iter': MAX_ITER,
       'misim_mode': MISIM_MODE}}
agg = np.zeros(3)
for k, test_index in enumerate(folds):
    t0 = time.time()
    train_tensor = np.array(d.type_tensor, copy=True)
    train_tensor[test_index] = 0
    train_matrix = train_tensor.sum(2)
    train_matrix[train_matrix > 0] = 1

    miSIM = np.asmatrix(d.mi_sim)
    diSIM = np.asmatrix(d.dis_sim)
    nd, nm = train_matrix.shape[1], train_matrix.shape[0]
    rd = np.array([np.linalg.norm(train_matrix[:, i]) ** 2 for i in range(nd)])
    gamad = nd / rd.sum()
    rm = np.array([np.linalg.norm(train_matrix[j, :]) ** 2 for j in range(nm)])
    gamam = nm / rm.sum()
    DD = train_matrix.T @ train_matrix
    diff = np.diag(DD)[:, None] + np.diag(DD)[None, :] - 2 * DD
    DGSM = np.exp(-gamad * diff)
    MM = train_matrix @ train_matrix.T
    diffM = np.diag(MM)[:, None] + np.diag(MM)[None, :] - 2 * MM
    MGSM = np.exp(-gamam * diffM)
    ID = np.where(np.asarray(diSIM) == 0, DGSM, np.asarray(diSIM))
    if MISIM_MODE == 'static':
        IM = np.where(np.asarray(miSIM) == 0, MGSM, np.asarray(miSIM))
    elif MISIM_MODE == 'fold':
        FS = functional_sim(train_matrix, np.asarray(diSIM))
        IM = np.where(FS == 0, MGSM, FS)
    else:
        IM = MGSM
    tA = (train_tensor.sum(2) > 0).astype(float)
    concat_m = np.asmatrix(np.hstack([tA, IM]))
    concat_d = np.asmatrix(np.hstack([tA.T, ID]))
    W = tl.tensor(np.ones(train_tensor.size).reshape(train_tensor.shape))
    np.random.seed(100 + k)  # deterministic init; SPLD itself seeds folds only
    pred = mdl()(train_tensor, concat_m, concat_d, W, r=4, alpha=2, beta=2,
                 lam_t=0.001, lam_c=0.3, tol=1e-5, max_iter=MAX_ITER)
    pred = np.asarray(pred)

    # SPLD internal metric (verbatim semantics)
    TP = 0.0
    rec = 0.0
    real_sum = 0.0
    n = np.array(test_index).shape[1]
    for t in range(n):
        ps = np.asmatrix(pred[test_index[0][t], test_index[1][t]].flatten())
        rs = np.asmatrix(d.type_tensor[test_index[0][t], test_index[1][t]].flatten())
        pos = rs.sum()
        real_sum += pos
        si = np.array(np.argsort(ps))[0]
        ps[np.where(ps != 0)] = 0
        ps[:, si[-1:]] = 1
        tp = ps * rs.T
        TP += tp[0, 0]
        rec += tp[0, 0] / pos
    spld_fold = [TP / n, TP / real_sum, rec / n]

    np.save(OUT.replace('.json', f'_pred_fold{k}.npy'), pred.astype(np.float32))
    mine = top1_metrics(pred, d.type_tensor, test_index)
    ml = multilabel_metrics(pred, d.type_tensor, test_index)
    mine.update({f'ml_{k_}': v for k_, v in ml.items()})
    mine['spld_internal'] = spld_fold
    mine['elapsed_s'] = round(time.time() - t0, 1)
    mine['_hit_vec'] = mine['_hit_vec'].tolist()
    mine['_rec_vec'] = mine['_rec_vec'].tolist()
    json.dump(res, open(OUT, 'w'), indent=2, default=float)
    assert abs(mine['legacy_top1_precision'] - spld_fold[0]) < 1e-9
    assert abs(mine['legacy_micro_recall'] - spld_fold[1]) < 1e-9
    assert abs(mine['legacy_macro_recall'] - spld_fold[2]) < 1e-9
    print(f"fold{k}: F1={mine['legacy_f1']:.4f} P={spld_fold[0]:.4f} "
          f"miR={spld_fold[1]:.4f} maR={spld_fold[2]:.4f} [{mine['elapsed_s']}s]", flush=True)
    res['folds'].append(mine)
    agg += np.array(spld_fold)

agg /= len(folds)
res['aggregate'] = {'legacy_top1_precision': float(agg[0]),
                    'legacy_micro_recall': float(agg[1]),
                    'legacy_macro_recall': float(agg[2]),
                    'legacy_f1': float(2 * agg[0] * agg[2] / (agg[0] + agg[2]))}
json.dump(res, open(OUT, 'w'), indent=2, default=float)
print("AGGREGATE:", res['aggregate'])
