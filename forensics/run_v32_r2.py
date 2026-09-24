"""run_v32_r2.py — R2: leakage-free DHGCMDA run on the v3.2 lineage approximation.

Differences vs released pipeline (see forensics/FORENSICS_MODEL_PATH.md):
  * Y[m,d,5] multi-label tensor preserved end-to-end (no scalar collapse).
  * Per fold, ALL label-derived inputs are rebuilt from TRAIN triplets only:
      - association matrix fed to node features / KNN hypergraphs  -> M_train
      - GIP similarity                                             -> Gauss(M_train)
      - hetero 'associates' edges                                  -> M_train > 0
      - supervision target                                         -> Y_train (masked)
      - class weights                                              -> train per-type counts
  * Evaluator accepts K=5 types (corrected Top-1 metric, same hybrid
    micro-P + macro-R + F1 definition validated on v2.0).

External (non-label) similarities used as the two real views:
      - miRNA View1 feature : M_MISIM (MISIM 2.0 functional similarity slice)
      - disease View1 feature: D_MESH (Wang-style MeSH-tree semantic similarity)
      - View2 KNN geometry  : per-fold GIP matrices
  These are leak-free: computed from external databases, independent of labels.

Reuses repo model code: hetero_model.HeterogenousGraphCLAMIR,
hypergraph_construct_KNN, main_experiments_hetero1.SimplifiedMultiTypeAssociationLoss
(loss_mode='multilabel_bce'). No repo source files are modified.

Usage: python forensics/run_v32_r2.py --epoch 300 --validation 5 --seed 0
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DATA = os.environ.get('R2_DATA', os.path.join(REPO, 'v3.2_lineage_approx'))

from hetero_model import HeterogenousGraphCLAMIR
import hypergraph_construct_KNN
from main_experiments_hetero1 import SimplifiedMultiTypeAssociationLoss, get_L2reg
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score

device = torch.device('cpu')
TYPE_ORDER = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']


def Gauss_M(adj, N):
    adj = adj.astype(np.float32)
    sq = adj * adj
    s = sq.sum(1, keepdims=True)
    diff = s + s.T - 2 * adj @ adj.T
    rm = N / sq.sum() if sq.sum() > 0 else 1.0
    return np.exp(-rm * diff).astype(np.float32)

def Gauss_D(adj, M):
    T = adj.T.astype(np.float32)
    sq = T * T
    s = sq.sum(1, keepdims=True)
    diff = s + s.T - 2 * T @ T.T
    rd = M / sq.sum() if sq.sum() > 0 else 1.0
    return np.exp(-rd * diff).astype(np.float32)

def knn_G(X, k=13):
    H = hypergraph_construct_KNN.construct_H_with_KNN(np.asarray(X, dtype=np.float32), [k], False)
    return hypergraph_construct_KNN._generate_G_from_H(H).float()

def build_hetero(M_train, mi_sim, dis_sim, thr=0.5):
    hd = HeteroData()
    hd['miRNA'].x = torch.eye(M_train.shape[0])
    hd['disease'].x = torch.eye(M_train.shape[1])
    md = torch.nonzero(torch.from_numpy(M_train) > 0, as_tuple=False)
    if len(md):
        hd['miRNA', 'associates', 'disease'].edge_index = md.t().long()
        hd['disease', 'associates', 'miRNA'].edge_index = md.flip(1).t().long()
    mm = torch.nonzero(torch.from_numpy(mi_sim) > thr, as_tuple=False)
    if len(mm):
        hd['miRNA', 'similar', 'miRNA'].edge_index = mm.t().long()
    dd = torch.nonzero(torch.from_numpy(dis_sim) > thr, as_tuple=False)
    if len(dd):
        hd['disease', 'similar', 'disease'].edge_index = dd.t().long()
    return hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch', type=int, default=300)
    ap.add_argument('--validation', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--predictor_mode', default='full_bilinear')
    ap.add_argument('--K_neigs', type=int, default=13)
    ap.add_argument('--update_graph_frequency', type=int, default=5)
    ap.add_argument('--data', default=None, help='dataset dir (default: v3.2_lineage_approx)')
    args = ap.parse_args()
    if args.data:
        globals()['DATA'] = args.data

    Y = np.load(os.path.join(DATA, 'Y_multilabel.npy'))      # [m,d,5]
    M_MISIM = np.load(os.path.join(DATA, 'M_MISIM.npy'))     # [m,m]
    D_MESH = np.load(os.path.join(DATA, 'D_MESH.npy'))       # [d,d]
    triplets = np.argwhere(Y.sum(-1) > 0)
    trip_types = [(i, j, k) for (i, j) in triplets for k in np.where(Y[i, j] > 0)[0]]
    trip_types = np.array(trip_types)
    mi_num, dis_num, K = Y.shape
    print(f"Y: {mi_num}x{dis_num}x{K}, triplets={len(trip_types)}, pairs={len(triplets)}")

    # args shim for the repo loss/model
    class A: pass
    margs = A()
    margs.alpha = 0.5
    margs.dataset = 'v3.2_wang_multilabel'  # gives 5-length class_weights buffer
    margs.loss_mode = 'multilabel_bce'
    margs.exist_weight = 0.1                # paper-aligned (M5 finding)
    margs.num_association_types = K
    margs.predictor_mode = args.predictor_mode
    margs.ablation = 'none'
    margs.enable_inter_view_cl = True
    margs.inter_view_weight = 0.3
    margs.n_head = 4
    margs.nlayer = 2
    margs.dropout = 0.3
    margs.cl_weight_override = 1.0
    margs.recon_weight_override = 1.0
    margs.multilabel_target_path = ''

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    kf = KFold(n_splits=args.validation, shuffle=True, random_state=args.seed)

    all_real, all_pred, all_exist_y, all_exist_s = [], [], [], []
    fold_rows = []
    zero_pairs = np.argwhere(Y.sum(-1) == 0)

    for fold, (tr_i, te_i) in enumerate(kf.split(trip_types)):
        t0 = time.time()
        tr, te = trip_types[tr_i], trip_types[te_i]
        # --- leakage-free fold data ---
        Y_train = np.zeros_like(Y)
        M_train = np.zeros((mi_num, dis_num), np.float32)
        for i, j, k in tr:
            Y_train[i, j, k] = 1.0
            M_train[i, j] = 1.0
        GIP_M = Gauss_M(M_train, mi_num)
        GIP_D = Gauss_D(M_train, dis_num)

        x_mi = np.concatenate([M_train, M_MISIM], 1)          # [m, d+m]
        x_dis = np.concatenate([M_train.T, D_MESH], 1)        # [d, m+d]
        x_mi_v2 = np.concatenate([M_train, GIP_M], 1)
        x_dis_v2 = np.concatenate([M_train.T, GIP_D], 1)

        G_mi_v1, G_mi_v2 = knn_G(x_mi, args.K_neigs), knn_G(x_mi_v2, args.K_neigs)
        G_dis_v1, G_dis_v2 = knn_G(x_dis, args.K_neigs), knn_G(x_dis_v2, args.K_neigs)
        hetero = build_hetero(M_train, M_MISIM, D_MESH)

        # train indices (pairs)
        pos_pairs = np.unique(tr[:, :2], axis=0)
        neg_pool = zero_pairs.copy()
        # negatives must be zero in BOTH train and test contexts -> use true-zero pairs only
        n_neg = min(len(neg_pool), len(pos_pairs) * 10)
        neg_pairs = neg_pool[np.random.choice(len(neg_pool), n_neg, replace=False)]
        one_index = torch.from_numpy(pos_pairs).long()
        zero_index = torch.from_numpy(neg_pairs).long()

        # per-fold class weights (Effective Number) from TRAIN per-type counts
        counts = [max(1, int(Y_train[:, :, k].sum())) for k in range(K)]
        beta = 0.99999
        en = [(1 - beta ** n) / (1 - beta) for n in counts]
        rw = [1.0 / e for e in en]
        sw = sum(rw)
        fold_weights = torch.tensor([w * K / sw for w in rw])

        model = HeterogenousGraphCLAMIR(mi_num, dis_num, [256, 256], 64, margs).to(device)
        crit = SimplifiedMultiTypeAssociationLoss(margs, model)
        crit.class_weights = fold_weights.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        Ytr = torch.from_numpy(Y_train).to(device)
        x_mi_t = torch.from_numpy(x_mi).to(device)
        x_dis_t = torch.from_numpy(x_dis).to(device)
        M_MISIM_t = torch.from_numpy(M_MISIM).to(device)
        D_MESH_t = torch.from_numpy(D_MESH).to(device)

        model.train()
        for ep in range(1, args.epoch + 1):
            score, mi_cl, dis_cl, mi_rec, dis_rec = model(
                x_mi_t, x_dis_t, G_mi_v1, G_mi_v2, G_dis_v1, G_dis_v2, hetero)
            if ep % args.update_graph_frequency == 0:
                hetero = build_hetero(M_train,
                                      mi_rec.detach().cpu().numpy(),
                                      dis_rec.detach().cpu().numpy())
            loss = crit(one_index, zero_index, score, Ytr)
            rec = F.mse_loss(mi_rec, M_MISIM_t) + F.mse_loss(dis_rec, D_MESH_t)
            total = loss + (mi_cl + dis_cl) + rec + 1e-4 * get_L2reg(model.parameters())
            if torch.isnan(total):
                print(f"fold{fold} ep{ep} NaN — stop"); break
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ep % 100 == 0:
                print(f"fold{fold} ep{ep} loss={total.item():.4f}")

        # --- eval ---
        model.eval()
        with torch.no_grad():
            score, *_ = model(x_mi_t, x_dis_t, G_mi_v1, G_mi_v2, G_dis_v1, G_dis_v2, hetero)
            score = score.cpu().numpy()
        # Top-1 on test triplets
        yt, yp = [], []
        for i, j, k in te:
            yt.append(int(k)); yp.append(int(np.argmax(score[i, j, 1:])))
            all_real.append(int(k)); all_pred.append(int(np.argmax(score[i, j, 1:])))
        # binary on test pairs vs equal sampled true-zero pairs
        test_pairs = np.unique(te[:, :2], axis=0)
        zneg = zero_pairs[np.random.choice(len(zero_pairs), len(test_pairs), replace=False)]
        e_y = np.r_[np.ones(len(test_pairs)), np.zeros(len(zneg))]
        e_s = np.r_[score[test_pairs[:, 0], test_pairs[:, 1], 0],
                    score[zneg[:, 0], zneg[:, 1], 0]]
        auc = roc_auc_score(e_y, e_s); aupr = average_precision_score(e_y, e_s)
        acc = accuracy_score(yt, yp)
        mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
        all_exist_y += list(e_y); all_exist_s += list(e_s)
        print(f"[fold {fold}] top1-acc={acc:.4f} macroP={mp:.4f} macroR={mr:.4f} macroF1={mf:.4f} "
              f"AUC={auc:.4f} AUPR={aupr:.4f} time={time.time()-t0:.0f}s")
        fold_rows.append(dict(fold=fold, acc=acc, macroP=mp, macroR=mr, macroF1=mf,
                              auc=auc, aupr=aupr, train_counts=counts))

    # overall corrected Top-1 (same hybrid def: micro-P + macro-R + F1)
    yt, yp = np.array(all_real), np.array(all_pred)
    acc = accuracy_score(yt, yp)
    tc = defaultdict(int); tt = defaultdict(int)
    for a, b in zip(yt, yp):
        tt[a] += 1; tc[a] += (a == b)
    macroR = np.mean([tc[k] / tt[k] for k in tt])
    P = acc
    F1 = 2 * P * macroR / (P + macroR) if P + macroR > 0 else 0
    auc = roc_auc_score(all_exist_y, all_exist_s)
    aupr = average_precision_score(all_exist_y, all_exist_s)
    mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    summary = dict(top1_precision=P, top1_recall_macro=macroR, top1_f1_hybrid=F1,
                   acc=acc, macroP=mp, macroR_sk=mr, macroF1=mf,
                   auc=auc, aupr=aupr, folds=fold_rows,
                   config=dict(epoch=args.epoch, predictor=args.predictor_mode,
                               K=args.K_neigs, seed=args.seed))
    print("\n=== R2 OVERALL ===")
    print(json.dumps({k: v for k, v in summary.items() if k != 'folds'}, indent=2))
    out = os.path.join(REPO, 'forensics', 'r2_v32_results.json')
    json.dump(summary, open(out, 'w'), indent=2)
    print("saved", out)


if __name__ == '__main__':
    main()
