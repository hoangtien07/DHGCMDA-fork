"""run_v32_spldeval.py — DHGCMDA (fixed multilabel head) on the exact paper
artifact MDAv3.2-3 under the SPLD golden pair-fold protocol.

Leakage rules per outer fold: M_train/GIP/KNN/hetero edges/class weights
rebuilt from TRAIN pairs only; Y masked per pair (all 5 channels).
Similarity policy (M0 F24): DSSM = static external; MISIM = label-derived
(opt-in --misim legacy track only); GIP recomputed per fold.

Eval: forensics/eval_top1 (golden SPLD evaluator) + multilabel metrics.
Inner 80/20 split of outer-train for early stopping.

Usage: python3 -u forensics/run_v32_spldeval.py --epoch 300 --seed 0 [--misim]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'forensics'))
from hetero_model import HeterogenousGraphCLAMIR
import hypergraph_construct_KNN
from main_experiments_hetero1 import SimplifiedMultiTypeAssociationLoss, get_L2reg
from eval_top1 import build_pair_folds, mask_pair, evaluate_split

device = torch.device('cpu')
TYPE_ORDER = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']


def Gauss(adj, N):
    adj = adj.astype(np.float32)
    sq = adj * adj
    s = sq.sum(1, keepdims=True)
    diff = s + s.T - 2 * adj @ adj.T
    rm = N / sq.sum() if sq.sum() > 0 else 1.0
    return np.exp(-rm * diff).astype(np.float32)


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
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--predictor_mode', default='full_bilinear')
    ap.add_argument('--K_neigs', type=int, default=13)
    ap.add_argument('--update_graph_frequency', type=int, default=5)
    ap.add_argument('--misim', action='store_true', help='legacy leaky track: static MISIM as view1 miRNA feature')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    DATA = os.path.join(REPO, 'v3.2_spld_paper')
    Y = np.load(os.path.join(DATA, 'Y_multilabel.npy'))      # [411,271,5]
    M_MISIM = np.load(os.path.join(DATA, 'M_MISIM.npy'))     # MISIM (leaky opt-in)
    D_MESH = np.load(os.path.join(DATA, 'D_MESH.npy'))       # DSSM (static)
    mi_num, dis_num, K = Y.shape
    A = (Y.sum(-1) > 0).astype(int)
    folds = build_pair_folds(A)
    prov = 'DSSM+GIP(train)' + ('+MISIM(LEAKY)' if args.misim else '')
    print(f"Y {mi_num}x{dis_num}x{K} pairs={A.sum()} prov={prov}")

    class A2: pass
    margs = A2()
    margs.alpha = 0.5
    margs.dataset = 'v3.2_wang_multilabel'
    margs.loss_mode = 'multilabel_bce'
    margs.exist_weight = 0.1
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
    fold_rows = []
    zero_pairs = np.argwhere(Y.sum(-1) == 0)

    for fold, test_index in enumerate(folds):
        t0 = time.time()
        Y_train = mask_pair(Y, test_index[0], test_index[1])
        M_train = (Y_train.sum(-1) > 0).astype(np.float32)
        GIP_M = Gauss(M_train, mi_num)
        GIP_D = Gauss(M_train.T, dis_num)

        mi_v1 = GIP_M if not args.misim else np.where(M_MISIM == 0, GIP_M, M_MISIM)
        x_mi = np.concatenate([M_train, mi_v1], 1)
        x_dis = np.concatenate([M_train.T, D_MESH], 1)
        x_mi_v2 = np.concatenate([M_train, GIP_M], 1)
        x_dis_v2 = np.concatenate([M_train.T, GIP_D], 1)

        G_mi_v1, G_mi_v2 = knn_G(x_mi, args.K_neigs), knn_G(x_mi_v2, args.K_neigs)
        G_dis_v1, G_dis_v2 = knn_G(x_dis, args.K_neigs), knn_G(x_dis_v2, args.K_neigs)
        hetero = build_hetero(M_train, mi_v1, D_MESH)

        pos_pairs = np.argwhere(M_train > 0)
        # inner 80/20 split for early stop
        rng = np.random.RandomState(args.seed * 10 + fold)
        perm = rng.permutation(len(pos_pairs))
        n_in = int(0.85 * len(pos_pairs))
        tr_idx, va_idx = pos_pairs[perm[:n_in]], pos_pairs[perm[n_in:]]
        n_neg = min(len(zero_pairs), len(pos_pairs) * 10)
        neg_pairs = zero_pairs[np.random.choice(len(zero_pairs), n_neg, replace=False)]
        one_index = torch.from_numpy(tr_idx).long()
        zero_index = torch.from_numpy(neg_pairs).long()

        counts = [max(1, int(Y_train[:, :, k].sum())) for k in range(K)]
        beta = 0.99999
        en = [(1 - beta ** n) / (1 - beta) for n in counts]
        rw = [1.0 / e for e in en]
        fold_weights = torch.tensor([w * K / sum(rw) for w in rw])

        model = HeterogenousGraphCLAMIR(mi_num, dis_num, [256, 256], 64, margs).to(device)
        crit = SimplifiedMultiTypeAssociationLoss(margs, model)
        crit.class_weights = fold_weights.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        Ytr = torch.from_numpy(Y_train).to(device)
        x_mi_t = torch.from_numpy(x_mi).to(device)
        x_dis_t = torch.from_numpy(x_dis).to(device)
        mi_v1_t = torch.from_numpy(mi_v1).to(device)
        D_MESH_t = torch.from_numpy(D_MESH).to(device)

        best_f1, best_state, patience = -1.0, None, 0
        for ep in range(1, args.epoch + 1):
            model.train()
            score, mi_cl, dis_cl, mi_rec, dis_rec = model(
                x_mi_t, x_dis_t, G_mi_v1, G_mi_v2, G_dis_v1, G_dis_v2, hetero)
            if ep % args.update_graph_frequency == 0:
                hetero = build_hetero(M_train, mi_rec.detach().cpu().numpy(),
                                      dis_rec.detach().cpu().numpy())
            loss = crit(one_index, zero_index, score, Ytr)
            rec = F.mse_loss(mi_rec, mi_v1_t) + F.mse_loss(dis_rec, D_MESH_t)
            total = loss + (mi_cl + dis_cl) + rec + 1e-4 * get_L2reg(model.parameters())
            if torch.isnan(total):
                print(f"fold{fold} ep{ep} NaN — stop")
                break
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ep % 25 == 0:
                model.eval()
                with torch.no_grad():
                    s_val, *_ = model(x_mi_t, x_dis_t, G_mi_v1, G_mi_v2,
                                      G_dis_v1, G_dis_v2, hetero)
                    s_val = s_val.cpu().numpy()[:, :, 1:]
                met = evaluate_split(s_val, Y, (va_idx[:, 0], va_idx[:, 1]))
                if met['legacy_f1'] > best_f1:
                    best_f1, patience = met['legacy_f1'], 0
                    best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
                else:
                    patience += 1
                if ep % 100 == 0:
                    print(f"fold{fold} ep{ep} loss={total.item():.4f} "
                          f"valF1={met['legacy_f1']:.4f} best={best_f1:.4f}")
                if patience >= 4:
                    break
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            score, *_ = model(x_mi_t, x_dis_t, G_mi_v1, G_mi_v2,
                              G_dis_v1, G_dis_v2, hetero)
            score = score.cpu().numpy()[:, :, 1:]
        met = evaluate_split(score, Y, test_index)
        met = {k2: v for k2, v in met.items() if not k2.startswith('_')}
        met['prov'] = prov
        met['elapsed_s'] = round(time.time() - t0, 1)
        met['best_val_f1'] = best_f1
        fold_rows.append(met)
        print(f"[fold {fold}] F1={met['legacy_f1']:.4f} "
              f"P={met['legacy_top1_precision']:.4f} miR={met['legacy_micro_recall']:.4f} "
              f"maR={met['legacy_macro_recall']:.4f} Hit@1={met['Hit@1']:.4f} "
              f"macroAUPR={met['macro_AUPR']:.4f} [{met['elapsed_s']}s]", flush=True)

    keys = ['legacy_top1_precision', 'legacy_micro_recall', 'legacy_macro_recall',
            'legacy_f1', 'macro_AUPR', 'micro_AUPR', 'Hit@1', 'Recall@3']
    agg = {key: float(np.nanmean([f[key] for f in fold_rows])) for key in keys}
    tag = args.tag or ('misim' if args.misim else 'dssm_gip')
    out = os.path.join(REPO, 'forensics', f'v32_spldeval_{tag}.json')
    def _c(o):
        try: return float(o)
        except TypeError: return o
    json.dump({'aggregate': agg, 'folds': fold_rows,
               'config': vars(args), 'prov': prov}, open(out, 'w'), indent=2, default=_c)
    print("AGGREGATE:", json.dumps(agg, indent=2))
    print("saved", out)


if __name__ == '__main__':
    main()
