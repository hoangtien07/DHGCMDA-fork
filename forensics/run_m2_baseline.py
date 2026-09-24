"""M2 simple-ceiling tournament on MDAv3.2-3 (SPLD pair-fold protocol).

Models (provenance labeled):
  cooccur_prior : marginal type frequency (sanity floor)
  distmult      : <e_m, r_k, e_d> trained BCE           (train-derived)
  bilinear      : e_m^T W_k e_d trained BCE              (train-derived)
  simknn_gip    : GIP_mi(train) x DSSM-weighted votes   (train + static DSSM)
  simknn_misim  : same, miRNA sim = static MISIM        (LEGACY-LEAKY)
  simknn_j      : simknn_gip + lambda * J·tanh(z)        (train + static DSSM)

Usage: python3 -u run_m2_baseline.py [comma-model-list]
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = '/home/ubuntu/repos/DHGCMDA-fork'
sys.path.insert(0, os.path.join(REPO, 'forensics'))
from eval_top1 import build_pair_folds, mask_pair, evaluate_split

torch.manual_seed(0)
np.random.seed(0)

T = pd.read_csv(os.path.join(REPO, 'v3.2_spld_paper/triplets.csv'))
M, D, K = 411, 271, 5
Y = np.zeros((M, D, K))
for _, r in T.iterrows():
    Y[int(r['mi']), int(r['di']), int(r['type'])] = 1
A = (Y.sum(2) > 0).astype(int)
DSSM = pd.read_csv(os.path.join(REPO, 'HMDD_data/MDAv3.2-3/DSSM3.2_3.csv'),
                   index_col=0).values.astype(np.float32)
MISIM = pd.read_csv(os.path.join(REPO, 'HMDD_data/MDAv3.2-3/mi_fun_sim_3.2_3.csv'),
                    index_col=0).values.astype(np.float32)
folds = build_pair_folds(A)


def gip(X):
    """Gaussian interaction profile; X is [n_entities, n_other] binary."""
    n = X.shape[0]
    norms = (X * X).sum(1)
    g = n / norms.sum() if norms.sum() > 0 else 1.0
    S = X @ X.T
    diff = norms[:, None] + norms[None, :] - 2 * S
    return np.exp(-g * diff)


def j_prior(Yt, alpha=1.0):
    C = np.zeros((K, K))
    pos = np.argwhere(Yt.sum(2) > 0)
    for m, d in pos:
        ts = np.where(Yt[m, d] > 0)[0]
        for a in ts:
            for b in ts:
                if a != b:
                    C[a, b] += 1
    nk = C.sum(1, keepdims=True)
    J = (C + alpha) / (nk + alpha * (K - 1))
    np.fill_diagonal(J, 0.0)
    return J


class Factor(nn.Module):
    def __init__(self, dim, kind):
        super().__init__()
        self.kind = kind
        self.em = nn.Embedding(M, dim)
        self.ed = nn.Embedding(D, dim)
        nn.init.normal_(self.em.weight, std=0.1)
        nn.init.normal_(self.ed.weight, std=0.1)
        if kind == 'bilinear':
            self.W = nn.Parameter(torch.randn(K, dim, dim) * 0.05)
        else:
            self.r = nn.Embedding(K, dim)
        self.b = nn.Parameter(torch.zeros(K))

    def scores(self, mi, di):
        em, ed = self.em(mi), self.ed(di)
        if self.kind == 'bilinear':
            return torch.einsum('ni,kij,nj->nk', em, self.W, ed) + self.b
        return torch.einsum('ni,ki,ni->nk', em, self.r.weight, ed) + self.b


def fit_factor(model, tr_p, va_p, pw, epochs=400, lr=0.05):
    mi = torch.tensor([p[0] for p in tr_p])
    di = torch.tensor([p[1] for p in tr_p])
    yv = torch.tensor(np.array([Y[m, d] for m, d in tr_p]), dtype=torch.float32)
    mi_v = torch.tensor([p[0] for p in va_p])
    di_v = torch.tensor([p[1] for p in va_p])
    yv_v = torch.tensor(np.array([Y[m, d] for m, d in va_p]), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_state, best_loss, patience = None, np.inf, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model.scores(mi, di), yv)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model.scores(mi_v, di_v), yv_v).item()
        if vl < best_loss - 1e-5:
            best_loss, patience = vl, 0
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 30:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_factor(model):
    model.eval()
    with torch.no_grad():
        S = np.zeros((M, D, K))
        di = torch.arange(D)
        for m in range(M):
            mi = torch.full((D,), m, dtype=torch.long)
            S[m] = model.scores(mi, di).numpy()
    return S


def simknn_scores(Yt, IM, ID_):
    S = np.zeros((M, D, K))
    imw = IM / (np.abs(IM).sum(1, keepdims=True) + 1e-8)
    idw = ID_ / (np.abs(ID_).sum(1, keepdims=True) + 1e-8)
    for k in range(K):
        S[:, :, k] = imw @ Yt[:, :, k] + (idw @ Yt[:, :, k].T).T
    return S


MODELS = sys.argv[1].split(',') if len(sys.argv) > 1 else [
    'cooccur_prior', 'distmult', 'bilinear', 'simknn_gip', 'simknn_j']
LAMBDAS = [0.0, 0.2, 0.5]
DIM = 32
results = {m: {'folds': []} for m in MODELS}
t_all = time.time()

for k, test in enumerate(folds):
    t0 = time.time()
    test_set = set(zip(test[0].tolist(), test[1].tolist()))
    train_pairs = [(m, d) for m in range(M) for d in range(D)
                   if A[m, d] and (m, d) not in test_set]
    rng = np.random.RandomState(k)
    rng.shuffle(train_pairs)
    n_in = int(0.8 * len(train_pairs))
    tr_p, va_p = train_pairs[:n_in], train_pairs[n_in:]
    va_idx = (np.array([p[0] for p in va_p]), np.array([p[1] for p in va_p]))
    Yt = mask_pair(Y, test[0], test[1])
    trainA = (Yt.sum(2) > 0).astype(np.float32)
    GIP_mi = gip(trainA)
    GIP_d = gip(trainA.T)
    ID_ = np.where(DSSM == 0, GIP_d, DSSM)
    J = j_prior(Yt)
    cnt = Yt.reshape(-1, K).sum(0)
    pw = torch.tensor((cnt.sum() - cnt) / (cnt + 1), dtype=torch.float32)

    for name in MODELS:
        prov = {'cooccur_prior': 'train', 'distmult': 'train', 'bilinear': 'train',
                'simknn_gip': 'train+static_dssm', 'simknn_j': 'train+static_dssm',
                'simknn_misim': 'LEAKY(misim_static)'}[name]
        if name == 'cooccur_prior':
            marg = Yt.reshape(-1, K).mean(0)
            S = np.tile(marg, (M, D, 1))
        elif name in ('distmult', 'bilinear'):
            mdl = Factor(DIM, name)
            mdl = fit_factor(mdl, tr_p, va_p, pw)
            S = predict_factor(mdl)
        elif name == 'simknn_gip':
            S = simknn_scores(Yt, GIP_mi, ID_)
        elif name == 'simknn_misim':
            IM_l = np.where(MISIM == 0, GIP_mi, MISIM)
            S = simknn_scores(Yt, IM_l, ID_)
        elif name == 'simknn_j':
            S0 = simknn_scores(Yt, GIP_mi, ID_)
            best_f1, S = -1.0, S0
            for lam in LAMBDAS:
                Sj = S0 + lam * np.tensordot(np.tanh(S0), J.T, axes=([2], [0]))
                met = evaluate_split(Sj, Y, va_idx)
                if met['legacy_f1'] > best_f1:
                    best_f1, S = met['legacy_f1'], Sj
        met = evaluate_split(S, Y, test)
        met = {k2: v for k2, v in met.items() if not k2.startswith('_')}
        met['prov'] = prov
        met['elapsed_s'] = round(time.time() - t0, 1)
        results[name]['folds'].append(met)
        print(f"fold{k} {name}: F1={met['legacy_f1']:.4f} "
              f"P={met['legacy_top1_precision']:.4f} "
              f"miR={met['legacy_micro_recall']:.4f} "
              f"maR={met['legacy_macro_recall']:.4f} "
              f"Hit@1={met['Hit@1']:.4f} [{prov}]", flush=True)

out = {}
for name in MODELS:
    fs = results[name]['folds']
    keys = ['legacy_top1_precision', 'legacy_micro_recall', 'legacy_macro_recall',
            'legacy_f1', 'macro_AUPR', 'micro_AUPR', 'Hit@1', 'Recall@3',
            'micro_F1@0.5', 'macro_F1@0.5']
    agg = {key: float(np.nanmean([f[key] for f in fs])) for key in keys}
    out[name] = {'aggregate': agg, 'folds': fs}
    print(f"== {name}: {agg}", flush=True)

json.dump(out, open(os.path.join(REPO, 'forensics/m2_baseline_results.json'), 'w'),
          indent=2, default=float)
print("total", round(time.time() - t_all, 1), "s")
