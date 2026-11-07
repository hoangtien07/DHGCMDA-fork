"""Paper-faithful DHGCMDA (Sun et al., BMC Bioinformatics 2026) re-implementation.

Rule: where the paper specifies, follow the paper; where it does not, keep the author's released code
(CDMBlab/DHGCMDA) behaviour. Every deviation from the released code is listed in DEVIATIONS below.
"""
import argparse, json, math, os, sys, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HGTConv
from sklearn import metrics as skm

DEVIATIONS = {
    'leakage': 'association matrix used in features/graphs/losses = training pairs only (released code used the full matrix incl. test pairs)',
    'gip': 'miRNA view-1 (released file M_GSM = GIP of full HMDD v2.0 matrix) recomputed per fold from A_train',
    'views': 'branch v receives its own features X^(v)=[S^(v),A] (released code fed view-1 features to both branches) [Eq.1-2]',
    'knn': 'KNN hyperedges by cosine similarity, e_i = {v_i} U KNN(v_i,K) [Sec 2.3] (released: Euclidean)',
    'avf': 'attention gate + learnable beta [Eq.16-18] (released: fixed 0.6/0.4 sum)',
    'recon': 'Z_proj=W2 ReLU(W1 Z), S^=Z_proj Z_proj^T, L=||S-S^||_F^2 [Eq.19-21], lambda3=1 (released: cosine decoder, 0.15*MSE)',
    'dyn_update': 'every 5 epochs: S^ min-max to [0,1], edges S^>0.5, hypergraphs rebuilt by KNN on [S^,A] [Sec 2.8] (released: every epoch if MSE change>0.01, hypergraphs never rebuilt)',
    'loss': 'L = L_type + 1.0 L_intra + 0.3 L_inter + 1.0 L_recon [Eq.32] (+ released existence loss, paper does not define one); no label smoothing, no extra L2 term',
    'type_loss': 'multi-label weighted CE over all known types of a pair [Eq.33], effective-number weights from training counts [Eq.34]',
    'hgt': '2 layers, 4 heads, x=LayerNorm(x+Dropout(HGT(x))) [Eq.29] (released default 8 heads)',
    'optim': 'Adam [Sec 2.9] (released AdamW); early stopping on a validation split carved from training data',
    'protocol': 'CV_triplet and CV_type as two separate 5-fold protocols [Sec 3.2] (released: one pair split, 10% dropped)',
    'types': 'number of association types C read from data (released: hardcoded 4)',
}


def P():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a('--data', default='v2.0'); a('--data_dir', default=os.path.expanduser('~/dhg_runs/faithful/data'))
    a('--protocol', choices=['type', 'triplet'], default='type'); a('--seed', type=int, default=1234)
    a('--folds', type=int, default=5); a('--only_fold', type=int, default=-1)
    a('--epochs', type=int, default=650); a('--K', type=int, default=13)
    a('--hidden', type=int, default=256); a('--proj', type=int, default=64)
    a('--heads', type=int, default=4); a('--nlayer', type=int, default=2); a('--dropout', type=float, default=0.3)
    a('--t', type=float, default=0.5); a('--lam1', type=float, default=1.0); a('--lam2', type=float, default=0.3)
    a('--lam3', type=float, default=1.0); a('--margin', type=float, default=0.5); a('--margin_w', type=float, default=0.1)
    a('--exist_w', type=float, default=0.3); a('--neg_w', type=float, default=0.5); a('--beta', type=float, default=0.99999)
    a('--lr', type=float, default=1e-4); a('--wd', type=float, default=1e-5); a('--neg_ratio', type=int, default=10)
    a('--update_every', type=int, default=5); a('--theta', type=float, default=0.5)
    a('--early_stop', type=int, default=1); a('--val_frac', type=float, default=0.1)
    a('--eval_every', type=int, default=10); a('--patience', type=int, default=10)
    a('--ablation', default='none', choices=['none', 'no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv'])
    a('--recon_reduction', choices=['sum', 'mean'], default='sum')  # sum = paper Eq.21 ||.||_F^2; mean = sensitivity only
    a('--diag_leak', type=int, default=0)  # DIAGNOSTIC ONLY: full association matrix in features/graphs (released-code leakage)
    a('--threads', type=int, default=2); a('--out', default='')
    return p.parse_args()


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def gip(A):
    sq = (A ** 2).sum(1); g = 1.0 / max(sq.mean(), 1e-12)
    return np.exp(-g * np.maximum(sq[:, None] + sq[None] - 2 * A @ A.T, 0)).astype(np.float32)


def knn_G(X, K):
    """Sec 2.3: cosine KNN hyperedges incl. centre node; Eq.4 G = Dv^-1/2 H W De^-1 H^T Dv^-1/2, W=I."""
    X = np.asarray(X, np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, np.inf)
    n = len(X); H = np.zeros((n, n), np.float32)
    nb = np.argsort(-S, axis=1)[:, :K + 1]  # first is self (inf)
    for i in range(n):
        H[nb[i], i] = 1.0
    dv = H.sum(1); de = H.sum(0)
    Dv = np.diag(dv ** -0.5); De = np.diag(1.0 / de)
    return torch.tensor(Dv @ H @ De @ H.T @ Dv, dtype=torch.float32)


def gcn_G(X, K):
    """w/o HGCN: standard GCN propagation on the pairwise cosine-KNN graph, D^-1/2 (A+I) D^-1/2."""
    X = np.asarray(X, np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    n = len(X); A = np.zeros((n, n))
    nb = np.argsort(-S, axis=1)[:, :K]
    A[np.repeat(np.arange(n), K), nb.ravel()] = 1
    A = np.maximum(A, A.T) + np.eye(n)
    d = A.sum(1) ** -0.5
    return torch.tensor(d[:, None] * A * d[None], dtype=torch.float32)


class HGNNConv(nn.Module):  # released HGNN_conv: G (X Theta + b)
    def __init__(s, i, o):
        super().__init__(); s.W = nn.Parameter(torch.empty(i, o)); s.b = nn.Parameter(torch.empty(o))
        stdv = 1. / math.sqrt(o); s.W.data.uniform_(-stdv, stdv); s.b.data.uniform_(-stdv, stdv)

    def forward(s, x, G):
        return G @ (x @ s.W + s.b)


class CLHGCN(nn.Module):
    """Sec 2.4 / Eq.5-9 (matches released CL_HGCN): two HGCN branches + projection head g, GRACE-style loss."""
    def __init__(s, din, d, dp, t, alpha=0.5):
        super().__init__()
        s.h1, s.h2 = HGNNConv(din, d), HGNNConv(din, d)
        s.act = nn.LeakyReLU(0.25); s.fc1, s.fc2 = nn.Linear(d, dp), nn.Linear(dp, d); s.t, s.alpha = t, alpha

    def g(s, z):
        return s.fc2(F.elu(s.fc1(z)))

    def l_cl(s, u, v):
        f = lambda x: torch.exp(x / s.t)
        un, vn = F.normalize(u, dim=1), F.normalize(v, dim=1)
        refl, betw = f(un @ un.T), f(un @ vn.T)
        return -torch.log(betw.diag() / (refl.sum(1) + betw.sum(1) - refl.diag() + 1e-8)).mean()

    def forward(s, x1, G1, x2, G2, single=False):
        z1 = s.act(s.h1(x1, G1))
        if single:
            return z1, None, torch.zeros((), device=x1.device)
        z2 = s.act(s.h2(x2, G2))
        h1, h2 = s.g(z1), s.g(z2)
        return z1, z2, s.alpha * s.l_cl(h1, h2) + (1 - s.alpha) * s.l_cl(h2, h1)


def inter_loss(zm, zd, A, t, margin, margin_w):
    """Eq.10-15 (released InterViewContrastiveLoss logic, vectorised; margin weight 0.1 as released)."""
    S = F.normalize(zm, dim=1) @ F.normalize(zd, dim=1).T / t
    pos = A > 0; neg = ~pos
    E = torch.exp(torch.clamp(S, -10, 10))
    negsum = (E * neg).sum(1, keepdim=True)
    L = -torch.log(E / (E + negsum + 1e-8) + 1e-8)
    info = L[pos].mean()
    ps, ns = S[pos], S[neg]
    ns = ns[torch.randperm(len(ns), device=S.device)[:min(len(ns), 10 * len(ps))]]
    return info + margin_w * F.relu(margin - (ps.mean() - ns.mean()))


class AVF(nn.Module):
    """Eq.16-18: alpha_v = sigmoid(W2 ReLU(W1 GAP(Z_v))); Z~_v = ReLU(alpha_v * Z_v); Z = sum_v beta_v Z~_v.
    Channel-wise gate (reduction 4) and softmax-normalised beta are not specified by the paper [assumption]."""
    def __init__(s, d, r=4):
        super().__init__(); s.W1, s.W2 = nn.Linear(d, d // r), nn.Linear(d // r, d); s.beta = nn.Parameter(torch.zeros(2))

    def forward(s, zs):
        b = torch.softmax(s.beta, 0)
        return sum(b[v] * F.relu(torch.sigmoid(s.W2(F.relu(s.W1(z.mean(0))))) * z) for v, z in enumerate(zs))


class Recon(nn.Module):  # Eq.19-20
    def __init__(s, d):
        super().__init__(); s.W1, s.W2 = nn.Linear(d, d), nn.Linear(d, d // 2)

    def forward(s, z):
        zp = s.W2(F.relu(s.W1(z)))
        return zp @ zp.T


class Predictor(nn.Module):
    """Released SimplifiedTypePredictor (diagonal bilinear, existence + C types), C generalised."""
    def __init__(s, din, h, C, p=0.2):
        super().__init__()
        s.pm = nn.Sequential(nn.Linear(din, h), nn.LayerNorm(h), nn.Dropout(p))
        s.pd = nn.Sequential(nn.Linear(din, h), nn.LayerNorm(h), nn.Dropout(p))
        sc = math.sqrt(2.0 / h)
        q, _ = torch.linalg.qr(torch.randn(C, h).T)
        s.rt = nn.Parameter(q.T[:C] * sc); s.re = nn.Parameter(torch.randn(h) * sc); s.temp = nn.Parameter(torch.tensor(2.0))

    def forward(s, zm, zd):
        m, d = s.pm(zm), s.pd(zd)
        ex = torch.sigmoid((m * s.re) @ d.T)
        logits = torch.einsum('ih,kh,jh->ijk', m, s.rt, d)
        return ex, torch.softmax(logits / torch.clamp(s.temp, 0.5, 5.0), 2)


class HGTBlock(nn.Module):  # Eq.22-29
    def __init__(s, d, meta, heads, p):
        super().__init__(); s.conv = HGTConv(d, d, meta, heads); s.ln = nn.ModuleDict({k: nn.LayerNorm(d) for k in meta[0]}); s.dp = nn.Dropout(p)

    def forward(s, x, ei):
        o = s.conv(x, ei)
        return {k: s.ln[k](x[k] + s.dp(o[k])) for k in x}


META = (['miRNA', 'disease'], [('miRNA', 'associates', 'disease'), ('disease', 'associates', 'miRNA'),
                               ('miRNA', 'similar', 'miRNA'), ('disease', 'similar', 'disease')])


class DHGCMDA(nn.Module):
    def __init__(s, nm, nd, C, a):
        super().__init__(); s.a = a; d = a.hidden
        s.clm, s.cld = CLHGCN(nm + nd, d, a.proj, a.t), CLHGCN(nd + nm, d, a.proj, a.t)
        fd = 2 * d if a.ablation == 'no_avf' else d
        if a.ablation != 'no_avf':
            s.avf_m, s.avf_d = AVF(d), AVF(d)
        s.rec_m, s.rec_d = Recon(fd), Recon(fd)
        s.inp = nn.ModuleDict({'miRNA': nn.Linear(fd, d), 'disease': nn.Linear(fd, d)})
        s.hgt = nn.ModuleList([HGTBlock(d, META, a.heads, a.dropout) for _ in range(a.nlayer)])
        s.pred = Predictor(d, 128, C)

    def forward(s, X, G, ei):
        single = s.a.ablation == 'no_dv'
        zm1, zm2, lm = s.clm(X['m1'], G['m1'], X['m2'], G['m2'], single)
        zd1, zd2, ld = s.cld(X['d1'], G['d1'], X['d2'], G['d2'], single)
        if single:
            zm, zd = zm1, zd1
        elif s.a.ablation == 'no_avf':
            zm, zd = torch.cat([zm1, zm2], 1), torch.cat([zd1, zd2], 1)
        else:
            zm, zd = s.avf_m([zm1, zm2]), s.avf_d([zd1, zd2])
        Sm, Sd = s.rec_m(zm), s.rec_d(zd)
        x = {'miRNA': s.inp['miRNA'](zm), 'disease': s.inp['disease'](zd)}
        if s.a.ablation != 'no_hgt':
            for blk in s.hgt:
                x = blk(x, ei)
        ex, tp = s.pred(x['miRNA'], x['disease'])
        return ex, tp, lm + ld, zm, zd, Sm, Sd


def edges(A, Sm, Sd, theta):
    md = torch.nonzero(A > 0).T
    em = torch.nonzero((Sm > theta) & ~torch.eye(len(Sm), dtype=torch.bool)).T
    ed = torch.nonzero((Sd > theta) & ~torch.eye(len(Sd), dtype=torch.bool)).T
    return {META[1][0]: md, META[1][1]: md.flip(0), META[1][2]: em, META[1][3]: ed}


def mm01(S):
    return (S - S.min()) / (S.max() - S.min() + 1e-12)


def top1(Ytrue, Pt):
    """Released Calculate_Metrics.compute_top1_metrics definition (P = fraction correct, R = mean per-type recall,
    F1 = 2PR/(P+R)), generalised to C types; multi-type pair counts as correct if argmax is one of its types."""
    pred = Pt.argmax(1)
    hit = Ytrue[np.arange(len(pred)), pred] > 0
    Pp = hit.mean()
    rec = [(pred[Ytrue[:, k] > 0] == k).mean() for k in range(Ytrue.shape[1]) if (Ytrue[:, k] > 0).any()]
    R = float(np.mean(rec)); F1 = 2 * Pp * R / (Pp + R) if Pp + R > 0 else 0.0
    return dict(top1_precision=float(Pp), top1_recall=R, top1_f1=float(F1), n=int(len(pred)),
                per_type_recall=[float(r) for r in rec], pred_hist=np.bincount(pred, minlength=Ytrue.shape[1]).tolist())


def binary(y, s):
    """Released Calculate_Metrics binary metrics: AUC, AUPR=auc(recall,precision), F1 at the F1-maximising threshold."""
    fpr, tpr, _ = skm.roc_curve(y, s); auc = skm.auc(fpr, tpr)
    pr, rc, th = skm.precision_recall_curve(y, s); aupr = skm.auc(rc, pr)
    f1 = (2 * pr * rc / np.maximum(pr + rc, 1e-12)).max()
    return dict(auc=float(auc), aupr=float(aupr), f1=float(f1), n=int(len(y)))


def make_splits(Y, a, rng):
    """Sec 3.2. Returns list of dicts with boolean masks over triplets (nm,nd,C) or pairs (nm,nd)."""
    nm, nd, C = Y.shape
    folds = []
    if a.protocol == 'type':
        pairs = np.argwhere(Y.sum(2) > 0); rng.shuffle(pairs)
        for f, test in enumerate(np.array_split(pairs, a.folds)):
            rest = np.concatenate([p for g, p in enumerate(np.array_split(pairs, a.folds)) if g != f])
            nv = int(round(a.val_frac * len(rest))) if a.early_stop else 0
            folds.append(dict(test=test, val=rest[:nv], train=rest[nv:]))
    else:
        tri = np.argwhere(Y > 0); rng.shuffle(tri)
        unk = np.argwhere(Y == 0); rng.shuffle(unk)
        parts = np.array_split(tri, a.folds)
        for f, test in enumerate(parts):
            rest = np.concatenate([p for g, p in enumerate(parts) if g != f])
            nv = int(round(a.val_frac * len(rest))) if a.early_stop else 0
            # equal number of unknown triplets for test (and for validation), disjoint
            tneg = unk[f * len(test):(f + 1) * len(test)] if (f + 1) * len(test) <= len(unk) else unk[rng.choice(len(unk), len(test), False)]
            vneg = unk[rng.choice(len(unk), nv, False)] if nv else unk[:0]
            folds.append(dict(test=test, test_neg=tneg, val=rest[:nv], val_neg=vneg, train=rest[nv:]))
    return folds


def run_fold(D, fold, a, dev, log):
    Y = D['Y']; nm, nd, C = Y.shape
    Ytr = np.zeros_like(Y)
    if a.protocol == 'type':
        Ytr[fold['train'][:, 0], fold['train'][:, 1]] = Y[fold['train'][:, 0], fold['train'][:, 1]]
    else:
        Ytr[tuple(fold['train'].T)] = 1
    A = (Ytr.sum(2) > 0).astype(np.float32)
    Af = (Y.sum(2) > 0).astype(np.float32) if a.diag_leak else A
    sims = {}
    for k, T in [('sim_m1', Af), ('sim_m2', Af), ('sim_d1', Af.T), ('sim_d2', Af.T)]:
        v = D[k]; sims[k] = gip(T) if v.ndim == 0 else v.astype(np.float32)
    X = {'m1': np.hstack([sims['sim_m1'], Af]), 'm2': np.hstack([sims['sim_m2'], Af]),
         'd1': np.hstack([sims['sim_d1'], Af.T]), 'd2': np.hstack([sims['sim_d2'], Af.T])}
    build = gcn_G if a.ablation == 'no_hgcn' else knn_G
    G = {k: build(v, a.K).to(dev) for k, v in X.items()}
    X = {k: torch.tensor(v).to(dev) for k, v in X.items()}
    At = torch.tensor(A).to(dev); Yt = torch.tensor(Ytr).to(dev)
    S_M, S_D = torch.tensor(sims['sim_m2']).to(dev), torch.tensor(sims['sim_d2']).to(dev)  # recon targets (released choice)
    Aft = torch.tensor(Af).to(dev)
    ei = edges(Aft, S_M, S_D, a.theta)
    # training negatives for existence: pairs unknown in training, excluding held-out test/val pairs
    excl = np.zeros((nm, nd), bool)
    for k in ['test', 'val', 'test_neg', 'val_neg']:
        if k in fold and len(fold[k]):
            excl[fold[k][:, 0], fold[k][:, 1]] = True
    neg_pool = np.argwhere((A == 0) & ~excl)
    pos = torch.tensor(np.argwhere(A > 0)).to(dev)
    rng = np.random.default_rng(a.seed)
    neg = torch.tensor(neg_pool[rng.choice(len(neg_pool), min(len(neg_pool), a.neg_ratio * len(pos)), False)]).to(dev)
    n_k = Ytr.sum((0, 1)); w = (1 - a.beta) / (1 - a.beta ** np.maximum(n_k, 1)); w = w * C / w.sum()
    w = torch.tensor(w, dtype=torch.float32).to(dev)
    lam1, lam2 = (0.0, 0.0) if a.ablation == 'no_cl' else (a.lam1, a.lam2)
    model = DHGCMDA(nm, nd, C, a).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def evaluate(split, negs=None):
        model.eval()
        with torch.no_grad():
            ex, tp, *_ = model(X, G, ei)
        model.train(); ex, tp = ex.cpu().numpy(), tp.cpu().numpy()
        if a.protocol == 'type':
            return top1(Y[split[:, 0], split[:, 1]], tp[split[:, 0], split[:, 1]])
        sc = lambda t: ex[t[:, 0], t[:, 1]] * tp[t[:, 0], t[:, 1], t[:, 2]]  # triplet score [assumption]
        return binary(np.r_[np.ones(len(split)), np.zeros(len(negs))], np.r_[sc(split), sc(negs)])

    key = 'top1_f1' if a.protocol == 'type' else 'aupr'
    best, best_state, bad, best_ep = -1, None, 0, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        ex, tp, lintra, zm, zd, Sm, Sd = model(X, G, ei)
        p_s, n_s = ex[pos[:, 0], pos[:, 1]], ex[neg[:, 0], neg[:, 1]]
        l_ex = ((-(1 - p_s) ** 2 * torch.log(p_s + 1e-7)).sum() + a.neg_w * (-(n_s ** 2) * torch.log(1 - n_s + 1e-7)).sum()) / (len(p_s) + len(n_s))
        yp = Yt[pos[:, 0], pos[:, 1]]; tpp = tp[pos[:, 0], pos[:, 1]]
        l_type = -(w * yp * torch.log(tpp + 1e-9)).sum(1).mean()
        l_inter = inter_loss(zm, zd, At, a.t, a.margin, a.margin_w) if lam2 > 0 else torch.zeros((), device=dev)
        l_rec = ((S_M - Sm) ** 2).sum() + ((S_D - Sd) ** 2).sum() if a.recon_reduction == 'sum' else ((S_M - Sm) ** 2).mean() + ((S_D - Sd) ** 2).mean()
        loss = l_type + lam1 * lintra + lam2 * l_inter + a.lam3 * l_rec + a.exist_w * l_ex
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % a.update_every == 0:  # Sec 2.8
            with torch.no_grad():
                Smn, Sdn = mm01(Sm.detach()), mm01(Sd.detach())
                ei = edges(Aft, Smn, Sdn, a.theta)
                Hm = build(np.hstack([Smn.cpu().numpy(), Af]), a.K).to(dev)
                Hd = build(np.hstack([Sdn.cpu().numpy(), Af.T]), a.K).to(dev)
                G = {'m1': Hm, 'm2': Hm, 'd1': Hd, 'd2': Hd}
        if ep % 50 == 0 or ep == 1:
            log(f'  ep {ep} loss {loss.item():.4f} type {l_type.item():.4f} ex {l_ex.item():.4f} intra {lintra.item():.4f} '
                f'inter {l_inter.item():.4f} rec {l_rec.item():.2f} mm_edges {ei[META[1][2]].shape[1]} dd_edges {ei[META[1][3]].shape[1]} {time.time()-t0:.0f}s')
        if a.early_stop and ep % a.eval_every == 0:
            v = evaluate(fold['val'], fold.get('val_neg'))[key]
            if v > best:
                best, bad, best_ep = v, 0, ep
                best_state = (copy.deepcopy(model.state_dict()), {k: x.clone() for k, x in G.items()}, {k: x.clone() for k, x in ei.items()})
            else:
                bad += 1
                if bad >= a.patience:
                    log(f'  early stop at ep {ep}, best val {key}={best:.4f} @ep {best_ep}'); break
    if best_state is not None:
        model.load_state_dict(best_state[0]); G, ei = best_state[1], best_state[2]
    res = evaluate(fold['test'], fold.get('test_neg'))
    res.update(best_epoch=best_ep, best_val=best, train_time_s=round(time.time() - t0, 1))
    return res


def main():
    a = P(); torch.set_num_threads(a.threads); seed_all(a.seed); dev = torch.device('cpu')
    Dz = np.load(os.path.join(a.data_dir, a.data + '.npz'), allow_pickle=True); D = {k: Dz[k] for k in Dz.files}
    out = a.out or os.path.expanduser(f'~/dhg_runs/faithful/results/{a.data}_{a.protocol}_{a.ablation}_s{a.seed}' + ('_rmean' if a.recon_reduction == 'mean' else '') + ('_LEAK' if a.diag_leak else '') + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = lambda m: print(m, flush=True)
    log(f'args {vars(a)}'); log(f'data {a.data} Y{D["Y"].shape} triplets {int(D["Y"].sum())}')
    folds = make_splits(D['Y'], a, np.random.default_rng(a.seed))
    res = []
    for f, fold in enumerate(folds):
        if a.only_fold >= 0 and f != a.only_fold:
            continue
        seed_all(a.seed + f)
        r = run_fold(D, fold, a, dev, log); r['fold'] = f; res.append(r)
        log(f'FOLD {f} ' + json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}))
    keys = ['top1_precision', 'top1_recall', 'top1_f1'] if a.protocol == 'type' else ['auc', 'aupr', 'f1']
    summ = {k: [float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))] for k in keys}
    log('SUMMARY ' + json.dumps(summ))
    json.dump(dict(args=vars(a), deviations=DEVIATIONS, folds=res, summary=summ), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
