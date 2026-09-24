"""run_final_suite.py — Final Fairness Suite on MDAv3.2-3.

Ablation variants of the similarity-vote baseline (all leakage-audited),
each evaluated on the golden SPLD pair-fold protocol with the full metric set:
  gip_only    : score = imw @ Y_k                  (miRNA-side vote)
  dssm_only   : score = idw @ Y_k.T                (disease-side vote)
  gip_dssm    : both votes (== simknn_gip)
  gip_dssm_j  : both + lambda*J(tanh)              (label interaction)
  fmisim_dssm : fold-MISIM(mi-vote) + DSSM(dis-vote)  (honest MISIM counterpart)
  gip_gip     : GIP on both sides (no DSSM at all)
  misim_dssm  : static MISIM + DSSM                (LEAKY control)

Also: repeated pair-CV with seeds 1,2 for simknn_gip (secondary robustness;
SPLD seed-0 folds remain primary).

Writes forensics/final_suite_results.json.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

REPO = '/home/ubuntu/repos/DHGCMDA-fork'
sys.path.insert(0, os.path.join(REPO, 'forensics'))
from eval_top1 import build_pair_folds, mask_pair, evaluate_split
from fold_misim import functional_sim

REPO_D = os.path.join(REPO, 'v3.2_spld_paper')
T = pd.read_csv(os.path.join(REPO_D, 'triplets.csv'))
M, D, K = 411, 271, 5
Y = np.zeros((M, D, K))
for _, r in T.iterrows():
    Y[int(r['mi']), int(r['di']), int(r['type'])] = 1
A = (Y.sum(2) > 0).astype(int)
DSSM = pd.read_csv(os.path.join(REPO, 'HMDD_data/MDAv3.2-3/DSSM3.2_3.csv'),
                   index_col=0).values.astype(np.float32)
MISIM = pd.read_csv(os.path.join(REPO, 'HMDD_data/MDAv3.2-3/mi_fun_sim_3.2_3.csv'),
                    index_col=0).values.astype(np.float32)


def gip(X):
    n = X.shape[0]
    nm = (X * X).sum(1)
    g = n / nm.sum() if nm.sum() > 0 else 1.0
    S = X @ X.T
    diff = nm[:, None] + nm[None, :] - 2 * S
    return np.exp(-g * diff)


def rn(S):
    return S / (np.abs(S).sum(1, keepdims=True) + 1e-8)


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


def vote(Yt, IM, ID_, two=True, lam=0.0, J=None):
    imw, idw = rn(IM), rn(ID_)
    S = np.zeros((M, D, K))
    for c in range(K):
        v = imw @ Yt[:, :, c]
        if two:
            v = v + (idw @ Yt[:, :, c].T).T
        S[:, :, c] = v
    if lam and J is not None:
        S = S + lam * np.tensordot(np.tanh(S), J.T, axes=([2], [0]))
    return S


VARIANTS = sys.argv[1].split(',') if len(sys.argv) > 1 else [
    'gip_only', 'dssm_only', 'gip_dssm', 'gip_dssm_j', 'fmisim_dssm',
    'gip_gip', 'misim_dssm']
MET_KEYS = ['legacy_top1_precision', 'legacy_micro_recall', 'legacy_macro_recall',
            'legacy_f1', 'macro_AUPR', 'micro_AUPR', 'macro_AUC', 'micro_AUC',
            'micro_F1@0.5', 'macro_F1@0.5', 'Hit@1', 'Recall@3']


def run_variant(name, folds, seed_tag):
    fs = []
    for k, test in enumerate(folds):
        t0 = time.time()
        Yt = mask_pair(Y, test[0], test[1])
        tA = (Yt.sum(2) > 0).astype(np.float32)
        Gm, Gd = gip(tA), gip(tA.T)
        ID_ = np.where(DSSM == 0, Gd, DSSM)
        J = j_prior(Yt)
        if name == 'gip_only':
            S = vote(Yt, Gm, ID_, two=False)
        elif name == 'dssm_only':
            S = np.zeros((M, D, K))
            idw = rn(ID_)
            for c in range(K):
                S[:, :, c] = (idw @ Yt[:, :, c].T).T
        elif name == 'gip_dssm':
            S = vote(Yt, Gm, ID_)
        elif name == 'gip_dssm_j':
            S = vote(Yt, Gm, ID_, lam=0.2, J=J)
        elif name == 'fmisim_dssm':
            FS = functional_sim(tA, DSSM)
            IM = np.where(FS == 0, Gm, FS)
            S = vote(Yt, IM, ID_)
        elif name == 'gip_gip':
            S = vote(Yt, Gm, Gd)
        elif name == 'misim_dssm':
            IM = np.where(MISIM == 0, Gm, MISIM)
            S = vote(Yt, IM, ID_)
        met = evaluate_split(S, Y, test)
        met['_hit_vec'] = met['_hit_vec'].tolist()
        met['_rec_vec'] = met['_rec_vec'].tolist()
        met['elapsed_s'] = round(time.time() - t0, 1)
        fs.append(met)
        print(f"{seed_tag} fold{k} {name}: F1={met['legacy_f1']:.4f} "
              f"P={met['legacy_top1_precision']:.4f} maR={met['legacy_macro_recall']:.4f} "
              f"macroAUPR={met['macro_AUPR']:.4f}", flush=True)
    agg = {key: float(np.nanmean([f[key] for f in fs])) for key in MET_KEYS}
    return {'aggregate': agg, 'folds': fs}


out = {}
folds0 = build_pair_folds(A, 5, 0)
for name in VARIANTS:
    out[name] = {'seed0': run_variant(name, folds0, 's0')}
# secondary robustness: seeds 1,2 for the canonical vote
if 'gip_dssm' in VARIANTS:
    for s in (1, 2):
        out['gip_dssm'][f'seed{s}'] = run_variant('gip_dssm',
                                                 build_pair_folds(A, 5, s), f's{s}')
json.dump(out, open(os.path.join(REPO, 'forensics/final_suite_results.json'), 'w'),
          indent=2)
print("done", os.path.join(REPO, 'forensics/final_suite_results.json'))
