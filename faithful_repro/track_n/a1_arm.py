# -*- coding: utf-8 -*-
"""Track N A1 arm — exactly one: weighted multi-label sigmoid-BCE
MLP on train-only similarity-row inputs (deviation from draft's
"U1 embedding inputs": producing U1 embeddings would itself be an
unscheduled training run; similarity rows are the closest
train-only-inductive input — recorded in GATE_DECISION).

PU auxiliary OFF (council lock). Config frozen here; threshold via
inner-val only. Sanity cell: pair track fold0 seed1234.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

TENSOR = ROOT / 'data' / \
    'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
EVALDIR = ROOT / 'independent_v32_outer_refit_20261117' / 'evaluation'
OUT = HERE / 'runs_g2' / 'pair' / 'fold0_seed1234' / 'a1_sanity'
CFG = dict(hidden=256, epochs=300, lr=1e-3, wd=1e-4,
           beta_en=0.99999, inner_frac=0.10, seed=1234, fold=0)
TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def en_weights(y, beta):
    """Effective-number weights per type (Cui et al.): (1-β)/(1-β^n)."""
    n = y.sum(0)
    eff = (1 - beta) / (1 - np.power(beta, np.maximum(n, 1)))
    return torch.tensor(eff / eff.mean(), dtype=torch.float32)


def feats(Sm, Sd, pairs):
    i, j = pairs[:, 0], pairs[:, 1]
    return np.concatenate([Sm[i], Sd[j]], axis=1)


def main():
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(np.float32)
    Sm = (z['sim_m1'] + z['sim_m2']) / 2
    Sd = (z['sim_d1'] + z['sim_d2']) / 2
    f = np.load(EVALDIR / 'fold0_seed1234_static.npz',
                allow_pickle=True)
    tep = f['pairs']
    teset = set(map(tuple, tep.tolist()))
    trp = np.array([p for p in np.argwhere(Y.sum(-1) > 0)
                    if tuple(p) not in teset])
    rng = np.random.default_rng(CFG['seed'])
    perm = rng.permutation(len(trp))
    niv = int(CFG['inner_frac'] * len(trp))
    iv, ftr = perm[:niv], perm[niv:]

    Xtr = torch.tensor(feats(Sm, Sd, trp[ftr]))
    ytr = torch.tensor(Y[trp[ftr, 0], trp[ftr, 1]])
    Xiv = torch.tensor(feats(Sm, Sd, trp[iv]))
    yiv = torch.tensor(Y[trp[iv, 0], trp[iv, 1]])
    Xte = torch.tensor(feats(Sm, Sd, tep))
    yte = Y[tep[:, 0], tep[:, 1]]

    w = en_weights(ytr.numpy(), CFG['beta_en'])
    net = nn.Sequential(nn.Linear(Xtr.shape[1], CFG['hidden']),
                        nn.ReLU(),
                        nn.Linear(CFG['hidden'], 5))
    opt = torch.optim.Adam(net.parameters(), lr=CFG['lr'],
                           weight_decay=CFG['wd'])
    bce = nn.BCEWithLogitsLoss(pos_weight=w)
    gnorms = []
    for ep in range(CFG['epochs']):
        opt.zero_grad()
        loss = bce(net(Xtr), ytr)
        loss.backward()
        gn = sum(p.grad.norm().item() for p in net.parameters())
        gnorms.append(gn)
        if not np.isfinite(loss.item()) or not np.isfinite(gn):
            raise RuntimeError(f'non-finite at ep{ep}')
        opt.step()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({'sd': net.state_dict(), 'cfg': CFG},
               OUT / 'ckpt.pt')
    # checkpoint integrity: reload must reproduce outputs
    net2 = nn.Sequential(nn.Linear(Xtr.shape[1], CFG['hidden']),
                         nn.ReLU(), nn.Linear(CFG['hidden'], 5))
    net2.load_state_dict(torch.load(OUT / 'ckpt.pt')['sd'])
    assert torch.equal(net(Xte), net2(Xte))
    # tau on inner-val
    with torch.no_grad():
        p_iv = torch.sigmoid(net(Xiv)).numpy()
        p_te = torch.sigmoid(net(Xte)).numpy()
    taus = np.zeros(5)
    for k in range(5):
        best, tau = -1, 0.5
        for t in TAU_GRID:
            pr = (p_iv[:, k] >= t).astype(int)
            tp = int((pr & (yiv[:, k].numpy().astype(int)
                          == 1)).sum())
            fp = int((pr & (yiv[:, k].numpy().astype(int)
                          == 0)).sum())
            fn = int(((1 - pr) & (yiv[:, k].numpy().astype(int)
                                  == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best, tau = f1, t
        taus[k] = tau
    np.savez(OUT / 'scores.npz', test_pairs=tep, y_test=yte,
             A1_bce=p_te)
    (OUT / 'taus.json').write_text(json.dumps(
        {'A1_bce': taus.tolist()}))
    (OUT / 'train_log.json').write_text(json.dumps({
        'final_loss': float(loss.item()),
        'grad_norm_last10': gnorms[-10:],
        'taus': taus.tolist(), 'cfg': CFG}))
    print('A1 sanity done: loss', float(loss.item()),
          'taus', taus)


if __name__ == '__main__':
    main()
