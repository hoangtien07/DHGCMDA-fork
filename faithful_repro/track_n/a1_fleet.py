# -*- coding: utf-8 -*-
"""Track N G4 fleet: A1 similarity-row MLP + weighted BCE (PU OFF),
frozen CFG — 5 folds x 3 seeds, T-pair track, runs_g4/ immutable.
Reuses the approved sanity spec; inner-val tau per cell.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from a1_arm import en_weights, feats, CFG, TAU_GRID  # noqa
from run_g2c import INNER_FRAC, TENSOR, EVALDIR  # noqa

OUT = HERE / 'runs_g4'
SEEDS = (1234, 42, 2025)


def run_cell(fold, seed, Y, Sm, Sd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    z = np.load(EVALDIR / f'fold{fold}_seed1234_static.npz',
                allow_pickle=True)
    tep = z['pairs']
    teset = set(map(tuple, tep.tolist()))
    trp = np.array([p for p in np.argwhere(Y.sum(-1) > 0)
                    if tuple(p) not in teset])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trp))
    niv = int(INNER_FRAC * len(trp))
    iv, ftr = perm[:niv], perm[niv:]
    Xtr = torch.tensor(feats(Sm, Sd, trp[ftr]))
    ytr = torch.tensor(Y[trp[ftr, 0], trp[ftr, 1]],
                       dtype=torch.float32)
    Xiv = torch.tensor(feats(Sm, Sd, trp[iv]))
    yiv = Y[trp[iv, 0], trp[iv, 1]]
    Xte = torch.tensor(feats(Sm, Sd, tep))
    yte = Y[tep[:, 0], tep[:, 1]]
    w = en_weights(ytr.numpy(), CFG['beta_en'])
    net = nn.Sequential(nn.Linear(Xtr.shape[1], CFG['hidden']),
                        nn.ReLU(),
                        nn.Linear(CFG['hidden'], 5))
    opt = torch.optim.Adam(net.parameters(), lr=CFG['lr'],
                           weight_decay=CFG['wd'])
    bce = nn.BCEWithLogitsLoss(pos_weight=w)
    for ep in range(CFG['epochs']):
        opt.zero_grad()
        loss = bce(net(Xtr), ytr)
        loss.backward()
        gn = sum(p.grad.norm().item() for p in net.parameters())
        if not np.isfinite(loss.item()) or not np.isfinite(gn):
            raise RuntimeError(f'non-finite f{fold}s{seed}ep{ep}')
        opt.step()
    cdir = OUT / f'fold{fold}_seed{seed}'
    cdir.mkdir(parents=True, exist_ok=True)
    torch.save({'sd': net.state_dict(), 'cfg': CFG},
               cdir / 'ckpt.pt')
    with torch.no_grad():
        p_iv = torch.sigmoid(net(Xiv)).numpy()
        p_te = torch.sigmoid(net(Xte)).numpy()
    taus = np.zeros(5)
    for k in range(5):
        best, tau = -1, 0.5
        for t in TAU_GRID:
            pr = (p_iv[:, k] >= t).astype(int)
            tp = int((pr & (yiv[:, k] == 1)).sum())
            fp = int((pr & (yiv[:, k] == 0)).sum())
            fn = int(((1 - pr) & (yiv[:, k] == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best, tau = f1, t
        taus[k] = tau
    np.savez(cdir / 'scores.npz', test_pairs=tep, y_test=yte,
             train_pairs=trp[ftr], iv_pairs=trp[iv], y_iv=yiv,
             A1_bce=p_te)
    (cdir / 'taus.json').write_text(json.dumps(
        {'taus': {'A1_bce': taus.tolist()}}))
    (cdir / 'train_log.json').write_text(json.dumps({
        'final_loss': float(loss.item()), 'taus': taus.tolist(),
        'cfg': CFG, 'fold': fold, 'seed': seed}))
    return float(loss.item())


def main():
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(np.float32)
    Sm = (z['sim_m1'] + z['sim_m2']) / 2.0
    Sd = (z['sim_d1'] + z['sim_d2']) / 2.0
    man = {'arm': 'A1_bce', 'cfg': CFG, 'cells': []}
    for f in range(5):
        for s in SEEDS:
            # reuse approved sanity cell only if spec hashes match:
            # same CFG + same fold/seed -> run again and verify
            # result equality against exposed sanity if identical
            ls = run_cell(f, s, Y, Sm, Sd)
            sha = hashlib.sha256(
                (OUT / f'fold{f}_seed{s}' / 'scores.npz')
                .read_bytes()).hexdigest()
            man['cells'].append({'fold': f, 'seed': s,
                                 'loss': ls, 'sha256': sha})
            print(f, s, 'done', ls)
    (OUT / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
