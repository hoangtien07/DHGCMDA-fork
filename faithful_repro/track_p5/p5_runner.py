"""P5 execution adapter — consumes P5_CONFIG_LOCK.yaml and fails closed.

NOT faithful.py::main(). Differences:
- dataset + fold manifests SHA256-verified against SOURCE_MANIFEST.json
- fold manifests contribute PAIRS ONLY (their logits key is never read
  into features, training or evaluation)
- inner-validation carved deterministically from outer-train pairs only
- checkpoint selection by MULTIHOT_MACRO_F1, not top1_f1
- outer test evaluated exactly once after freeze
- diag_leak=1 rejected outright in this runner
- resolved config + all hashes emitted for every proposed run

Code-only gate G4.1: `resolve_run` is safe to call; `train_fold` runs
optimizer steps and is used ONLY on synthetic fixtures until owner
authorization.
"""
import os, sys, json, hashlib, copy, types, time
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'code'))
import faithful as fz
import p5_evaluator as ev
import p5_diagnostics as dg

REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
MANIFEST = os.path.join(_HERE, 'SOURCE_MANIFEST.json')
LOCK = os.path.join(_HERE, 'P5_CONFIG_LOCK.yaml')

# locked values the runner enforces verbatim
LOCKED_HYPERPARAMS = {
    'hidden': 256, 'proj': 64, 'heads': 4, 'nlayer': 2, 'dropout': 0.3,
    'K': 13, 't': 0.5, 'lam1': 1.0, 'lam2': 0.3, 'lam3': 1.0,
    'margin': 0.5, 'margin_w': 0.1, 'exist_w': 0.3, 'neg_w': 0.5,
    'beta': 0.99999, 'lr': 1e-4, 'wd': 1e-5, 'neg_ratio': 10,
    'update_every': 5, 'theta': 0.5, 'val_frac': 0.1,
    'eval_every': 10, 'patience': 10, 'epochs': 650, 'n_neg_draws': 10,
    'predictor': 'diag', 'recon_reduction': 'sum',
}
REJECT = {'diag_leak': 1}          # hard reject in the P5 runner
SUPPORTED_PROFILES = {'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                      'P5_SPLIT_AWARE_EXTERNAL',
                      'P5_RELEASE_K4_CONTROL'}
PROFILE_CLAIM_CEILING = {
    'P5_BENCHMARK_UNKNOWN_PROVENANCE': 'exploratory_only',
    'P5_SPLIT_AWARE_EXTERNAL': 'blocked_pending_provenance',
    'P5_RELEASE_K4_CONTROL': 'sanity_only'}


class LockViolation(Exception):
    pass


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load_lock(path=LOCK):
    cfg = yaml.safe_load(open(path))
    required = ['model', 'training', 'loss', 'dynamic_refresh', 'evaluation', 'data']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise LockViolation(f'config lock missing sections: {missing}')
    return cfg


def validate_hashes(data_path, fold_dir, seed=1234, manifest_path=MANIFEST):
    """Exact SHA validation of tensor + fold manifests; fails closed."""
    man = json.load(open(manifest_path))
    exp_data = man['artifacts']['strict_tensor']['sha256']
    got_data = _sha(data_path)
    if got_data != exp_data:
        raise LockViolation(f'tensor sha mismatch {got_data[:12]} != {exp_data[:12]}')
    folds, shas = [], {}
    exp_folds = man['artifacts']['locked_pair_folds']['files']
    for f in range(5):
        name = f'fold{f}_seed{seed}_static.npz'
        p = os.path.join(fold_dir, name)
        if name not in exp_folds:
            raise LockViolation(f'{name} not in locked manifest list')
        h = _sha(p)
        if h != exp_folds[name]:
            raise LockViolation(f'{name} sha mismatch')
        shas[name] = h
        z = np.load(p, allow_pickle=True)
        # pairs are the split definition; logits are NEVER touched
        folds.append({'test': np.asarray(z['pairs']), 'y': np.asarray(z['y'])})
    return folds, shas


def build_splits(Y, folds, val_frac, seed=1234):
    """Deterministic inner-val from outer-train pairs only; outer test
    never reaches train/checkpoint data."""
    allpos = set(map(tuple, np.argwhere(Y.sum(2) > 0).tolist()))
    for f, fold in enumerate(folds):
        test = set(map(tuple, fold['test'].tolist()))
        rest = np.array(sorted(allpos - test))
        rng = np.random.default_rng(seed * 1000 + f)
        rng.shuffle(rest)
        nv = int(round(val_frac * len(rest)))
        fold['val'] = rest[:nv]
        fold['train'] = rest[nv:]
        # hard invariant: no overlap with test
        tr = set(map(tuple, fold['train'].tolist()))
        va = set(map(tuple, fold['val'].tolist()))
        assert not (tr & test) and not (va & test) and not (tr & va)
    return folds


def args_from_lock(cfg, profile, overrides=None):
    """Map the YAML lock onto the flat args namespace faithful.py expects.
    Any lock value diverging from LOCKED_HYPERPARAMS fails closed."""
    m, t, l, dr, e = cfg['model'], cfg['training'], cfg['loss'], cfg['dynamic_refresh'], cfg['evaluation']
    resolved = {
        'hidden': m['hidden'], 'proj': m['proj'], 'heads': m['heads'],
        'nlayer': m['nlayer'], 'dropout': m['dropout'], 'K': m['K'],
        't': l['l_intra']['form'].endswith('t=0.5') and 0.5 or None,
        'lam1': l['l_intra']['coefficient_lam1'],
        'lam2': l['l_inter']['coefficient_lam2'],
        'lam3': l['l_recon']['coefficient_lam3'],
        'margin': l['l_inter']['margin'], 'margin_w': l['l_inter']['margin_w'],
        'exist_w': l['l_exist']['coefficient_exist_w'],
        'neg_w': 0.5, 'beta': 0.99999,
        'lr': t['optimizer']['lr'], 'wd': t['optimizer']['weight_decay'],
        'neg_ratio': t['neg_ratio'], 'epochs': t['epochs'],
        'update_every': 5 if dr['trigger'] == 'every_5_epochs' else None,
        'theta': 0.5,
        'early_stop': 1, 'val_frac': t['inner_val_frac'],
        'eval_every': t['early_stop']['eval_every'],
        'patience': t['early_stop']['patience'],
        'n_neg_draws': e['cv_triplet']['n_neg_draws'],
        'predictor': 'diag' if m['predictor'] == 'diag_bilinear' else m['predictor'],
        'recon_reduction': l['l_recon']['reduction'],
        'diag_leak': 0, 'ablation': 'none', 'protocol': 'type',
        'folds': 5, 'only_fold': -1, 'cap_threads': 0, 'threads': 2,
        'seed': 1234,
    }
    if resolved['t'] is None:
        raise LockViolation('l_intra temperature not parseable as t=0.5')
    if resolved['update_every'] is None:
        raise LockViolation(f"refresh trigger {dr['trigger']} unsupported")
    for k, v in resolved.items():
        if k in LOCKED_HYPERPARAMS and v != LOCKED_HYPERPARAMS[k]:
            raise LockViolation(f'lock divergence {k}: {v} != {LOCKED_HYPERPARAMS[k]}')
    if resolved['diag_leak'] == REJECT['diag_leak']:
        raise LockViolation('diag_leak=1 rejected in P5 runner')
    if profile not in SUPPORTED_PROFILES:
        raise LockViolation(f'unknown profile {profile}')
    resolved['profile'] = profile
    if overrides:
        for k, v in overrides.items():
            if k == 'diag_leak' and v == 1:
                raise LockViolation('diag_leak=1 rejected in P5 runner')
            resolved[k] = v
    return types.SimpleNamespace(**resolved)


def resolve_run(data_path=None, fold_dir=None, profile='P5_BENCHMARK_UNKNOWN_PROVENANCE'):
    """Produce the full resolved configuration + all hashes for a
    proposed run — NO training. Safe entrypoint for the gate."""
    cfg = load_lock()
    data_path = data_path or os.path.join(
        REPO, 'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')
    fold_dir = fold_dir or os.path.join(
        REPO, 'faithful_repro/independent_v32_outer_refit_20261117/evaluation')
    folds, fold_shas = validate_hashes(data_path, fold_dir)
    Y = np.load(data_path, allow_pickle=True)['Y']
    folds = build_splits(Y, folds, cfg['training']['inner_val_frac'])
    a = args_from_lock(cfg, profile)
    split_shas = {}
    for f, fo in enumerate(folds):
        h = hashlib.sha256()
        for k in ('train', 'val', 'test'):
            h.update(np.asarray(fo[k]).tobytes())
        split_shas[f'fold{f}'] = h.hexdigest()
    return {
        'resolved_args': vars(a),
        'claim_ceiling': PROFILE_CLAIM_CEILING[profile],
        'data_sha256': _sha(data_path),
        'fold_manifest_sha256': fold_shas,
        'split_sha256': split_shas,
        'code_sha256': {k: _sha(os.path.join(_HERE, k)) for k in
                        ['p5_runner.py', 'p5_evaluator.py', 'p5_diagnostics.py']},
        'model_sha256': _sha(os.path.join(REPO, 'faithful_repro/code/faithful.py')),
        'interpretation_limitation': ev.INTERPRETATION_LIMITATION,
        'n_folds': len(folds),
    }


# ---------------- training loop (synthetic-only until authorized) ----------------

def _val_predictions(model, X, G, ei, val_pairs):
    model.eval()
    with torch.no_grad():
        _, tp, *_ = model(X, G, ei)
    model.train()
    tp = tp.cpu().numpy()
    return tp[val_pairs[:, 0], val_pairs[:, 1]]


def train_fold(D, fold, a, diag=None, epochs=None, return_state=False):
    """P5 training loop: same loss assembly as faithful.run_fold but with
    P5 evaluator semantics — checkpoint by MULTIHOT_MACRO_F1 on inner-val,
    thresholds from val only, test evaluated exactly once.
    Intended for synthetic fixtures in G4.1."""
    epochs = epochs or a.epochs
    Y = D['Y']; nm, nd, C = Y.shape
    Ytr = np.zeros_like(Y)
    Ytr[fold['train'][:, 0], fold['train'][:, 1]] = Y[fold['train'][:, 0], fold['train'][:, 1]]
    A = (Ytr.sum(2) > 0).astype(np.float32)
    Af = A  # diag_leak rejected at args level — Af is always train A
    sims = {k: (fz.gip(T) if D[k].ndim == 0 else D[k].astype(np.float32))
            for k, T in [('sim_m1', Af), ('sim_m2', Af), ('sim_d1', Af.T), ('sim_d2', Af.T)]}
    X = {'m1': np.hstack([sims['sim_m1'], Af]), 'm2': np.hstack([sims['sim_m2'], Af]),
         'd1': np.hstack([sims['sim_d1'], Af.T]), 'd2': np.hstack([sims['sim_d2'], Af.T])}
    build = fz.gcn_G if a.ablation == 'no_hgcn' else fz.knn_G
    G = {k: build(v, a.K) for k, v in X.items()}
    Xt = {k: torch.tensor(v) for k, v in X.items()}
    At, Yt = torch.tensor(A), torch.tensor(Ytr)
    S_M, S_D = torch.tensor(sims['sim_m2']), torch.tensor(sims['sim_d2'])
    ei = fz.edges(torch.tensor(Af), S_M, S_D, a.theta)
    excl = np.zeros((nm, nd), bool)
    excl[fold['test'][:, 0], fold['test'][:, 1]] = True
    excl[fold['val'][:, 0], fold['val'][:, 1]] = True
    neg_pool = np.argwhere((A == 0) & ~excl)
    pos = torch.tensor(np.argwhere(A > 0))
    rng = np.random.default_rng(a.seed)
    neg = torch.tensor(neg_pool[rng.choice(len(neg_pool), min(len(neg_pool), a.neg_ratio * len(pos)), False)])
    n_k = Ytr.sum((0, 1)); w = (1 - a.beta) / (1 - a.beta ** np.maximum(n_k, 1))
    w = torch.tensor((w * C / w.sum()), dtype=torch.float32)
    model = fz.DHGCMDA(nm, nd, C, a)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bad, hist = -1, None, 0, []
    refresh_epochs = []
    for ep in range(1, epochs + 1):
        ex, tp, lintra, zm, zd, Sm, Sd = model(Xt, G, ei)
        p_s, n_s = ex[pos[:, 0], pos[:, 1]], ex[neg[:, 0], neg[:, 1]]
        l_ex = ((-(1 - p_s) ** 2 * torch.log(p_s + 1e-7)).sum()
                + a.neg_w * (-(n_s ** 2) * torch.log(1 - n_s + 1e-7)).sum()) / (len(p_s) + len(n_s))
        yp = Yt[pos[:, 0], pos[:, 1]]; tpp = tp[pos[:, 0], pos[:, 1]]
        l_type = -(w * yp * torch.log(tpp + 1e-9)).sum(1).mean()
        l_inter = fz.inter_loss(zm, zd, At, a.t, a.margin, a.margin_w)
        l_rec = ((S_M - Sm) ** 2).sum() + ((S_D - Sd) ** 2).sum() if a.recon_reduction == 'sum' \
            else ((S_M - Sm) ** 2).mean() + ((S_D - Sd) ** 2).mean()
        loss = l_type + a.lam1 * lintra + a.lam2 * l_inter + a.lam3 * l_rec + a.exist_w * l_ex
        opt.zero_grad(); loss.backward()
        pre = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        post = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        if diag:
            diag.loss_components(ep, dict(
                total=float(loss), l_type=float(l_type), l_intra_w=float(a.lam1 * lintra),
                l_inter_w=float(a.lam2 * l_inter), l_rec_w=float(a.lam3 * l_rec), l_ex_w=float(a.exist_w * l_ex)))
            diag.grad_clip_stats(ep, float(pre), post)
            diag.param_update(model)
        if ep % a.update_every == 0:
            with torch.no_grad():
                before = dg.edge_counts(ei) if diag else None
                Smn, Sdn = fz.mm01(Sm.detach()), fz.mm01(Sd.detach())
                oldG = {k: v.data_ptr() for k, v in G.items()}
                ei = fz.edges(torch.tensor(Af), Smn, Sdn, a.theta)
                Hm = build(np.hstack([Smn.cpu().numpy(), Af]), a.K)
                Hd = build(np.hstack([Sdn.cpu().numpy(), Af.T]), a.K)
                G = {'m1': Hm, 'm2': Hm, 'd1': Hd, 'd2': Hd}
                refresh_epochs.append(ep)
                if diag:
                    diag.refresh(ep, before, dg.edge_counts(ei),
                                 {k: v.data_ptr() for k, v in G.items()} != oldG)
        if a.early_stop and ep % a.eval_every == 0:
            pv = _val_predictions(model, Xt, G, ei, fold['val'])
            taus, _ = ev.per_class_thresholds(pv, Y[fold['val'][:, 0], fold['val'][:, 1]])
            m = ev.multihot_macro_f1(Y[fold['val'][:, 0], fold['val'][:, 1]], pv, taus)
            hist.append({'ep': ep, 'val_macro_f1': m['macro_f1']})
            if m['macro_f1'] > best:
                best, bad, best_state = m['macro_f1'], 0, copy.deepcopy(
                    (model.state_dict(), {k: x.clone() for k, x in G.items()}, {k: x.clone() for k, x in ei.items()}))
                if diag:
                    diag.checkpoint(ep, m['macro_f1'], 'inner_val')
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state[0]); G, ei = best_state[1], best_state[2]
    # ---- freeze, then evaluate outer test EXACTLY once ----
    pv = _val_predictions(model, Xt, G, ei, fold['val'])
    taus, val_f1 = ev.per_class_thresholds(pv, Y[fold['val'][:, 0], fold['val'][:, 1]])
    model.eval()
    with torch.no_grad():
        _, tp_t, *_ = model(Xt, G, ei)
    tp_t = tp_t.cpu().numpy()
    te = fold['test']
    res = ev.multihot_macro_f1(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]], taus)
    res.update(
        aux_top1_anyhit=ev.top1_anyhit_harmonic(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]]),
        aux_top1_firstpos=ev.top1_firstpos_harmonic(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]]),
        best_val_macro_f1=best, refresh_epochs=refresh_epochs,
        val_history=hist, threshold_source='inner_validation_only')
    if return_state:
        return res, model, G, ei
    return res


def synthetic_fixture(seed=0, nm=40, nd=30, C=5, density=0.35, views='random'):
    """Synthetic tensor-shaped fixture; 'random' views are label-free."""
    rng = np.random.default_rng(seed)
    Y = (rng.random((nm, nd, C)) < density).astype(np.float32)
    Y = Y * (rng.random((nm, nd, 1)) < 0.9)  # some empty pairs
    D = {'Y': Y}
    if views == 'random':
        for k, n in [('sim_m1', nm), ('sim_m2', nm), ('sim_d1', nd), ('sim_d2', nd)]:
            s = rng.random((n, n)).astype(np.float32)
            D[k] = (s + s.T) / 2
    elif views == 'gip':
        A = (Y.sum(2) > 0).astype(np.float32)
        D.update(sim_m1=np.zeros(0), sim_m2=np.zeros(0), sim_d1=np.zeros(0), sim_d2=np.zeros(0))
    return D


def synthetic_fold(Y, f=0, folds=5, val_frac=0.1, seed=1234):
    """Pair-level disjoint synthetic folds with deterministic inner-val."""
    allpos = np.argwhere(Y.sum(2) > 0)
    rng = np.random.default_rng(seed); rng.shuffle(allpos)
    test = np.array_split(allpos, folds)[f]
    rest = np.concatenate([p for i, p in enumerate(np.array_split(allpos, folds)) if i != f])
    rng2 = np.random.default_rng(seed * 1000 + f); rng2.shuffle(rest)
    nv = int(round(val_frac * len(rest)))
    return {'test': test, 'val': rest[:nv], 'train': rest[nv:]}
