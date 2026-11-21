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
import os, sys, json, hashlib, copy, types, time, random
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


class HardStop(RuntimeError):
    """Runtime safety abort — non-finite loss/gradient, wall-clock cap,
    or interrupted run. The partial state is returned, never discarded."""


# profiles whose provenance gate has not passed may not bind a run context
BLOCKED_PROFILES = {'P5_SPLIT_AWARE_EXTERNAL': 'BLOCKED_FEATURE_PROVENANCE (MISIM upstream)'}

# overrides permitted in a REAL run context (non-scientific runtime only)
RUNTIME_OVERRIDABLE = {'threads', 'cap_threads', 'seed', 'out'}
# additional keys permitted ONLY in a synthetic/test context
SYNTHETIC_OVERRIDABLE = {'hidden', 'proj', 'heads', 'nlayer', 'dropout', 'K',
                         'epochs', 'eval_every', 'patience', 'update_every',
                         'neg_ratio', 'n_neg_draws', 'lr', 'wd', 'lam1',
                         'lam2', 'lam3', 'exist_w', 'neg_w', 'beta', 't',
                         'margin', 'margin_w', 'theta', 'val_frac',
                         'predictor', 'recon_reduction', 'ablation',
                         'protocol', 'folds', 'only_fold', 'early_stop'}

THRESHOLD_GRID = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # pinned
AGGREGATION_POLICY = 'mean_of_fold_macro_f1 primary; pooled TP/FP/FN variant also reported'

# Explicitly registered protocol amendments: amendment id -> {locked key:
# amended value}. An amendment unlocks exactly the listed keys in a real
# run context; every other locked value still fails closed.
REGISTERED_AMENDMENTS = {
    'G7_A1_RECON_MEAN': {'recon_reduction': 'mean'},
}


def atomic_write(path, write_fn):
    """Write to tmp file then os.replace — a crash never leaves a
    half-written artifact."""
    tmp = path + '.tmp'
    write_fn(tmp)
    os.replace(tmp, path)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    atomic_write(path, lambda t: json.dump(obj, open(t, 'w'), indent=2,
                                           ensure_ascii=False))


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


def args_from_lock(cfg, profile, overrides=None, synthetic=False,
                   amendment=None):
    """Map the YAML lock onto the flat args namespace faithful.py expects.
    Any lock value diverging from LOCKED_HYPERPARAMS fails closed.
    Overrides: in a real run context only RUNTIME_OVERRIDABLE keys;
    in a synthetic/test context additionally SYNTHETIC_OVERRIDABLE.
    Everything else is rejected — nothing is silently accepted.
    `amendment` must name a REGISTERED_AMENDMENTS entry; it swaps exactly
    the registered locked values and nothing else."""
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
        allowed = set(RUNTIME_OVERRIDABLE) | (SYNTHETIC_OVERRIDABLE if synthetic else set())
        for k, v in overrides.items():
            if k == 'diag_leak' and v == 1:
                raise LockViolation('diag_leak=1 rejected in P5 runner')
            if k not in allowed:
                raise LockViolation(
                    f'unauthorized override {k}={v} in '
                    f'{"synthetic" if synthetic else "real"} context')
            resolved[k] = v
        resolved['applied_overrides'] = dict(overrides)
    amended = {}
    if amendment is not None:
        if amendment not in REGISTERED_AMENDMENTS:
            raise LockViolation(f'unregistered protocol amendment {amendment}')
        amended = REGISTERED_AMENDMENTS[amendment]
        for k, v in amended.items():
            resolved[k] = v
        resolved['amendment'] = amendment
    # re-enforce every locked value AFTER overrides/amendment — an override
    # may not silently change a locked parameter in a real context
    if not synthetic:
        for k, v in LOCKED_HYPERPARAMS.items():
            want = amended.get(k, v)
            if resolved.get(k) != want:
                raise LockViolation(
                    f'locked param {k} diverges: {resolved.get(k)} != {want}')
    return types.SimpleNamespace(**resolved)


def bind_context(profile, seed=1234, data_path=None, fold_dir=None):
    """Validated run context: verified SHAs, locked folds, approved
    profile, pinned seed. train_fold requires this; it cannot be built
    with an unverified tensor, a tampered manifest or a blocked profile."""
    if profile in BLOCKED_PROFILES:
        raise LockViolation(f'profile {profile} blocked: {BLOCKED_PROFILES[profile]}')
    if profile not in SUPPORTED_PROFILES:
        raise LockViolation(f'unknown profile {profile}')
    cfg = load_lock()
    data_path = data_path or os.path.join(
        REPO, 'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')
    fold_dir = fold_dir or os.path.join(
        REPO, 'faithful_repro/independent_v32_outer_refit_20261117/evaluation')
    folds, fold_shas = validate_hashes(data_path, fold_dir, seed=seed)
    D = {k: v for k, v in np.load(data_path, allow_pickle=True).items()}
    Y = D['Y']
    folds = build_splits(Y, folds, cfg['training']['inner_val_frac'], seed=seed)
    # verification-only cross-check: manifest y must equal tensor Y at
    # the test coords — disagreement means a corrupted manifest
    for f, fo in enumerate(folds):
        ty = np.asarray(np.load(os.path.join(
            fold_dir, f'fold{f}_seed{seed}_static.npz'), allow_pickle=True)['y'])
        assert np.array_equal(ty, Y[fo['test'][:, 0], fo['test'][:, 1]]), \
            f'fold{f} manifest y != tensor Y at test coords'
    return {
        'cfg': cfg, 'folds': folds, 'Y': Y, 'D': D,
        'data_sha256': _sha(data_path), 'fold_shas': fold_shas,
        'profile': profile, 'seed': seed,
        'claim_ceiling': PROFILE_CLAIM_CEILING[profile],
        'threshold_grid': THRESHOLD_GRID,
        'aggregation_policy': AGGREGATION_POLICY,
    }


def resolve_run(data_path=None, fold_dir=None, profile='P5_BENCHMARK_UNKNOWN_PROVENANCE'):
    """Produce the full resolved configuration + all hashes for a
    proposed run — NO training. Safe entrypoint for the gate."""
    ctx = bind_context(profile, data_path=data_path, fold_dir=fold_dir)
    cfg = ctx['cfg']
    a = args_from_lock(cfg, profile)
    split_shas = {}
    for f, fo in enumerate(ctx['folds']):
        h = hashlib.sha256()
        for k in ('train', 'val', 'test'):
            h.update(np.asarray(fo[k]).tobytes())
        split_shas[f'fold{f}'] = h.hexdigest()
    return {
        'resolved_args': vars(a),
        'claim_ceiling': ctx['claim_ceiling'],
        'data_sha256': ctx['data_sha256'],
        'fold_manifest_sha256': ctx['fold_shas'],
        'split_sha256': split_shas,
        'threshold_grid': THRESHOLD_GRID,
        'aggregation_policy': AGGREGATION_POLICY,
        'code_sha256': {k: _sha(os.path.join(_HERE, k)) for k in
                        ['p5_runner.py', 'p5_evaluator.py', 'p5_diagnostics.py']},
        'model_sha256': _sha(os.path.join(REPO, 'faithful_repro/code/faithful.py')),
        'interpretation_limitation': ev.INTERPRETATION_LIMITATION,
        'n_folds': len(ctx['folds']),
    }


# ---------------- training loop (synthetic-only until authorized) ----------------

def _score_diagnostics(y, p, taus):
    """Per-class TEST-score diagnostics: AUROC (rank, tie-averaged), AP,
    score quantiles, positive prediction rate. Analysis-only — never
    used for model or threshold selection."""
    y = np.asarray(y); p = np.asarray(p)
    C = y.shape[1]; out = {}
    for k in range(C):
        col_y, col_p = y[:, k], p[:, k]
        npos = int(col_y.sum()); nneg = len(col_y) - npos
        # AUROC via tie-averaged Mann-Whitney
        order = np.argsort(col_p, kind='mergesort')
        ranks = np.empty(len(col_p)); ranks[order] = np.arange(len(col_p))
        # average ranks within ties
        sp = col_p[order]; i = 0
        while i < len(sp):
            j = i
            while j + 1 < len(sp) and sp[j + 1] == sp[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        auc = float('nan') if npos == 0 or nneg == 0 else float(
            (ranks[col_y == 1].sum() - npos * (npos - 1) / 2.0) / (npos * nneg))
        # AP: mean precision at each positive rank
        desc = np.argsort(-col_p, kind='mergesort')
        hits = col_y[desc]; cum = np.cumsum(hits)
        pos_idx = np.nonzero(hits)[0]
        ap = float('nan') if npos == 0 else float(
            (cum[pos_idx] / (pos_idx + 1)).sum() / npos)
        out[f'class_{k}'] = {
            'auroc': auc, 'ap': ap,
            'score_quantiles': [float(q) for q in np.quantile(col_p, [0, .25, .5, .75, 1])],
            'pos_pred_rate': float((col_p >= taus[k]).mean()),
            'support': npos}
    return out


def _val_predictions(model, X, G, ei, val_pairs):
    model.eval()
    with torch.no_grad():
        _, tp, *_ = model(X, G, ei)
    model.train()
    tp = tp.cpu().numpy()
    return tp[val_pairs[:, 0], val_pairs[:, 1]]


def train_fold(D, fold, a, diag=None, epochs=None, return_state=False,
               max_wall_s=None):
    """P5 training loop: same loss assembly as faithful.run_fold but with
    P5 evaluator semantics — checkpoint by MULTIHOT_MACRO_F1 on inner-val,
    thresholds from val only, test evaluated exactly once.
    Hard stops: non-finite loss/gradient -> HardStop; max_wall_s ->
    safe interruption with partial state marked 'interrupted'.
    Intended for synthetic fixtures in G4.x until G5 authorization."""
    epochs = epochs or a.epochs
    # seed EVERYTHING before any sampling or model construction — two
    # runs with the same seed/config must produce identical checkpoints
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    t_start = time.time()
    res_status = {'interrupted': None}
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
    best, best_state, best_epoch, bad, hist = -1, None, None, 0, []
    opt_steps = 0
    grid = np.asarray(THRESHOLD_GRID)
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
        if not torch.isfinite(loss):
            raise HardStop(f'non_finite_loss at epoch {ep}')
        opt.zero_grad(); loss.backward()
        bad_g = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_g:
            raise HardStop(f'non_finite_gradient at epoch {ep}: {bad_g[:3]}')
        pre = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        post = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step(); opt_steps += 1
        if max_wall_s is not None and (time.time() - t_start) > max_wall_s:
            res_status['interrupted'] = f'wall_cap_{max_wall_s}s at epoch {ep}'
            break
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
            taus, _ = ev.per_class_thresholds(
                pv, Y[fold['val'][:, 0], fold['val'][:, 1]], grid=grid)
            m = ev.multihot_macro_f1(Y[fold['val'][:, 0], fold['val'][:, 1]], pv, taus)
            hist.append({'ep': ep, 'val_macro_f1': m['macro_f1']})
            if m['macro_f1'] > best:
                best, best_epoch, bad, best_state = m['macro_f1'], ep, 0, copy.deepcopy(
                    (model.state_dict(), {k: x.clone() for k, x in G.items()}, {k: x.clone() for k, x in ei.items()}))
                if diag:
                    diag.checkpoint(ep, m['macro_f1'], 'inner_val')
            else:
                bad += 1
                if bad >= a.patience:
                    break
    interrupted = res_status['interrupted'] is not None
    if interrupted or best_state is None:
        # NEVER score outer-test on an interrupted run, and never report
        # a valid result without a validation-selected checkpoint.
        res = {'metric': ev.PRIMARY, 'macro_f1': None,
               'outcome': 'interrupted' if interrupted else 'failed_no_checkpoint',
               'reason': res_status['interrupted'] or 'no validation-selected checkpoint',
               'best_val_macro_f1': best if best >= 0 else None,
               'selected_epoch': best_epoch, 'val_history': hist,
               'refresh_epochs': refresh_epochs,
               'optimizer_steps': opt_steps,
               'thresholds': None, 'threshold_source': 'none_selected',
               'threshold_grid': list(THRESHOLD_GRID),
               'wall_s': round(time.time() - t_start, 2)}
        res.update(res_status)
        if return_state:
            return res, model, G, ei
        return res
    model.load_state_dict(best_state[0]); G, ei = best_state[1], best_state[2]
    # ---- freeze, then evaluate outer test EXACTLY once ----
    pv = _val_predictions(model, Xt, G, ei, fold['val'])
    taus, val_f1 = ev.per_class_thresholds(
        pv, Y[fold['val'][:, 0], fold['val'][:, 1]], grid=grid)
    model.eval()
    with torch.no_grad():
        _, tp_t, *_ = model(Xt, G, ei)
    tp_t = tp_t.cpu().numpy()
    te = fold['test']
    res = ev.multihot_macro_f1(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]], taus)
    res['score_diagnostics'] = _score_diagnostics(
        Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]], taus)
    res.update(
        aux_top1_anyhit=ev.top1_anyhit_harmonic(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]]),
        aux_top1_firstpos=ev.top1_firstpos_harmonic(Y[te[:, 0], te[:, 1]], tp_t[te[:, 0], te[:, 1]]),
        best_val_macro_f1=best, refresh_epochs=refresh_epochs,
        val_history=hist, threshold_source='inner_validation_only',
        selected_epoch=best_epoch, optimizer_steps=opt_steps,
        threshold_grid=list(THRESHOLD_GRID), outcome='completed',
        wall_s=round(time.time() - t_start, 2))
    res.update(res_status)
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


def save_run_artifacts(outdir, res, model, resolved, diag=None, graph=None):
    """Atomic artifact bundle: checkpoint (model + SELECTED graph state
    G/ei when supplied), selected epoch, val thresholds, resolved
    config/hash manifest, diagnostics."""
    os.makedirs(outdir, exist_ok=True)
    atomic_json(os.path.join(outdir, 'result.json'), res)
    atomic_json(os.path.join(outdir, 'resolved_manifest.json'), resolved)
    if diag:
        atomic_json(os.path.join(outdir, 'diagnostics.json'), diag.report())
    ck = os.path.join(outdir, 'checkpoint.pt')
    payload = {'state_dict': model.state_dict(),
               'best_val_macro_f1': res.get('best_val_macro_f1'),
               'selected_epoch': res.get('selected_epoch'),
               'thresholds': res.get('thresholds'),
               'threshold_source': res.get('threshold_source'),
               'graph_state': ('selected' if res.get('outcome') == 'completed'
                               else 'final_unselected')}
    if graph is not None:
        payload['G'] = graph['G']; payload['ei'] = graph['ei']
    atomic_write(ck, lambda t: torch.save(payload, t))
    return {'dir': outdir,
            'files': ['result.json', 'resolved_manifest.json',
                      'diagnostics.json' if diag else None, 'checkpoint.pt']}


def main(argv=None):
    """G5 CLI. --dry-run resolves ONLY (never an optimizer step).
    --once runs one fold under the bound context (requires owner
    authorization on real data). --synthetic exercises the path on a
    synthetic fixture without real-data SHAs."""
    import argparse
    p = argparse.ArgumentParser(prog='p5_runner')
    p.add_argument('--dry-run', action='store_true',
                   help='resolve config+hashes only; NEVER trains')
    p.add_argument('--once', action='store_true', help='train one fold')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--profile', default='P5_BENCHMARK_UNKNOWN_PROVENANCE')
    p.add_argument('--data', default=None); p.add_argument('--fold-dir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--max-wall-s', type=float, default=None)
    p.add_argument('--synthetic', action='store_true',
                   help='synthetic fixture only — no real-data context')
    p.add_argument('--amendment', default=None,
                   help='registered protocol amendment id (e.g. G7_A1_RECON_MEAN)')
    a = p.parse_args(argv)
    if a.dry_run:
        if a.synthetic:
            cfg = load_lock()
            res = {'synthetic': True,
                   'resolved_args': vars(args_from_lock(
                       cfg, a.profile, synthetic=True)),
                   'threshold_grid': THRESHOLD_GRID,
                   'aggregation_policy': AGGREGATION_POLICY,
                   'note': 'synthetic dry-run: no SHAs to bind'}
        else:
            res = resolve_run(a.data, a.fold_dir, a.profile)
        out = a.out or os.path.join(_HERE, 'runs', 'dry_run.json')
        atomic_json(out, res)
        print(json.dumps({'dry_run': True, 'optimizer_steps': 0, 'out': out}))
        return 0
    if a.once:
        if a.synthetic:
            cfg = load_lock()
            ar = args_from_lock(cfg, a.profile, synthetic=True,
                                overrides={'hidden': 32, 'proj': 16,
                                           'epochs': 3, 'eval_every': 1,
                                           'patience': 3, 'update_every': 2,
                                           'K': 4, 'seed': a.seed})
            D = synthetic_fixture(seed=a.seed)
            fold = synthetic_fold(D['Y'], seed=a.seed)
            diag = dg.Diagnostics()
            manifest = {'resolved_args': vars(ar),
                        'threshold_grid': THRESHOLD_GRID,
                        'aggregation_policy': AGGREGATION_POLICY,
                        'context': 'synthetic', 'seed': a.seed}
            outdir = a.out or os.path.join(_HERE, 'runs', 'synthetic_once')
            try:
                res, model, G_sel, ei_sel = train_fold(
                    D, fold, ar, diag=diag,
                    max_wall_s=a.max_wall_s, return_state=True)
            except HardStop as e:
                rep = {'outcome': 'failed', 'hard_stop': str(e),
                       'context': 'synthetic', 'manifest': manifest}
                atomic_json(os.path.join(outdir, 'failure_report.json'), rep)
                atomic_json(os.path.join(outdir, 'diagnostics.json'),
                            diag.report())
                print(json.dumps({'ran': False, 'outcome': 'failed',
                                  'hard_stop': str(e), 'outdir': outdir}))
                return 1
            res['context'] = 'synthetic'
            art = save_run_artifacts(outdir, res, model, manifest, diag,
                                     graph={'G': G_sel, 'ei': ei_sel})
            print(json.dumps({'ran': True, 'context': 'synthetic',
                              'outcome': res['outcome'],
                              'macro_f1': res['macro_f1'], 'outdir': outdir}))
            return 0
        # real-data path — full context binding (sha + fold + profile + seed)
        ctx = bind_context(a.profile, a.seed, a.data, a.fold_dir)
        ar = args_from_lock(ctx['cfg'], a.profile, overrides={'seed': a.seed},
                            amendment=a.amendment)
        diag = dg.Diagnostics()
        manifest = {'resolved_args': vars(ar),
                    'threshold_grid': THRESHOLD_GRID,
                    'aggregation_policy': AGGREGATION_POLICY,
                    'data_sha256': ctx['data_sha256'],
                    'fold_shas': ctx['fold_shas'],
                    'claim_ceiling': ctx['claim_ceiling'],
                    'amendment': a.amendment,
                    'profile': a.profile, 'seed': a.seed, 'fold': a.fold}
        suffix = f'_{a.amendment}' if a.amendment else ''
        outdir = a.out or os.path.join(
            _HERE, 'runs', f'g5_fold{a.fold}_s{a.seed}{suffix}')
        try:
            res, model, G_sel, ei_sel = train_fold(
                ctx['D'], ctx['folds'][a.fold], ar,
                diag=diag, max_wall_s=a.max_wall_s, return_state=True)
        except HardStop as e:
            rep = {'outcome': 'failed', 'hard_stop': str(e),
                   'context': manifest}
            atomic_json(os.path.join(outdir, 'failure_report.json'), rep)
            atomic_json(os.path.join(outdir, 'diagnostics.json'),
                        diag.report())
            print(json.dumps({'ran': False, 'outcome': 'failed',
                              'hard_stop': str(e), 'outdir': outdir}))
            return 1
        res['context'] = {'profile': a.profile, 'seed': a.seed, 'fold': a.fold,
                          'data_sha256': ctx['data_sha256'],
                          'amendment': a.amendment,
                          'claim_ceiling': ctx['claim_ceiling']}
        art = save_run_artifacts(outdir, res, model, manifest, diag,
                                 graph={'G': G_sel, 'ei': ei_sel})
        print(json.dumps({'ran': True, 'fold': a.fold,
                          'outcome': res['outcome'],
                          'macro_f1': res['macro_f1'], 'outdir': outdir}))
        return 0
    p.error('choose --dry-run or --once')


if __name__ == '__main__':
    main()
