# -*- coding: utf-8 -*-
"""S1 runner — build graphs, freeze manifest, train, checkpoint, export
(train/val only).

Graph construction (static, immutable through training):
  - cosine-KNN incidence per view on that view's OWN input matrix
    (author semantics: top-k per center, self guaranteed in own
    hyperedge; k capped at n-1 for small fixtures, declared)
  - hypergraph Laplacian G = Dv^-1/2 H W De^-1 H^T Dv^-1/2 (author)
  - 4-relation HGT edges: sim edges = view-1 KNN member->center pairs
    (self excluded — declared policy), assoc/rev_assoc edges = the
    EXACT train_pairs list (train-only A; mask must equal it exactly)

Contracts: data['views'] IS the input dict — four DISTINCT augmented
inputs per author concat semantics (caller-built, train-only A inside).
Pairs are caller-supplied fold segments; this module never derives its
own split and refuses any 'test' keys in the data bundle.

Run order (hard gates):
  preflight(cfg, data)          — pure-numpy validation, NO torch/model
  env_check(cfg)                — exact pinned key set torch/np/PyG
  frozen manifest               — source+import+cfg+data+split+env hashes
  expected-manifest compare     — COMPLETE manifest REQUIRED, BEFORE
                                  any model construction or output
  graphs -> model -> Adam loop -> instrumentation each eval epoch
  nonfinite abort               — keeps last-good ckpt, NO prediction
                                  export, CLI exits nonzero (2)

Exit codes: 0 ok / 2 aborted-nonfinite / 3 preflight-or-gate failure.

Data/prediction NPZ schemas are FLAT (no object arrays):
  data npz:  view_mi_v1, view_mi_v2, view_dis_v1, view_dis_v2,
             train_mask, train_pairs, val_pairs, y_train, y_val
  pred npz:  pairs (N,2) i64, logits (N,C) f32, y (N,C) i8
  sidecar:   predictions_<split>.meta.json
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

VIEW_KEYS = ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2')
ALLOWED_DATA_KEYS = {
    'views', 'train_mask', 'train_pairs', 'val_pairs',
    'y_train', 'y_val', 'entity_names', 'provenance'}
FORBIDDEN_DATA_KEYS = {'test_pairs', 'y_test'}

ENV_MODULES = {'torch': 'torch', 'numpy': 'numpy',
               'torch_geometric': 'torch_geometric'}


# ---------- graph builders (pure numpy -> torch lazily) -----------------
def cosine_distance_rows(X):
    """1 - cosine_similarity, diag 0, clipped [0,2]. Deterministic."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    S = (X / n) @ (X / n).T
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def knn_incidence_cos(X, k):
    """(n, n) 0/1 incidence: column c = center c's k-nearest nodes by
    cosine distance + self (self guaranteed in own hyperedge — author
    KNN semantics; replaces the farthest member if absent).

    k counts NEIGHBORS: hyperedge size = k + 1 (author default 13 -> 14
    per main param.py K_neigs=[13]). Caller caps k < n."""
    D = cosine_distance_rows(X)
    n = D.shape[0]
    if not 1 <= k < n:
        raise ValueError('k must be in [1, n)')
    H = np.zeros((n, n), dtype=np.float64)
    for c in range(n):
        order = np.argsort(D[c], kind='stable')
        members = [m for m in order.tolist() if m != c][:k]
        members.append(c)
        for m in members:
            H[m, c] = 1.0
    return H


def hypergraph_laplacian(H):
    """G = Dv^-1/2 H W De^-1 H^T Dv^-1/2 (author _generate_G_from_H,
    unit edge weights). Rows with zero degree -> zero row (no NaN)."""
    H = np.asarray(H, dtype=np.float64)
    dv = H.sum(axis=1)
    de = H.sum(axis=0)
    dv2 = np.where(dv > 0, dv ** -0.5, 0.0)
    inv_de = np.where(de > 0, 1.0 / de, 0.0)
    return (dv2[:, None] * H * inv_de[None, :]) @ H.T * dv2[None, :]


def build_view_graphs(views, k):
    """views: {'mi_v1':(n_m,f1),'mi_v2':(n_m,f2),'dis_v1':(n_d,f3),
    'dis_v2':(n_d,f4)} -> {name: torch float32 G} + incidence H dict.
    k neighbors + self per hyperedge; capped at n-1 (declared)."""
    import torch
    G, knn = {}, {}
    for name in VIEW_KEYS:
        if name not in views:
            raise ValueError('views missing ' + name)
        X = np.asarray(views[name])
        kk = min(int(k), X.shape[0] - 1)
        H = knn_incidence_cos(X, kk)
        G[name] = torch.tensor(hypergraph_laplacian(H),
                               dtype=torch.float32)
        knn[name] = H
    return G, knn


def build_edge_index_dict(train_pairs, knn_mi_H, knn_dis_H):
    """4-relation static edges. sim edges: member->center KNN pairs
    excluding self (view-1 incidence; declared). assoc/rev_assoc edges:
    EXACTLY the caller's train_pairs (both typed directions)."""
    import torch
    def sim_edges(H):
        src, dst = [], []
        for center in range(H.shape[1]):
            for member in np.flatnonzero(H[:, center]):
                if member != center:
                    src.append(int(member))
                    dst.append(int(center))
        return src, dst
    m1, m2 = sim_edges(knn_mi_H)
    d1, d2 = sim_edges(knn_dis_H)
    tp = np.asarray(train_pairs, dtype=np.int64)
    ei = {
        ('mir', 'sim', 'mir'):
            torch.tensor([m1, m2], dtype=torch.long),
        ('dis', 'sim', 'dis'):
            torch.tensor([d1, d2], dtype=torch.long),
        ('mir', 'assoc', 'dis'):
            torch.tensor([tp[:, 0].tolist(), tp[:, 1].tolist()],
                         dtype=torch.long),
        ('dis', 'rev_assoc', 'mir'):
            torch.tensor([tp[:, 1].tolist(), tp[:, 0].tolist()],
                         dtype=torch.long),
    }
    return ei


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_path(p):
    return _sha_bytes(open(p, 'rb').read())


def _arr_sha(a):
    return _sha_bytes(np.ascontiguousarray(np.asarray(a)).tobytes())


# ---------- data bundle IO (FLAT npz schema) -----------------------------
def load_data_npz(path):
    """Load a FLAT data bundle. Keys: view_mi_v1/view_mi_v2/
    view_dis_v1/view_dis_v2, train_mask, train_pairs, val_pairs,
    y_train, y_val. No object arrays; returns the data dict."""
    z = np.load(path, allow_pickle=False)
    data = {'views': {}}
    for k in z.files:
        a = z[k]
        if a.dtype == object:
            raise ValueError('object-array npz key not allowed: ' + k)
        if k.startswith('view_'):
            data['views'][k[len('view_'):]] = a
        else:
            data[k] = a
    return data


def save_data_npz(path, data):
    """Write the flat data-bundle schema (fixture/Task3 use)."""
    flat = {'view_' + k: np.asarray(v)
            for k, v in data['views'].items()}
    for k in ('train_mask', 'train_pairs', 'val_pairs',
              'y_train', 'y_val'):
        flat[k] = np.asarray(data[k])
    np.savez(path, **flat)


def read_predictions(npz_path):
    """Load + validate a flat predictions file. Returns dict with
    pairs (N,2) int64, logits (N,C) float, y (N,C) 0/1."""
    z = np.load(npz_path, allow_pickle=False)
    for k in ('pairs', 'logits', 'y'):
        if k not in z.files:
            raise ValueError('predictions npz missing ' + k)
        if z[k].dtype == object:
            raise ValueError('object array in predictions: ' + k)
    out = {'pairs': np.asarray(z['pairs'], dtype=np.int64),
           'logits': np.asarray(z['logits'], dtype=np.float64),
           'y': np.asarray(z['y'], dtype=np.int64)}
    if out['pairs'].ndim != 2 or out['pairs'].shape[1] != 2:
        raise ValueError('pairs must be (N,2)')
    if out['logits'].ndim != 2 or \
            out['logits'].shape[0] != len(out['pairs']):
        raise ValueError('logits must be (N,C)')
    if out['y'].shape != out['logits'].shape or \
            not np.isin(out['y'], (0, 1)).all():
        raise ValueError('y must be binary (N,C)')
    return out


def _export_predictions(out_dir, split, pairs, logits, y, meta):
    npz_path = os.path.join(out_dir, 'predictions_%s.npz' % split)
    np.savez(npz_path, pairs=np.asarray(pairs, dtype=np.int64),
             logits=np.asarray(logits, dtype=np.float32),
             y=np.asarray(y, dtype=np.int8))
    meta = dict(meta)
    meta['split'] = split
    meta['npz_sha256'] = _sha_path(npz_path)
    with open(npz_path.replace('.npz', '.meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=1)


# ---------- preflight (NO torch/model import, genuine early exit) --------
def _pairs_checked(arr, name, n_mi, n_dis, rep_ok):
    """Integer (N,2) pairs in bounds, no duplicates, no fractional
    values smuggled via float dtype. Returns int64 array."""
    a = np.asarray(arr)
    rep_ok('%s 2d' % name, a.ndim == 2 and a.shape[1] == 2,
           str(a.shape))
    rep_ok('%s nonempty' % name, a.shape[0] > 0)
    if not np.issubdtype(a.dtype, np.integer):
        rep_ok('%s integral' % name,
               np.isfinite(a).all()
               and (a == np.floor(a)).all(),
               'fractional or nonfinite pair indices')
    a = a.astype(np.int64)
    rep_ok('%s bounds' % name,
           a.min() >= 0 and a[:, 0].max() < n_mi
           and a[:, 1].max() < n_dis)
    rep_ok('%s unique' % name,
           len(set(map(tuple, a.tolist()))) == a.shape[0],
           'duplicate pairs')
    return a


def preflight(cfg, data):
    """Validate everything that does not need torch/model. Returns a
    report dict; raises ValueError on any violation."""
    rep = {'checks': []}
    def ok(name, cond, detail=''):
        rep['checks'].append((name, bool(cond), detail))
        if not cond:
            raise ValueError('preflight failed: %s %s' % (name, detail))
    req = ('hidden', 'readout', 'proj', 'hgt_layers', 'hgt_heads',
           'n_types', 'hgt_dropout', 'readout_dropout', 'knn_k', 'lr',
           'weight_decay', 'max_epochs', 'eval_every', 'beta', 'seed',
           'leaky_slope', 'temperature_init', 'temperature_min',
           'temperature_max')
    for k_ in req:
        ok('cfg.' + k_, k_ in cfg)
    ok('cfg.n_types==5', cfg['n_types'] == 5, str(cfg.get('n_types')))
    ok('cfg.beta in [0,1)', 0.0 <= cfg['beta'] < 1.0)
    ok('cfg.max_epochs>0', int(cfg['max_epochs']) > 0)
    ok('cfg.temp bounds', 0 < cfg['temperature_min']
       <= cfg['temperature_init'] <= cfg['temperature_max'])
    bad = FORBIDDEN_DATA_KEYS & set(data)
    ok('no test keys', not bad, str(sorted(bad)))
    extra = set(data) - ALLOWED_DATA_KEYS
    ok('known data keys', not extra, str(sorted(extra)))
    for k_ in ('views', 'train_mask', 'train_pairs', 'val_pairs',
               'y_train', 'y_val'):
        ok('data.' + k_, k_ in data)
    views = data['views']
    for vk in VIEW_KEYS:
        ok('view %s present' % vk, vk in views)
    x_mi = np.asarray(views['mi_v1'])
    x_dis = np.asarray(views['dis_v1'])
    n_mi, n_dis = x_mi.shape[0], x_dis.shape[0]
    for name, arr in views.items():
        a = np.asarray(arr)
        ok('view %s finite' % name, a.ndim == 2
           and np.isfinite(a).all())
        want = n_mi if name.startswith('mi') else n_dis
        ok('view %s rows' % name, a.shape[0] == want,
           '%d vs %d' % (a.shape[0], want))
    tm_raw = np.asarray(data['train_mask'])
    ok('mask shape', tm_raw.shape == (n_mi, n_dis), str(tm_raw.shape))
    ok('mask binary', np.isin(tm_raw, (0, 1, False, True)).all())
    tp = _pairs_checked(data['train_pairs'], 'train_pairs', n_mi, n_dis,
                        ok)
    vp = _pairs_checked(data['val_pairs'], 'val_pairs', n_mi, n_dis,
                        ok)
    # train_mask must EXACTLY equal the pair list — rebuilt and compared,
    # not trusted
    expect = np.zeros((n_mi, n_dis), dtype=bool)
    expect[tp[:, 0], tp[:, 1]] = True
    ok('train mask == train pairs',
       (tm_raw.astype(bool) == expect).all(),
       'mask must equal exactly the train_pairs cells')
    ok('val not in train', not (set(map(tuple, vp.tolist()))
                                & set(map(tuple, tp.tolist()))))
    yt, yv = np.asarray(data['y_train']), np.asarray(data['y_val'])
    ok('y shapes', yt.shape == (len(tp), cfg['n_types'])
       and yv.shape == (len(vp), cfg['n_types']))
    ok('y binary', np.isin(yt, (0, 1)).all()
       and np.isin(yv, (0, 1)).all())
    ok('y pair truth positive', (yt.sum(1) > 0).all()
       and (yv.sum(1) > 0).all(),
       'every evaluated pair needs >=1 positive type channel')
    rep['data_hashes'] = {
        'train_mask': _arr_sha(data['train_mask']),
        'train_pairs': _arr_sha(tp), 'val_pairs': _arr_sha(vp),
        'y_train': _arr_sha(yt), 'y_val': _arr_sha(yv)}
    rep['view_hashes'] = {n: _arr_sha(v) for n, v in views.items()}
    rep['n_pairs'] = {'train': int(len(tp)), 'val': int(len(vp))}
    rep['preflight'] = 'PASS'
    return rep


def env_check(cfg):
    """Pinned-runtime gate. cfg['env_require'] MUST declare exactly the
    supported key set {torch, numpy, torch_geometric} with nonempty
    version strings; missing, empty, or unknown keys fail. Versions
    compare after stripping a declared local suffix ('+cpu'-style —
    normalization, not a hardcoded wheel tag)."""
    req = cfg.get('env_require')
    if not isinstance(req, dict) or set(req) != set(ENV_MODULES):
        raise ValueError(
            'env_require must declare exactly %s; got %r'
            % (sorted(ENV_MODULES), None if req is None else sorted(req)))
    for k, v in req.items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError('env_require[%r] must be a nonempty '
                             'version string' % k)
    import importlib
    got, bad = {}, []
    for mod, want in req.items():
        try:
            v = importlib.import_module(ENV_MODULES[mod]) \
                .__version__.split('+')[0]
        except Exception as e:  # pragma: no cover
            got[mod] = 'MISSING(%s)' % e
            bad.append(mod)
            continue
        got[mod] = v
        if v != want:
            bad.append('%s!=%s' % (v, want))
    if bad:
        raise ValueError('env_check failed: %s (got %s)' % (bad, got))
    return got


def source_hashes():
    """Effective source hashes — model.py hashed WITHOUT importing it
    (genuine pre-model gate), runner/losses by path, plus the pure
    imported Task1 modules."""
    h = {'model.py': _sha_path(os.path.join(_HERE, 'model.py')),
         'runner.py': _sha_path(os.path.abspath(__file__)),
         'losses.py': _sha_path(os.path.join(_HERE, 'losses.py'))}
    return h


def imported_pure_hashes():
    """sha256 of Task1 pure sources consumed via losses.py."""
    for fn in ('contracts.py', 'metrics.py'):
        p = os.path.join(_HERE, fn)
        if not os.path.exists(p):
            raise ValueError('missing imported pure source ' + fn)
    return {fn: _sha_path(os.path.join(_HERE, fn))
            for fn in ('contracts.py', 'metrics.py')}


def frozen_manifest(cfg, preflight_rep, env_versions):
    """Everything that must be identical to a declared expected run,
    computed BEFORE model construction."""
    return {
        'stage': 'S1',
        'cfg': dict(cfg),
        'env': dict(env_versions,
                    python=platform.python_version(),
                    platform=platform.platform()),
        'data_hashes': preflight_rep['data_hashes'],
        'view_hashes': preflight_rep['view_hashes'],
        'n_pairs': preflight_rep['n_pairs'],
        'source_hashes': source_hashes(),
        'imported_pure_hashes': imported_pure_hashes(),
    }


EXPECTED_MANIFEST_KEYS = ('stage', 'cfg', 'env', 'data_hashes',
                          'view_hashes', 'n_pairs', 'source_hashes',
                          'imported_pure_hashes')
# extra top-level annotations a reviewer may add to a candidate manifest
EXPECTED_MANIFEST_META_KEYS = ('manifest_kind', 'generated_by', 'time',
                               'review_note')


def check_expected_manifest(frozen, expected):
    """Compare a COMPLETE candidate expected manifest against the frozen
    one — all EXPECTED_MANIFEST_KEYS required, nested maps compared
    fully, BEFORE model construction/output creation. 'env.platform' is
    the sole documented descriptive exception (host-descriptive string,
    never compared). env.python IS compared."""
    if not isinstance(expected, dict):
        raise ValueError('expected manifest must be a JSON object')
    missing = [k for k in EXPECTED_MANIFEST_KEYS if k not in expected]
    extra = [k for k in expected
             if k not in EXPECTED_MANIFEST_KEYS
             and k not in EXPECTED_MANIFEST_META_KEYS]
    if missing:
        raise ValueError('expected manifest missing keys: %s' % missing)
    if extra:
        raise ValueError('expected manifest unknown keys: %s' % extra)
    diffs = []
    def cmp(path, a, b):
        if a != b:
            diffs.append('%s: expected %r got %r' % (path, b, a))
    for key in EXPECTED_MANIFEST_KEYS:
        if key == 'env':
            ea, eb = expected['env'], frozen['env']
            if not isinstance(ea, dict):
                diffs.append('env: expected a map, got %r' % (ea,))
                continue
            for k in set(ea) | set(eb):
                if k == 'platform':
                    continue      # documented descriptive exception
                cmp('env.' + k, eb.get(k), ea.get(k))
        else:
            cmp(key, frozen.get(key), expected.get(key))
    if diffs:
        raise ValueError('expected-manifest mismatch:\n'
                         + '\n'.join(diffs))


def expected_manifest_for(cfg, data):
    """Step 1 of the documented freeze-then-train procedure: generate a
    genuine FULL candidate expected manifest (preflight + env gate +
    frozen hashes), for coordinator review BEFORE any training."""
    rep = preflight(cfg, data)
    envv = env_check(cfg)
    m = frozen_manifest(cfg, rep, envv)
    m['manifest_kind'] = 'candidate_expected'
    return m


# ---------- training ----------------------------------------------------
def _ckpt(model, opt, epoch, val_metric, G, ei, cfg):
    import torch
    return {
        'epoch': int(epoch), 'val_macro_f1': float(val_metric),
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'rng_torch': torch.get_rng_state(),
        'rng_numpy': np.random.get_state(),
        # ACTUAL graph state (not hash-only) + hashes for quick compare
        'G': dict(G), 'ei': dict(ei),
        'G_sha': {k: _sha_bytes(v.cpu().numpy().tobytes())
                  for k, v in G.items()},
        'ei_sha': {str(k): _sha_bytes(v.cpu().numpy().tobytes())
                   for k, v in ei.items()},
        'cfg': dict(cfg),
    }


def _flat_params(model):
    import torch
    return torch.cat([p.detach().reshape(-1)
                      for p in model.parameters()])


def build_optimizer(model, cfg):
    """Coordinator-selected baseline optimizer: torch.optim.Adam.

    This is the coordinator's explicit baseline choice for this
    engineering track — NOT a claim that the author/paper used Adam
    (no source citation proves that)."""
    import torch
    return torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                            weight_decay=float(cfg['weight_decay']))


def train_s1(cfg, data, out_dir, log=print, expected_manifest=None):
    """Train S1 on caller-supplied fold. Returns manifest dict.

    expected_manifest is REQUIRED (full candidate manifest produced by
    the freeze-then-train step and reviewed); None or incomplete is
    refused BEFORE model construction AND before out_dir is created.

    Aborts on nonfinite logits/loss/grad (train OR eval): manifest
    written, last-good best checkpoint preserved, NO predictions
    exported, manifest['aborted'] set — callers/CLI must treat as
    failure (nonzero exit)."""
    import torch
    import model as pg_model
    import losses

    rep = preflight(cfg, data)              # pure numpy, no model
    envv = env_check(cfg)                   # pinned runtime gate
    frozen = frozen_manifest(cfg, rep, envv)
    check_expected_manifest(frozen, expected_manifest)  # BEFORE model+out
    os.makedirs(out_dir, exist_ok=False)    # occupied namespace refused
    with open(os.path.join(out_dir, 'manifest_frozen.json'), 'w',
              encoding='utf-8') as f:
        json.dump(frozen, f, indent=1, default=str)

    device = torch.device('cpu')
    torch.manual_seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']) % (2 ** 32))

    views = data['views']
    inputs = {k: torch.tensor(np.asarray(views[k]),
                              dtype=torch.float32, device=device)
              for k in VIEW_KEYS}
    G, knn = build_view_graphs(views, int(cfg['knn_k']))
    ei = build_edge_index_dict(
        _pairs_checked_noerr(data['train_pairs']),
        knn['mi_v1'], knn['dis_v1'])
    g_sha0 = {k: _sha_bytes(v.numpy().tobytes()) for k, v in G.items()}
    e_sha0 = {str(k): _sha_bytes(v.numpy().tobytes())
              for k, v in ei.items()}

    model = pg_model.PaperGuidedS1(
        {k: inputs[k].shape[1] for k in VIEW_KEYS},
        n_types=int(cfg['n_types']), hidden=int(cfg['hidden']),
        readout=int(cfg['readout']), proj=int(cfg['proj']),
        hgt_layers=int(cfg['hgt_layers']),
        hgt_heads=int(cfg['hgt_heads']),
        hgt_dropout=float(cfg['hgt_dropout']),
        readout_dropout=float(cfg['readout_dropout']),
        leaky_slope=float(cfg['leaky_slope']),
        temperature_init=float(cfg['temperature_init']),
        temperature_min=float(cfg['temperature_min']),
        temperature_max=float(cfg['temperature_max']))
    opt = build_optimizer(model, cfg)
    tp = _pairs_checked_noerr(data['train_pairs'])
    vp = _pairs_checked_noerr(data['val_pairs'])
    tp_t = torch.tensor(tp, dtype=torch.long, device=device)
    vp_t = torch.tensor(vp, dtype=torch.long, device=device)
    y_tr = np.asarray(data['y_train'], dtype=np.float64)
    w = losses.class_weights_train(y_tr, beta=float(cfg['beta']))
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    trajectory, best = [], None
    aborted = None
    epoch = 0
    for epoch in range(1, int(cfg['max_epochs']) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        prev = _flat_params(model).clone()
        try:
            out = model(inputs, G, ei, tp_t)
            loss = losses.type_loss(out['logits'], y_tr, w_t)
        except (ValueError, FloatingPointError) as e:
            aborted = ('nonfinite input/logits rejected at epoch %d: %s'
                       % (epoch, e))
            break
        if not torch.isfinite(loss):
            aborted = ('nonfinite loss at epoch %d' % epoch)
            break
        loss.backward()
        gnorm = 0.0
        bad_grad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_grad = True
                break
            gnorm += float(p.grad.norm()) ** 2
        if bad_grad:
            aborted = ('nonfinite grad at epoch %d' % epoch)
            break
        gnorm = gnorm ** 0.5
        opt.step()
        upd = float((_flat_params(model) - prev).norm())
        rec = {'epoch': epoch, 'train_loss': float(loss.detach()),
               'grad_norm': gnorm, 'update_norm': upd}
        if epoch % int(cfg['eval_every']) == 0 or \
                epoch == int(cfg['max_epochs']):
            model.eval()
            try:
                with torch.no_grad():
                    tout = model(inputs, G, ei, tp_t)
                    vout = model(inputs, G, ei, vp_t)
                    if not (torch.isfinite(tout['logits']).all()
                            and torch.isfinite(vout['logits']).all()):
                        raise FloatingPointError(
                            'nonfinite eval logits')
                    tm_ = losses.type_metrics(
                        y_tr, tout['logits'].cpu().numpy())
                    vm = losses.type_metrics(
                        np.asarray(data['y_val']),
                        vout['logits'].cpu().numpy())
            except (ValueError, FloatingPointError) as e:
                aborted = ('nonfinite eval at epoch %d: %s'
                           % (epoch, e))
                trajectory.append(rec)
                break
            rec['train_macro_f1'] = tm_['primary']['macro_f1']
            rec['train_pred_hist'] = np.bincount(
                tout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            rec['train_true_hist'] = np.bincount(
                y_tr.argmax(1), minlength=int(cfg['n_types'])).tolist()
            rec['val_macro_f1'] = vm['primary']['macro_f1']
            rec['val_any_hit'] = vm['any_hit_rate']
            rec['val_pred_hist'] = np.bincount(
                vout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            rec['val_true_hist'] = np.bincount(
                np.asarray(data['y_val']).argmax(1),
                minlength=int(cfg['n_types'])).tolist()
            if best is None or rec['val_macro_f1'] > \
                    best['val_macro_f1']:
                best = dict(rec)
                torch.save(_ckpt(model, opt, epoch,
                                 rec['val_macro_f1'], G, ei, cfg),
                           os.path.join(out_dir, 'ckpt_best.pt'))
        trajectory.append(rec)
        log('epoch %d loss %.5f %s' % (
            epoch, rec['train_loss'],
            ('val_f1 %.4f' % rec['val_macro_f1'])
            if 'val_macro_f1' in rec else ''))

    # final checkpoint always written (even after abort — last state)
    torch.save(_ckpt(model, opt, epoch,
                     best['val_macro_f1'] if best else float('nan'),
                     G, ei, cfg),
               os.path.join(out_dir, 'ckpt_final.pt'))

    exports_from = None
    if aborted is None:
        # exports: train/val ONLY, from the BEST checkpoint state so the
        # files match a reloaded ckpt_best (declared)
        exports_from = 'final'
        best_ck = os.path.join(out_dir, 'ckpt_best.pt')
        if best is not None and os.path.exists(best_ck):
            model.load_state_dict(
                torch.load(best_ck, map_location='cpu',
                           weights_only=False)['model_state'])
            exports_from = 'best'
        model.eval()
        with torch.no_grad():
            tout = model(inputs, G, ei, tp_t)
            vout = model(inputs, G, ei, vp_t)
        meta = {'stage': 'S1', 'exports_from': exports_from,
                'best_epoch': best['epoch'] if best else None,
                'val_macro_f1': best['val_macro_f1'] if best else None}
        _export_predictions(out_dir, 'train', tp,
                            tout['logits'].cpu().numpy(), y_tr, meta)
        _export_predictions(out_dir, 'val', vp,
                            vout['logits'].cpu().numpy(),
                            np.asarray(data['y_val']), meta)

    manifest = dict(frozen)
    manifest.update({
        'optimizer': 'torch.optim.Adam(lr=%s, weight_decay=%s) '
        '[coordinator-selected baseline, not an author-code claim]'
        % (cfg['lr'], cfg['weight_decay']),
        'aborted': aborted,
        'best': best, 'final_epoch': epoch,
        'exports_from': exports_from,
        'trajectory': trajectory,
        'G_sha_start': g_sha0, 'ei_sha_start': e_sha0,
        'G_sha_end': {k: _sha_bytes(v.numpy().tobytes())
                      for k, v in G.items()},
        'ei_sha_end': {str(k): _sha_bytes(v.numpy().tobytes())
                       for k, v in ei.items()},
        'preflight': rep,
        'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    with open(os.path.join(out_dir, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


def _pairs_checked_noerr(arr):
    return np.asarray(arr, dtype=np.int64)


# ---------- CLI ----------------------------------------------------------
def main(argv=None):
    """Freeze-then-train two-step (documented reproducible procedure):
      1) runner.py --config c --data d --out o --preflight-only
         [--emit-manifest expected.json]
         -> validates, writes a FULL candidate expected manifest for
            review; creates NO run outputs, constructs NO model.
      2) runner.py --config c --data d --out o
         --expected-manifest expected.json
         -> refuses unless a COMPLETE reviewed manifest matches the
            frozen one; then trains.
    """
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--data', help='flat npz data bundle')
    p.add_argument('--out', required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--expected-manifest',
                   help='REQUIRED for training: reviewed full expected '
                        'manifest JSON (from --preflight-only '
                        '--emit-manifest)')
    p.add_argument('--emit-manifest',
                   help='preflight-only: write the candidate expected '
                        'manifest to this path for review')
    a = p.parse_args(argv)
    try:
        cfg = json.load(open(a.config, encoding='utf-8'))
        data = load_data_npz(a.data) if a.data else {}
        rep = preflight(cfg, data)
        envv = env_check(cfg)
        frozen = frozen_manifest(cfg, rep, envv)
        if a.preflight_only:
            frozen['manifest_kind'] = 'candidate_expected'
            if a.emit_manifest:
                with open(a.emit_manifest, 'w',
                          encoding='utf-8') as f:
                    json.dump(frozen, f, indent=1, default=str)
            print('PREFLIGHT-ONLY PASS (no model constructed)')
            print(json.dumps(frozen, indent=1)[:2000])
            return 0
        if not a.expected_manifest:
            print('FAIL: --expected-manifest is REQUIRED for training; '
                  'generate one with --preflight-only --emit-manifest',
                  file=sys.stderr)
            return 3
        check_expected_manifest(
            frozen, json.load(open(a.expected_manifest,
                                   encoding='utf-8')))
        if os.path.isdir(a.out) and os.listdir(a.out):
            print('occupied namespace: %s' % a.out, file=sys.stderr)
            return 3
        m = train_s1(cfg, data, a.out,
                     expected_manifest=json.load(
                         open(a.expected_manifest, encoding='utf-8')))
        print('DONE best', m['best'], 'aborted', m['aborted'])
        return 2 if m['aborted'] else 0
    except (ValueError, KeyError, FileExistsError) as e:
        print('FAIL: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
