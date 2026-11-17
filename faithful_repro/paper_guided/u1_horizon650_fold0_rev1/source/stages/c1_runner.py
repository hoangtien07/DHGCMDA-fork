# -*- coding: utf-8 -*-
"""C1 runner — S1 + paper Eq. 6-9 intra-modality contrastive loss.

SCOPE (declared): everything identical to frozen S1 except the single
added objective

    L_total = L_type + lam_intra * L_intra,
    L_intra = (intra_mi['sum'] + intra_dis['sum']) / (n_mi + n_dis)

where intra_* = stages/intra.intra_view_loss(proj_head(z_v1),
proj_head(z_v2), t=intra_temperature, alpha=intra_alpha) over the
per-view pre-fusion embeddings returned by the SAME encoder forward
(out['views']). This node-weighted mean is a declared engineering
reduction — NOT paper-literal lambda equivalence (paper Eq. 9 is a raw
sum; see stages/INTRA_CONTRACT.md and the mechanism audit M09).

- Uses the existing initialized model.mi_proj_head / dis_proj_head —
  no second stochastic forward, no new initialization draws.
- lam_intra == 0 skips the entire intra branch (no proj-head calls),
  which must reproduce frozen S1 loss/logits/grads/updates/RNG
  bit-for-bit on paired fixtures.
- No inter / reconstruction / existence / dynamic paths anywhere.
- Frozen S1 runtime files are imported read-only, never edited or
  monkeypatched. manifest['stage'] is 'C1' (never 'S1').

Run order mirrors S1 hard gates: preflight -> env_check -> C1 frozen
manifest -> COMPLETE expected-manifest compare BEFORE any model or
output creation -> static graphs -> Adam loop -> nonfinite abort
(exit 2) -> train/val exports only.

Data NPZ schema identical to S1 (flat, no test keys).
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
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runner as s1r          # frozen S1 helpers, imported read-only
import intra as intra_mod     # stages/intra.py — verified pure loss

VIEW_KEYS = s1r.VIEW_KEYS
ALLOWED_DATA_KEYS = s1r.ALLOWED_DATA_KEYS
FORBIDDEN_DATA_KEYS = s1r.FORBIDDEN_DATA_KEYS

INTRA_CFG_KEYS = ('lam_intra', 'intra_temperature', 'intra_alpha')


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_path(p):
    return _sha_bytes(open(p, 'rb').read())


# ---------- gates --------------------------------------------------------
def preflight_c1(cfg, data):
    """S1 preflight PLUS the C1 intra parameter contract."""
    rep = s1r.preflight(cfg, data)
    for k in INTRA_CFG_KEYS:
        if k not in cfg:
            raise ValueError('preflight failed: missing cfg.' + k)
    if not (float(cfg['lam_intra']) >= 0.0
            and np.isfinite(float(cfg['lam_intra']))):
        raise ValueError('preflight failed: lam_intra must be finite >=0')
    if not (0.0 < float(cfg['intra_temperature'])
            and np.isfinite(float(cfg['intra_temperature']))):
        raise ValueError('preflight failed: intra_temperature must be '
                         'positive finite')
    if not (0.0 <= float(cfg['intra_alpha']) <= 1.0):
        raise ValueError('preflight failed: intra_alpha must be in [0,1]')
    rep['stage'] = 'C1'
    return rep


def env_check(cfg):
    return s1r.env_check(cfg)


def source_hashes_c1():
    """Effective C1 stage sources: the scoped runner and the pure
    intra implementation it calls."""
    return {
        'stages/c1_runner.py': _sha_path(os.path.abspath(__file__)),
        'stages/intra.py': _sha_path(
            os.path.join(_HERE, 'intra.py')),
    }


def imported_pure_hashes_c1():
    """Frozen S1 + Task1 pure sources consumed by C1 — the C1 expected
    manifest fails closed if ANY of these drift."""
    out = {}
    for fn in ('model.py', 'losses.py', 'runner.py',
               'contracts.py', 'metrics.py'):
        p = os.path.join(_PARENT, fn)
        if not os.path.exists(p):
            raise ValueError('missing imported source ' + fn)
        out[fn] = _sha_path(p)
    return out


def frozen_manifest_c1(cfg, preflight_rep, env_versions):
    """C1 manifest — same schema as S1, stage 'C1', C1 source hashes."""
    return {
        'stage': 'C1',
        'cfg': dict(cfg),
        'env': dict(env_versions,
                    python=platform.python_version(),
                    platform=platform.platform()),
        'data_hashes': preflight_rep['data_hashes'],
        'view_hashes': preflight_rep['view_hashes'],
        'n_pairs': preflight_rep['n_pairs'],
        'source_hashes': source_hashes_c1(),
        'imported_pure_hashes': imported_pure_hashes_c1(),
    }


EXPECTED_MANIFEST_KEYS = s1r.EXPECTED_MANIFEST_KEYS
EXPECTED_MANIFEST_META_KEYS = s1r.EXPECTED_MANIFEST_META_KEYS


def check_expected_manifest(frozen, expected):
    s1r.check_expected_manifest(frozen, expected)
    if expected.get('stage') != 'C1':
        raise ValueError('expected manifest stage must be C1, got %r'
                         % expected.get('stage'))


def expected_manifest_for_c1(cfg, data):
    rep = preflight_c1(cfg, data)
    envv = env_check(cfg)
    m = frozen_manifest_c1(cfg, rep, envv)
    m['manifest_kind'] = 'candidate_expected'
    return m


# ---------- intra term ---------------------------------------------------
def _intra_terms(model, out, cfg):
    """Compute the intra term ONCE from the current forward's views.
    Returns dict of detached floats + the live scalar (or None when
    lam_intra==0 — the branch is skipped entirely, preserving S1 RNG,
    gradients and update behavior exactly)."""
    lam = float(cfg['lam_intra'])
    if lam == 0.0:
        return None
    z1m, z2m, z1d, z2d = out['views']
    t = float(cfg['intra_temperature'])
    a = float(cfg['intra_alpha'])
    mi = intra_mod.intra_view_loss(
        model.mi_proj_head(z1m), model.mi_proj_head(z2m),
        temperature=t, alpha=a)
    di = intra_mod.intra_view_loss(
        model.dis_proj_head(z1d), model.dis_proj_head(z2d),
        temperature=t, alpha=a)
    n_mi, n_dis = z1m.shape[0], z1d.shape[0]
    l_intra = (mi['sum'] + di['sum']) / (n_mi + n_dis)
    return {'lam': lam,
            'L_intra': l_intra,
            'intra_mi_sum': float(mi['sum'].detach()),
            'intra_dis_sum': float(di['sum'].detach()),
            'intra_mi_mean': float(mi['mean'].detach()),
            'intra_dis_mean': float(di['mean'].detach()),
            'n_mi': n_mi, 'n_dis': n_dis}


# ---------- training -----------------------------------------------------
def train_c1(cfg, data, out_dir, log=print, expected_manifest=None):
    """Train C1 (S1 + intra) on caller-supplied fold. Mirrors S1's
    gates: complete expected manifest REQUIRED before model/output;
    nonfinite abort keeps last-good checkpoint and exports nothing.
    """
    import torch
    import model as pg_model
    import losses

    rep = preflight_c1(cfg, data)
    envv = env_check(cfg)
    frozen = frozen_manifest_c1(cfg, rep, envv)
    check_expected_manifest(frozen, expected_manifest)
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
    G, knn = s1r.build_view_graphs(views, int(cfg['knn_k']))
    ei = s1r.build_edge_index_dict(
        s1r._pairs_checked_noerr(data['train_pairs']),
        knn['mi_v1'], knn['dis_v1'])
    g_sha0 = {k: _sha_bytes(v.numpy().tobytes()) for k, v in G.items()}
    e_sha0 = {str(k): _sha_bytes(v.cpu().numpy().tobytes())
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
    opt = s1r.build_optimizer(model, cfg)
    tp = s1r._pairs_checked_noerr(data['train_pairs'])
    vp = s1r._pairs_checked_noerr(data['val_pairs'])
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
        prev = s1r._flat_params(model).clone()
        try:
            out = model(inputs, G, ei, tp_t)
            l_type = losses.type_loss(out['logits'], y_tr, w_t)
            terms = _intra_terms(model, out, cfg)
            loss = l_type if terms is None else \
                l_type + terms['lam'] * terms['L_intra']
        except (ValueError, FloatingPointError) as e:
            aborted = ('nonfinite input/logits/intra rejected at epoch '
                       '%d: %s' % (epoch, e))
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
        upd = float((s1r._flat_params(model) - prev).norm())
        rec = {'epoch': epoch,
               'train_loss': float(loss.detach()),
               'L_type': float(l_type.detach()),
               'grad_norm': gnorm, 'update_norm': upd}
        if terms is not None:
            rec['lam_applied'] = terms['lam']
            rec['L_intra'] = float(terms['L_intra'].detach())
            rec['intra_mi_sum'] = terms['intra_mi_sum']
            rec['intra_dis_sum'] = terms['intra_dis_sum']
            rec['intra_mi_mean'] = terms['intra_mi_mean']
            rec['intra_dis_mean'] = terms['intra_dis_mean']
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
            rec['train_micro_f1'] = tm_['micro_f1']
            rec['train_pred_hist'] = np.bincount(
                tout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            rec['val_macro_f1'] = vm['primary']['macro_f1']
            rec['val_micro_f1'] = vm['micro_f1']
            rec['val_any_hit'] = vm['any_hit_rate']
            rec['val_pred_hist'] = np.bincount(
                vout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            if best is None or rec['val_macro_f1'] > \
                    best['val_macro_f1']:
                best = dict(rec)
                torch.save(s1r._ckpt(model, opt, epoch,
                                     rec['val_macro_f1'], G, ei, cfg),
                           os.path.join(out_dir, 'ckpt_best.pt'))
        trajectory.append(rec)
        log('epoch %d loss %.5f %s' % (
            epoch, rec['train_loss'],
            ('val_f1 %.4f' % rec['val_macro_f1'])
            if 'val_macro_f1' in rec else ''))

    torch.save(s1r._ckpt(model, opt, epoch,
                         best['val_macro_f1'] if best else float('nan'),
                         G, ei, cfg),
               os.path.join(out_dir, 'ckpt_final.pt'))

    exports_from = None
    if aborted is None:
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
        meta = {'stage': 'C1', 'exports_from': exports_from,
                'lam_intra': float(cfg['lam_intra']),
                'intra_temperature': float(cfg['intra_temperature']),
                'intra_alpha': float(cfg['intra_alpha']),
                'intra_reduction': '(mi.sum+dis.sum)/(n_mi+n_dis) '
                '— declared engineering choice, not paper-literal '
                'lambda equivalence',
                'best_epoch': best['epoch'] if best else None,
                'val_macro_f1': best['val_macro_f1'] if best else None}
        s1r._export_predictions(out_dir, 'train', tp,
                                tout['logits'].cpu().numpy(), y_tr, meta)
        s1r._export_predictions(out_dir, 'val', vp,
                                vout['logits'].cpu().numpy(),
                                np.asarray(data['y_val']), meta)

    manifest = dict(frozen)
    manifest.update({
        'optimizer': 'torch.optim.Adam(lr=%s, weight_decay=%s) '
        '[coordinator-selected baseline, not an author-code claim]'
        % (cfg['lr'], cfg['weight_decay']),
        'intra': {'lam_intra': float(cfg['lam_intra']),
                  'temperature': float(cfg['intra_temperature']),
                  'alpha': float(cfg['intra_alpha']),
                  'reduction': '(mi.sum+dis.sum)/(n_mi+n_dis)',
                  'declared': 'engineering reduction, not '
                              'paper-literal lambda equivalence'},
        'aborted': aborted,
        'best': best, 'final_epoch': epoch,
        'exports_from': exports_from,
        'trajectory': trajectory,
        'G_sha_start': g_sha0, 'ei_sha_start': e_sha0,
        'G_sha_end': {k: _sha_bytes(v.numpy().tobytes())
                      for k, v in G.items()},
        'ei_sha_end': {str(k): _sha_bytes(v.cpu().numpy().tobytes())
                       for k, v in ei.items()},
        'preflight': rep,
        'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    with open(os.path.join(out_dir, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


# ---------- CLI ----------------------------------------------------------
def main(argv=None):
    """C1 freeze-then-train (same two-step as S1):
      1) c1_runner.py --config c --data d --out o --preflight-only
         [--emit-manifest expected.json]
      2) c1_runner.py --config c --data d --out o
         --expected-manifest expected.json
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
        data = s1r.load_data_npz(a.data) if a.data else {}
        rep = preflight_c1(cfg, data)
        envv = env_check(cfg)
        frozen = frozen_manifest_c1(cfg, rep, envv)
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
        m = train_c1(cfg, data, a.out,
                     expected_manifest=json.load(
                         open(a.expected_manifest, encoding='utf-8')))
        print('DONE best', m['best'], 'aborted', m['aborted'])
        return 2 if m['aborted'] else 0
    except (ValueError, KeyError, FileExistsError) as e:
        print('FAIL: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
