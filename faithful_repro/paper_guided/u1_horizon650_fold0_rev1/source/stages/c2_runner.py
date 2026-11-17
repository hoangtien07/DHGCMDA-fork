# -*- coding: utf-8 -*-
"""C2 runner — S1 + paper Eq. 10-15 inter-modality contrastive loss.

SCOPE (declared): everything identical to frozen S1 except the single
added objective

    L_total = L_type + lam_inter * L_inter,
    L_inter = info_mean + margin_coef * margin_term        (Eq. 13-15)

computed by stages/inter.inter_modality_loss over the FUSED PRE-HGT
embeddings out['zm_pre']/out['zd_pre'] (Eq. 10 inputs ZM/ZD) from the
SAME encoder forward — no projection heads on this path, no second
stochastic pass, no new initialization.

- Positive mask = train_mask (A_train) only; the sampled negative bank
  is built ONCE pre-model by inter.sample_negative_bank with an
  independent seeded torch.Generator — never touches global RNG, never
  reads held-out labels. Unknown cells are ASSUMED negatives, not
  verified negatives (false-negative risk disclosed in manifest).
- lam_inter == 0 skips the inter branch AND the bank entirely,
  reproducing frozen S1 loss/logits/grads/updates/RNG bit-for-bit.
- No intra / reconstruction / existence / dynamic paths anywhere.
- margin_coef=.1 and margin=.5 are declared official-code fallbacks,
  not recovered paper values. All mean reductions are declared
  normalized engineering units — NOT paper-literal lambda equivalence.
- Frozen S1 runtime files imported read-only; manifest['stage']='C2'.

Run order mirrors S1/C1 hard gates: preflight -> env_check -> C2 frozen
manifest (incl. bank sha/counts) -> COMPLETE expected-manifest compare
BEFORE model/output -> static graphs -> Adam loop -> nonfinite abort
(exit 2) -> train/val exports only. Actual bank is stored in every
checkpoint (graph+bank coherent resume state).
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
import inter as inter_mod    # stages/inter.py — verified pure loss

VIEW_KEYS = s1r.VIEW_KEYS
ALLOWED_DATA_KEYS = s1r.ALLOWED_DATA_KEYS
FORBIDDEN_DATA_KEYS = s1r.FORBIDDEN_DATA_KEYS

INTER_CFG_KEYS = ('lam_inter', 'inter_temperature', 'inter_margin',
                  'inter_margin_coef', 'inter_cap_mult', 'inter_seed')


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_path(p):
    return _sha_bytes(open(p, 'rb').read())


# ---------- negative bank -------------------------------------------------
def build_negative_bank(cfg, data):
    """Deterministic inter negative bank over A_train-zero cells —
    computed from train_mask ONLY (preflight already proved
    train_mask == train_pairs exactly). Uses an independent local
    torch.Generator; global RNG untouched. Returns (bank ndarray bool,
    sha256, per-row counts list)."""
    import torch
    tm = np.asarray(data['train_mask']).astype(bool)
    g0 = torch.get_rng_state()
    bank = inter_mod.sample_negative_bank(
        torch.tensor(tm), int(cfg['inter_seed']),
        float(cfg['inter_cap_mult']))
    assert torch.equal(torch.get_rng_state(), g0)   # internal audit
    bank_np = bank.numpy()
    if (bank_np & tm).any():
        raise ValueError('negative bank overlaps train positives')
    sha = _sha_bytes(np.ascontiguousarray(bank_np).tobytes())
    return bank_np, sha, bank_np.sum(axis=1).tolist()


# ---------- gates ---------------------------------------------------------
def preflight_c2(cfg, data):
    rep = s1r.preflight(cfg, data)
    for k in INTER_CFG_KEYS:
        if k not in cfg:
            raise ValueError('preflight failed: missing cfg.' + k)
    if not (float(cfg['lam_inter']) >= 0.0
            and np.isfinite(float(cfg['lam_inter']))):
        raise ValueError('preflight failed: lam_inter must be finite >=0')
    if not (0.0 < float(cfg['inter_temperature'])
            and np.isfinite(float(cfg['inter_temperature']))):
        raise ValueError('preflight failed: inter_temperature positive')
    if not (0.0 <= float(cfg['inter_margin'])
            and np.isfinite(float(cfg['inter_margin']))):
        raise ValueError('preflight failed: inter_margin finite >=0')
    if not (0.0 <= float(cfg['inter_margin_coef'])
            and np.isfinite(float(cfg['inter_margin_coef']))):
        raise ValueError('preflight failed: inter_margin_coef >=0')
    if not (0.0 <= float(cfg['inter_cap_mult'])
            and np.isfinite(float(cfg['inter_cap_mult']))):
        raise ValueError('preflight failed: inter_cap_mult >=0')
    rep['stage'] = 'C2'
    return rep


def env_check(cfg):
    return s1r.env_check(cfg)


def source_hashes_c2():
    return {
        'stages/c2_runner.py': _sha_path(os.path.abspath(__file__)),
        'stages/inter.py': _sha_path(os.path.join(_HERE, 'inter.py')),
        'stages/c2_model.py': _sha_path(
            os.path.join(_HERE, 'c2_model.py')),
    }


def imported_pure_hashes_c2():
    out = {}
    for fn in ('model.py', 'losses.py', 'runner.py',
               'contracts.py', 'metrics.py'):
        p = os.path.join(_PARENT, fn)
        if not os.path.exists(p):
            raise ValueError('missing imported source ' + fn)
        out[fn] = _sha_path(p)
    return out


def frozen_manifest_c2(cfg, preflight_rep, env_versions, bank_sha,
                       bank_counts):
    return {
        'stage': 'C2',
        'cfg': dict(cfg),
        'env': dict(env_versions,
                    python=platform.python_version(),
                    platform=platform.platform()),
        'data_hashes': preflight_rep['data_hashes'],
        'view_hashes': preflight_rep['view_hashes'],
        'n_pairs': preflight_rep['n_pairs'],
        'source_hashes': source_hashes_c2(),
        'imported_pure_hashes': imported_pure_hashes_c2(),
        'neg_bank': {'sha256': bank_sha,
                     'n_neg': int(sum(bank_counts)),
                     'per_anchor_nonzero': int(sum(
                         1 for c in bank_counts if c > 0)),
                     'cap_mult': float(cfg['inter_cap_mult']),
                     'seed': int(cfg['inter_seed']),
                     'provenance': 'A_train-zero cells, ASSUMED '
                                   'negatives (not verified); local '
                                   'seeded generator, no held-out '
                                   'label access'},
    }


EXPECTED_MANIFEST_KEYS = s1r.EXPECTED_MANIFEST_KEYS + ('neg_bank',)
EXPECTED_MANIFEST_META_KEYS = s1r.EXPECTED_MANIFEST_META_KEYS


def check_expected_manifest(frozen, expected):
    """C2 variant of the S1 gate: same key/schema semantics plus the
    required 'neg_bank' provenance block compared field-by-field."""
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
                diffs.append('env: expected a map')
                continue
            for k in set(ea) | set(eb):
                if k == 'platform':
                    continue
                cmp('env.' + k, eb.get(k), ea.get(k))
        else:
            cmp(key, frozen.get(key), expected.get(key))
    if expected.get('stage') != 'C2':
        diffs.append('stage must be C2, got %r'
                     % (expected.get('stage'),))
    if diffs:
        raise ValueError('expected-manifest mismatch:\n'
                         + '\n'.join(diffs))


def expected_manifest_for_c2(cfg, data):
    rep = preflight_c2(cfg, data)
    envv = env_check(cfg)
    _, sha, counts = build_negative_bank(cfg, data)
    m = frozen_manifest_c2(cfg, rep, envv, sha, counts)
    m['manifest_kind'] = 'candidate_expected'
    return m


# ---------- inter term ----------------------------------------------------
def _inter_terms(model_out, pos_t, neg_t, cfg):
    """Inter term ONCE from the current forward's fused pre-HGT
    embeddings. Returns None when lam_inter==0 (branch fully skipped,
    S1-exact)."""
    lam = float(cfg['lam_inter'])
    if lam == 0.0:
        return None
    r = inter_mod.inter_modality_loss(
        model_out['zm_pre'], model_out['zd_pre'], pos_t, neg_t,
        temperature=float(cfg['inter_temperature']),
        margin=float(cfg['inter_margin']),
        margin_coef=float(cfg['inter_margin_coef']))
    return {'lam': lam, 'L_inter': r['l_inter'],
            'inter_info_sum': float(r['info_sum'].detach()),
            'inter_info_mean': float(r['info_mean'].detach()),
            'inter_pos_mean_sim': float(r['pos_mean_sim'].detach()),
            'inter_neg_mean_sim': float(r['neg_mean_sim'].detach()),
            'inter_margin_term': float(r['margin_term'].detach()),
            'inter_n_pos': r['n_pos'], 'inter_n_neg': r['n_neg']}


# ---------- checkpoints ----------------------------------------------------
def _ckpt_c2(model, opt, epoch, val_metric, G, ei, cfg, bank, bank_sha):
    ck = s1r._ckpt(model, opt, epoch, val_metric, G, ei, cfg)
    ck['neg_bank'] = bank              # ACTUAL bank tensor, not hash-only
    ck['neg_bank_sha'] = bank_sha
    return ck


# ---------- training -------------------------------------------------------
def train_c2(cfg, data, out_dir, log=print, expected_manifest=None):
    """Train C2 (S1 + inter) on caller-supplied fold. Same gates as S1:
    complete expected manifest REQUIRED before model/output; nonfinite
    abort keeps last-good checkpoint and exports nothing."""
    import torch
    import losses
    import c2_model as c2m

    rep = preflight_c2(cfg, data)
    envv = env_check(cfg)
    bank, bank_sha, bank_counts = build_negative_bank(cfg, data)
    frozen = frozen_manifest_c2(cfg, rep, envv, bank_sha, bank_counts)
    check_expected_manifest(frozen, expected_manifest)
    os.makedirs(out_dir, exist_ok=False)
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

    model = c2m.PaperGuidedC2(
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
    pos_t = torch.tensor(np.asarray(data['train_mask']).astype(bool),
                         device=device)
    neg_t = torch.tensor(bank, device=device)

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
            terms = _inter_terms(out, pos_t, neg_t, cfg)
            loss = l_type if terms is None else \
                l_type + terms['lam'] * terms['L_inter']
        except (ValueError, FloatingPointError) as e:
            aborted = ('nonfinite input/logits/inter rejected at epoch '
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
            rec['L_inter'] = float(terms['L_inter'].detach())
            rec['inter_info_sum'] = terms['inter_info_sum']
            rec['inter_info_mean'] = terms['inter_info_mean']
            rec['inter_pos_mean_sim'] = terms['inter_pos_mean_sim']
            rec['inter_neg_mean_sim'] = terms['inter_neg_mean_sim']
            rec['inter_margin_term'] = terms['inter_margin_term']
            rec['inter_n_pos'] = terms['inter_n_pos']
            rec['inter_n_neg'] = terms['inter_n_neg']
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
            rec['train_any_hit'] = tm_['any_hit_rate']
            rec['train_pred_hist'] = np.bincount(
                tout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            rec['train_true_hist'] = np.bincount(
                np.argmax(np.asarray(y_tr), 1),
                minlength=int(cfg['n_types'])).tolist()
            rec['val_macro_f1'] = vm['primary']['macro_f1']
            rec['val_micro_f1'] = vm['micro_f1']
            rec['val_any_hit'] = vm['any_hit_rate']
            rec['val_pred_hist'] = np.bincount(
                vout['logits'].argmax(1).cpu().numpy(),
                minlength=int(cfg['n_types'])).tolist()
            rec['val_true_hist'] = np.bincount(
                np.argmax(np.asarray(data['y_val']), 1),
                minlength=int(cfg['n_types'])).tolist()
            if best is None or rec['val_macro_f1'] > \
                    best['val_macro_f1']:
                best = dict(rec)
                torch.save(_ckpt_c2(model, opt, epoch,
                                    rec['val_macro_f1'], G, ei, cfg,
                                    neg_t, bank_sha),
                           os.path.join(out_dir, 'ckpt_best.pt'))
        trajectory.append(rec)
        log('epoch %d loss %.5f %s' % (
            epoch, rec['train_loss'],
            ('val_f1 %.4f' % rec['val_macro_f1'])
            if 'val_macro_f1' in rec else ''))

    torch.save(_ckpt_c2(model, opt, epoch,
                        best['val_macro_f1'] if best else float('nan'),
                        G, ei, cfg, neg_t, bank_sha),
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
        meta = {'stage': 'C2', 'exports_from': exports_from,
                'lam_inter': float(cfg['lam_inter']),
                'inter_temperature': float(cfg['inter_temperature']),
                'inter_margin': float(cfg['inter_margin']),
                'inter_margin_coef': float(cfg['inter_margin_coef']),
                'inter_reduction': 'info_mean + margin_coef*margin_term'
                ' — declared normalized engineering units, not '
                'paper-literal lambda equivalence',
                'neg_bank_sha': bank_sha,
                'neg_bank_assumed': 'A_train-zero cells, not verified '
                'negatives (false-negative risk disclosed)',
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
        'inter': {'lam_inter': float(cfg['lam_inter']),
                  'temperature': float(cfg['inter_temperature']),
                  'margin': float(cfg['inter_margin']),
                  'margin_coef': float(cfg['inter_margin_coef']),
                  'cap_mult': float(cfg['inter_cap_mult']),
                  'seed': int(cfg['inter_seed']),
                  'bank_sha256': bank_sha,
                  'bank_per_anchor_counts': bank_counts,
                  'embeddings': 'fused PRE-HGT zm_pre/zd_pre (Eq.10), '
                                'same forward, no proj heads',
                  'reduction': 'info_mean + margin_coef*margin_term — '
                               'declared normalized engineering units'},
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


# ---------- CLI -------------------------------------------------------------
def main(argv=None):
    """C2 freeze-then-train (same two-step as S1/C1):
      1) c2_runner.py --config c --data d --out o --preflight-only
         [--emit-manifest expected.json]
      2) c2_runner.py --config c --data d --out o
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
        rep = preflight_c2(cfg, data)
        envv = env_check(cfg)
        _, bsha, bcnt = build_negative_bank(cfg, data)
        frozen = frozen_manifest_c2(cfg, rep, envv, bsha, bcnt)
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
        m = train_c2(cfg, data, a.out,
                     expected_manifest=json.load(
                         open(a.expected_manifest, encoding='utf-8')))
        print('DONE best', m['best'], 'aborted', m['aborted'])
        return 2 if m['aborted'] else 0
    except (ValueError, KeyError, FileExistsError) as e:
        print('FAIL: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
