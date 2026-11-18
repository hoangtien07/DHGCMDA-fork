# -*- coding: utf-8 -*-
"""U1X training runner — pinned u1_runner.train_u1 loop + ONE added term.

Identical discipline to stages/u1_runner.py (same imports, gates,
dynamic refresh, recon, inter/intra terms, eval cadence, best/final
checkpoints, exports, manifest) with exactly these declared deltas
(PROTOCOL_CVTRIPLET_U1_REV2):

  * model = PaperGuidedU1X (C4 + exist_rel channel)
  * training forward on pair_idx_full = train_pairs ++ neg_exist_idx
    (one-time pinned mask-zero subsample, 10x|pos|, seed+77);
    L_exist = focal(exist[:n_pos], exist[n_pos:]) upstream port;
    loss = combine_objectives(U1 terms) + lam_exist * L_exist
  * expected-manifest compare is the same idea but stage 'U1X'
  * extra sanity record per eval: exist mean on val pairs vs a pinned
    mask-zero probe sample (design-validation signal, not a result)

Everything else bit-for-bit the U1 recipe.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                    'paper_guided', 'u1_horizon650_fold0_rev1', 'source')
for _p in (_SRC, os.path.join(_SRC, 'stages'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import losses                    # noqa: E402  (pinned S1 losses)
import losses_u1x                # noqa: E402
import runner as s1r             # noqa: E402
import u1_runner                 # noqa: E402
import c2_runner as c2r          # noqa: E402
import c3_runner as c3r          # noqa: E402
import dynamic_graph as dg       # noqa: E402
from model_u1x import PaperGuidedU1X  # noqa: E402
from objective_units import combine_objectives  # noqa: E402

_sha_bytes = u1_runner._sha_bytes

_EXIST_SAMPLING_NOTE = (
    'one-time default_rng(seed+77) subsample of train-mask-zero cells, '
    'min(10x n_pos, avail) — port of upstream single torch.randperm '
    'subsample; ASSUMED negatives, held-out positives included (same '
    'semantics as upstream zero_index)')


def _exist_supervision_block(cfg, neg_sha, n_neg):
    return {'neg_sha256': neg_sha, 'n_neg': int(n_neg),
            'sampling': _EXIST_SAMPLING_NOTE,
            'focal_gamma': losses_u1x.FOCAL_GAMMA,
            'focal_alpha': losses_u1x.FOCAL_ALPHA,
            'lam_exist': float(cfg['lam_exist'])}


def _sha_path(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def expected_manifest_for_u1x(cfg, data, meta_path=None, audit_path=None,
                              data_file_sha=None, neg_exist_idx=None,
                              neg_exist_sha=None):
    m = u1_runner.expected_manifest_for_u1(
        cfg, data, meta_path=meta_path, audit_path=audit_path,
        data_file_sha=data_file_sha)
    m['stage'] = 'U1X'
    m['exist_supervision'] = _exist_supervision_block(
        cfg, neg_exist_sha, len(neg_exist_idx))
    return m


def _check_expected(frozen, expected):
    if not isinstance(expected, dict):
        raise ValueError('expected manifest must be a JSON object')
    exp = dict(expected)
    exp.pop('manifest_kind', None)
    diffs = []

    def cmp(path, a, b):
        if a != b:
            diffs.append('%s: expected %r got %r' % (path, b, a))

    for k in set(exp) | set(frozen):
        if k == 'env':
            ea, eb = exp['env'], frozen['env']
            for kk in set(ea) | set(eb):
                if kk == 'platform':
                    continue
                cmp('env.' + kk, eb.get(kk), ea.get(kk))
        else:
            cmp(k, frozen.get(k), exp.get(k))
    if exp.get('stage') != 'U1X':
        diffs.append('stage must be U1X')
    if diffs:
        raise ValueError('expected-manifest mismatch:\n'
                         + '\n'.join(diffs))


def train_u1x(cfg, data, out_dir, neg_exist_idx, log=print,
              expected_manifest=None, meta_path=None, audit_path=None,
              data_file_sha=None):
    """train_u1 + existence term. neg_exist_idx: (N,2) mask-zero pairs
    from the pinned contract (pre-hashed in preflight)."""
    import torch
    rep = u1_runner.preflight_u1(cfg, data)
    rep['stage'] = 'U1X'
    envv = u1_runner.env_check(cfg)
    bank, bank_sha, bank_counts = c2r.build_negative_bank(cfg, data)
    recon_block, artifacts = u1_runner._resolve_recon(
        cfg, data, meta_path, audit_path, data_file_sha)
    frozen = u1_runner.frozen_manifest_u1(
        cfg, rep, envv, bank_sha, bank_counts, recon_block)
    frozen['stage'] = 'U1X'
    neg_np = np.asarray(neg_exist_idx, dtype=np.int64)
    neg_sha = _sha_bytes(np.ascontiguousarray(neg_np).tobytes())
    frozen['exist_supervision'] = _exist_supervision_block(
        cfg, neg_sha, len(neg_np))
    _check_expected(frozen, expected_manifest)
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, 'manifest_frozen.json'), 'w',
              encoding='utf-8') as f:
        json.dump(frozen, f, indent=1, default=str)
    if artifacts is not None:
        u1_runner._persist_targets(out_dir, artifacts)

    device = torch.device('cpu')
    torch.manual_seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']) % (2 ** 32))

    views = data['views']
    inputs = {k: torch.tensor(np.asarray(views[k]),
                              dtype=torch.float32, device=device)
              for k in s1r.VIEW_KEYS}
    G, knn = s1r.build_view_graphs(views, int(cfg['knn_k']))
    ei = s1r.build_edge_index_dict(
        s1r._pairs_checked_noerr(data['train_pairs']),
        knn['mi_v1'], knn['dis_v1'])
    g_sha0 = {k: _sha_bytes(v.numpy().tobytes()) for k, v in G.items()}
    e_sha0 = {str(k): _sha_bytes(v.cpu().numpy().tobytes())
              for k, v in ei.items()}

    model = PaperGuidedU1X(
        {k: inputs[k].shape[1] for k in s1r.VIEW_KEYS},
        n_types=int(cfg['n_types']), hidden=int(cfg['hidden']),
        readout=int(cfg['readout']), proj=int(cfg['proj']),
        hgt_layers=int(cfg['hgt_layers']),
        hgt_heads=int(cfg['hgt_heads']),
        hgt_dropout=float(cfg['hgt_dropout']),
        readout_dropout=float(cfg['readout_dropout']),
        leaky_slope=float(cfg['leaky_slope']),
        temperature_init=float(cfg['temperature_init']),
        temperature_min=float(cfg['temperature_min']),
        temperature_max=float(cfg['temperature_max']),
        recon_enabled=(float(cfg['lam_recon']) != 0.0),
        recon_seed=int(cfg['seed']),
        exist_seed=int(cfg['seed']))
    opt = s1r.build_optimizer(model, cfg)
    tp = s1r._pairs_checked_noerr(data['train_pairs'])
    vp = s1r._pairs_checked_noerr(data['val_pairs'])
    tp_t = torch.tensor(tp, dtype=torch.long, device=device)
    vp_t = torch.tensor(vp, dtype=torch.long, device=device)
    neg_exist_t = torch.tensor(neg_np, dtype=torch.long, device=device)
    pair_full_t = torch.cat([tp_t, neg_exist_t], dim=0)
    n_pos = len(tp_t)
    # sanity probe: pinned mask-zero cells, model-visible info only
    probe = np.random.default_rng(int(cfg['seed']) + 9000).choice(
        len(neg_np), size=min(len(vp_t), len(neg_np)), replace=False)
    probe_t = neg_exist_t[torch.tensor(
        np.sort(probe), dtype=torch.long, device=device)]
    y_tr = np.asarray(data['y_train'], dtype=np.float64)
    w = losses.class_weights_train(y_tr, beta=float(cfg['beta']))
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    unit_mass = float(
        (torch.tensor(y_tr, dtype=torch.float32)
         * w_t.view(1, -1)).sum())
    pos_t = torch.tensor(np.asarray(data['train_mask']).astype(bool),
                         device=device)
    neg_t = torch.tensor(bank, device=device)
    tensors = None
    if artifacts is not None:
        tensors = {
            'mi_target': torch.tensor(artifacts['mi_target'],
                                      dtype=torch.float32,
                                      device=device),
            'mi_valid': torch.tensor(artifacts['mi_valid_mask'],
                                     device=device),
            'dis_target': torch.tensor(artifacts['dis_target'],
                                       dtype=torch.float32,
                                       device=device),
            'dis_valid': torch.tensor(artifacts['dis_valid_mask'],
                                      device=device)}

    trajectory, best = [], None
    dynamic_state, dynamic_history = None, []
    training_forward_count = 0
    aborted = None
    epoch = 0
    for epoch in range(1, int(cfg['max_epochs']) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        prev = s1r._flat_params(model).clone()
        try:
            out = model(inputs, G, ei, pair_full_t)
            training_forward_count += 1
            l_type = losses.type_loss(out['logits'][:n_pos], y_tr, w_t)
            l_exist, exist_rec = losses_u1x.existence_focal(
                out['exist'], n_pos)
            it = c3r._intra_terms(model, out, cfg)
            et = u1_runner._inter_terms_u1(out, pos_t, neg_t, cfg)
            rt_ = u1_runner._recon_terms(model, out, tensors, cfg)
            loss = (combine_objectives(l_type, it, et, rt_, unit_mass,
                                       cfg)
                    + float(cfg['lam_exist']) * l_exist)
        except (ValueError, FloatingPointError) as e:
            aborted = ('nonfinite input/logits/term rejected at epoch '
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
        refresh_meta = None
        if cfg['dynamic_enabled'] and dg.should_update(epoch):
            try:
                old_g_sha = {k: _sha_bytes(v.numpy().tobytes())
                             for k, v in G.items()}
                old_ei_sha = {str(k): _sha_bytes(v.numpy().tobytes())
                              for k, v in ei.items()}
                dynamic_state = dg.refresh_similarity_graphs(
                    rt_['gram_mi'].detach(), rt_['gram_dis'].detach(),
                    ei, epoch, k=int(cfg['knn_k']),
                    threshold=cfg['dynamic_threshold'])
                refresh_meta = dict(dynamic_state['meta'])
                refresh_meta.update({
                    'producer_epoch': epoch,
                    'producer_phase': 'training_forward_pre_optimizer',
                    'application_phase': 'post_optimizer_pre_eval',
                    'producer_gram_mi_sha256': _sha_bytes(
                        rt_['gram_mi'].detach().numpy().tobytes()),
                    'producer_gram_dis_sha256': _sha_bytes(
                        rt_['gram_dis'].detach().numpy().tobytes()),
                    'input_G_sha': old_g_sha,
                    'input_ei_sha': old_ei_sha})
                dynamic_state['meta'] = refresh_meta
                G, ei = dynamic_state['G'], dynamic_state[
                    'edge_index_dict']
                dynamic_history.append(refresh_meta)
            except (ValueError, FloatingPointError) as e:
                aborted = ('dynamic refresh rejected at epoch %d: %s'
                           % (epoch, e))
                break
        upd = float((s1r._flat_params(model) - prev).norm())
        rec = {'epoch': epoch,
               'train_loss': float(loss.detach()),
               'L_type': float(l_type.detach()),
               'L_exist': exist_rec['L_exist'],
               'exist_pos_mean': exist_rec['exist_pos_mean'],
               'exist_neg_mean': exist_rec['exist_neg_mean'],
               'grad_norm': gnorm, 'update_norm': upd}
        if it is not None:
            rec['L_intra'] = float(it['L_intra'].detach())
        if et is not None:
            rec['L_inter'] = float(et['L_inter'].detach())
        if rt_ is not None:
            rec['L_recon'] = float(rt_['L_recon'].detach())
        if epoch % int(cfg['eval_every']) == 0 or \
                epoch == int(cfg['max_epochs']):
            model.eval()
            try:
                with torch.no_grad():
                    tout = model(inputs, G, ei, tp_t)
                    vout = model(inputs, G, ei, vp_t)
                    pout = model(inputs, G, ei, probe_t)
                    if not (torch.isfinite(tout['logits']).all()
                            and torch.isfinite(vout['logits']).all()
                            and torch.isfinite(vout['exist']).all()
                            and torch.isfinite(pout['exist']).all()):
                        raise FloatingPointError('nonfinite eval')
                    tm_ = losses.type_metrics(
                        y_tr, tout['logits'].cpu().numpy())
                    vm = losses.type_metrics(
                        np.asarray(data['y_val']),
                        vout['logits'].cpu().numpy())
            except (ValueError, FloatingPointError) as e:
                aborted = ('nonfinite eval at epoch %d: %s' % (epoch, e))
                trajectory.append(rec)
                break
            rec['train_macro_f1'] = tm_['primary']['macro_f1']
            rec['val_macro_f1'] = vm['primary']['macro_f1']
            rec['val_exist_mean'] = float(vout['exist'].mean())
            rec['probe_exist_mean'] = float(pout['exist'].mean())
            rec['val_exist_gap'] = (rec['val_exist_mean']
                                    - rec['probe_exist_mean'])
            if best is None or rec['val_macro_f1'] > \
                    best['val_macro_f1']:
                best = dict(rec)
                torch.save(u1_runner._ckpt_u1(
                    model, opt, epoch, rec['val_macro_f1'], G, ei, cfg,
                    neg_t, bank_sha, tensors, recon_block,
                    dynamic_state),
                    os.path.join(out_dir, 'ckpt_best.pt'))
        trajectory.append(rec)
        log('epoch %d loss %.5f exist_gap %.4f %s' % (
            epoch, rec['train_loss'],
            rec.get('val_exist_gap', float('nan')),
            ('val_f1 %.4f' % rec['val_macro_f1'])
            if 'val_macro_f1' in rec else ''))

    torch.save(u1_runner._ckpt_u1(
        model, opt, epoch,
        best['val_macro_f1'] if best else float('nan'),
        G, ei, cfg, neg_t, bank_sha, tensors, recon_block,
        dynamic_state), os.path.join(out_dir, 'ckpt_final.pt'))

    exports_from = None
    if aborted is None:
        exports_from = 'final'
        best_ck = os.path.join(out_dir, 'ckpt_best.pt')
        if best is not None and os.path.exists(best_ck):
            selected = torch.load(best_ck, map_location='cpu',
                                  weights_only=False)
            model.load_state_dict(selected['model_state'])
            G, ei = selected['G'], selected['ei']
            exports_from = 'best'
        model.eval()
        with torch.no_grad():
            tout = model(inputs, G, ei, tp_t)
            vout = model(inputs, G, ei, vp_t)
        meta = {'stage': 'U1X', 'exports_from': exports_from,
                'lam_exist': float(cfg['lam_exist']),
                'best_epoch': best['epoch'] if best else None,
                'val_macro_f1': best['val_macro_f1'] if best else None}
        s1r._export_predictions(out_dir, 'train', tp,
                                tout['logits'].cpu().numpy(), y_tr, meta)
        s1r._export_predictions(out_dir, 'val', vp,
                                vout['logits'].cpu().numpy(),
                                np.asarray(data['y_val']), meta)
        np.savez(os.path.join(out_dir, 'exist_scores.npz'),
                 train_pairs=tp, exist_train=tout['exist'].cpu().numpy(),
                 val_pairs=vp, exist_val=vout['exist'].cpu().numpy())

    manifest = dict(frozen)
    manifest.update({
        'aborted': aborted, 'best': best, 'final_epoch': epoch,
        'exports_from': exports_from,
        'trajectory': trajectory,
        'G_sha_start': g_sha0, 'ei_sha_start': e_sha0,
        'dynamic_history': dynamic_history,
        'training_forward_count': training_forward_count,
        'preflight': rep,
        'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    with open(os.path.join(out_dir, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest
