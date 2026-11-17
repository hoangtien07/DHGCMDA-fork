# -*- coding: utf-8 -*-
"""C4 runner — cumulative C3 + R1 reconstruction:

    L_total = L_type + 1.0*L_intra + 0.3*L_inter + lam_recon*L_recon

Scope per C4_CONTRACT.md: reuses frozen C3 terms (delegating
c1r._intra_terms / c2r._inter_terms) and the ACCEPTED
ReconstructionAdapter once, fed by out['zm_pre']/out['zd_pre'] of the
SAME forward. Targets = mi_v2[:, :n_mi] (train functional) and
dis_v2[:, :n_dis] (MeSH semantic) built by the accepted
reconstruction_targets builder from views + train_mask + meta
entity_names['dis'] + audits['descriptor_alias']; coverage = train
support (miRNA) / declared audit complement (disease), diagonals
excluded. lam_recon==0 -> adapter disabled (heads=None): no sidecar
read, no target build, no head init — bitwise C3. lam_intra==
lam_inter==lam_recon==0 -> bitwise S1.

Sidecar gates: meta['npz_sha256'] must equal the sha of the actual
--data file; entity_names['dis'] length/order and audit n_dis/
uncovered names are validated inside the accepted builder; malformed
or drifted sidecars reject BEFORE model/output. Actual target/mask
arrays persist as flat recon_targets.npz (no objects) and inside
every checkpoint together with bank/graphs; targets are fixed for the
whole run. Stage 'C4' fail-closed manifest incl. recon block. No
dynamic refresh, no presence/existence path.
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

import runner as s1r            # frozen S1 helpers, read-only
import c2_runner as c2r        # reviewed C2 helpers (bank), read-only
import c3_runner as c3r        # reviewed C3 helpers/terms, read-only
import reconstruction_targets as rt  # accepted R1 builder, read-only

VIEW_KEYS = s1r.VIEW_KEYS

RECON_CFG_KEYS = ('lam_recon',)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_path(p):
    return _sha_bytes(open(p, 'rb').read())


# ---------- sidecars / targets ----------------------------------------------
def load_sidecars(meta_path, audit_path):
    """Load the declared sidecars; returns (meta, audit)."""
    meta = json.load(open(meta_path, encoding='utf-8'))
    audit = json.load(open(audit_path, encoding='utf-8'))
    return meta, audit


def validate_sidecars(meta, audit, data_file_sha, n_dis):
    """Identity checks BEFORE the target builder: bundle sha match,
    disease names length, audit n_dis + required keys."""
    if not isinstance(meta, dict):
        raise ValueError('meta sidecar must be a JSON object')
    if data_file_sha is not None:
        want = meta.get('npz_sha256')
        if want != data_file_sha:
            raise ValueError(
                'meta.npz_sha256 %r does not match data bundle sha %r'
                % (want, data_file_sha))
    names = meta.get('entity_names', {}).get('dis')
    if not isinstance(names, list) or len(names) != n_dis \
            or any(not isinstance(s, str) for s in names):
        raise ValueError(
            'meta.entity_names.dis must be %d strings' % n_dis)
    if not isinstance(audit, dict) or 'descriptor_alias' not in audit:
        raise ValueError('audit sidecar missing descriptor_alias')
    da = audit['descriptor_alias']
    for k in ('n_dis', 'n_uncovered_rows', 'uncovered_rows'):
        if k not in da:
            raise ValueError('descriptor_alias missing key %s' % k)
    if da['n_dis'] != n_dis:
        raise ValueError('descriptor_alias n_dis %r != %d'
                         % (da['n_dis'], n_dis))
    return names, da


def build_recon_artifacts(data, names, da):
    """Accepted builder -> targets/masks/meta. train_mask forced bool.
    Never reads labels; miRNA coverage = train support, disease
    coverage = declared audit complement."""
    mask = np.asarray(data['train_mask']).astype(bool)
    return rt.build_reconstruction_targets(
        data['views'], mask, names, da)


def _recon_block_disabled():
    return {'enabled': False}


def _recon_block(artifacts, meta_path, audit_path, data_file_sha,
                 names_sha):
    m = artifacts['meta']
    return {
        'enabled': True,
        'mi_target_sha256': m['mi_target_sha256'],
        'dis_target_sha256': m['dis_target_sha256'],
        'mi_valid_mask_sha256': m['mi_valid_mask_sha256'],
        'dis_valid_mask_sha256': m['dis_valid_mask_sha256'],
        'train_mask_sha256': m['train_mask_sha256'],
        'mi_target_dtype': m['mi_target_dtype'],
        'dis_target_dtype': m['dis_target_dtype'],
        'n_mi': m['n_mi'], 'n_dis': m['n_dis'],
        'mi_valid_count': artifacts['mi_valid_count'],
        'dis_valid_count': artifacts['dis_valid_count'],
        'mi_covered_count': artifacts['mi_covered_count'],
        'dis_covered_count': artifacts['dis_covered_count'],
        'dis_uncovered_rows': m['dis_uncovered_rows'],
        'mi_target_id': m['mi_target_id'],
        'dis_target_id': m['dis_target_id'],
        'mi_coverage_policy': m['mi_coverage_policy'],
        'dis_coverage_policy': m['dis_coverage_policy'],
        'diagonal_policy': m['diagonal_policy'],
        'symmetry_atol': m['symmetry_atol'],
        'disease_names_sha256': names_sha,
        'sidecars': {
            'meta_path': meta_path,
            'meta_sha256': _sha_path(meta_path),
            'audit_path': audit_path,
            'audit_sha256': _sha_path(audit_path),
            'npz_sha256': data_file_sha},
        'builder': 'stages/reconstruction_targets.py',
        'builder_sha256': _sha_path(
            os.path.join(_HERE, 'reconstruction_targets.py')),
        'reduction': 'global valid-cell weighted mean '
                     '(masked_sum_mi+masked_sum_dis)/'
                     '(valid_count_mi+valid_count_dis) — declared '
                     'engineering unit, not literal Frobenius-sum',
    }


def _resolve_recon(cfg, data, meta_path=None, audit_path=None,
                   data_file_sha=None):
    """Returns recon block + artifacts (or disabled). lam_recon==0
    short-circuits BEFORE any sidecar/target/head work."""
    if float(cfg['lam_recon']) == 0.0:
        return _recon_block_disabled(), None
    if not meta_path or not audit_path:
        raise ValueError('lam_recon>0 requires explicit --meta and '
                         '--audit sidecar paths')
    if (not isinstance(data_file_sha, str) or len(data_file_sha) != 64
            or any(c not in '0123456789abcdef' for c in data_file_sha)):
        raise ValueError('enabled reconstruction requires the actual data '
                         'bundle SHA256 (64 lowercase hex characters)')
    mask = np.asarray(data['train_mask'])
    n_dis = mask.shape[1]
    meta, audit = load_sidecars(meta_path, audit_path)
    names, da = validate_sidecars(meta, audit, data_file_sha, n_dis)
    names_sha = _sha_bytes(
        json.dumps(names, ensure_ascii=True).encode('ascii'))
    artifacts = build_recon_artifacts(data, names, da)
    block = _recon_block(artifacts, meta_path, audit_path,
                         data_file_sha, names_sha)
    return block, artifacts


# ---------- gates -------------------------------------------------------------
def preflight_c4(cfg, data):
    rep = c3r.preflight_c3(cfg, data)
    for k in RECON_CFG_KEYS:
        if k not in cfg:
            raise ValueError('preflight failed: missing cfg.' + k)
    if not (float(cfg['lam_recon']) >= 0.0
            and np.isfinite(float(cfg['lam_recon']))):
        raise ValueError('preflight failed: lam_recon finite >=0')
    rep['stage'] = 'C4'
    return rep


def env_check(cfg):
    return s1r.env_check(cfg)


def source_hashes_c4():
    h = {
        'stages/c4_runner.py': _sha_path(os.path.abspath(__file__)),
        'stages/c4_model.py': _sha_path(
            os.path.join(_HERE, 'c4_model.py')),
        'stages/c3_runner.py': _sha_path(
            os.path.join(_HERE, 'c3_runner.py')),
        'stages/c2_model.py': _sha_path(
            os.path.join(_HERE, 'c2_model.py')),
        'stages/intra.py': _sha_path(os.path.join(_HERE, 'intra.py')),
        'stages/inter.py': _sha_path(os.path.join(_HERE, 'inter.py')),
        'stages/c1_runner.py': _sha_path(
            os.path.join(_HERE, 'c1_runner.py')),
        'stages/c2_runner.py': _sha_path(
            os.path.join(_HERE, 'c2_runner.py')),
        'stages/reconstruction.py': _sha_path(
            os.path.join(_HERE, 'reconstruction.py')),
        'stages/reconstruction_adapter.py': _sha_path(
            os.path.join(_HERE, 'reconstruction_adapter.py')),
        'stages/reconstruction_decoder.py': _sha_path(
            os.path.join(_HERE, 'reconstruction_decoder.py')),
        'stages/reconstruction_targets.py': _sha_path(
            os.path.join(_HERE, 'reconstruction_targets.py')),
    }
    return h


def imported_pure_hashes_c4():
    return c3r.imported_pure_hashes_c3()


def frozen_manifest_c4(cfg, preflight_rep, env_versions, bank_sha,
                       bank_counts, recon_block):
    m = {
        'stage': 'C4',
        'cfg': dict(cfg),
        'env': dict(env_versions,
                    python=platform.python_version(),
                    platform=platform.platform()),
        'data_hashes': preflight_rep['data_hashes'],
        'view_hashes': preflight_rep['view_hashes'],
        'n_pairs': preflight_rep['n_pairs'],
        'source_hashes': source_hashes_c4(),
        'imported_pure_hashes': imported_pure_hashes_c4(),
        'neg_bank': {'sha256': bank_sha,
                     'n_neg': int(sum(bank_counts)),
                     'per_anchor_nonzero': int(sum(
                         1 for c in bank_counts if c > 0)),
                     'cap_mult': float(cfg['inter_cap_mult']),
                     'seed': int(cfg['inter_seed']),
                     'provenance': 'A_train-zero cells, ASSUMED '
                                   'negatives (not verified); local '
                                   'seeded generator, no held-out '
                                   'label access; allocated even at '
                                   'lam_inter==0 (numerically inert)'},
        'recon': recon_block,
    }
    return m


EXPECTED_MANIFEST_KEYS = c3r.EXPECTED_MANIFEST_KEYS + ('recon',)
EXPECTED_MANIFEST_META_KEYS = c3r.EXPECTED_MANIFEST_META_KEYS


def check_expected_manifest(frozen, expected):
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
    if expected.get('stage') != 'C4':
        diffs.append('stage must be C4, got %r'
                     % (expected.get('stage'),))
    if diffs:
        raise ValueError('expected-manifest mismatch:\n'
                         + '\n'.join(diffs))


def expected_manifest_for_c4(cfg, data, meta_path=None,
                             audit_path=None, data_file_sha=None):
    rep = preflight_c4(cfg, data)
    envv = env_check(cfg)
    _, sha, counts = c2r.build_negative_bank(cfg, data)
    block, _ = _resolve_recon(cfg, data, meta_path, audit_path,
                              data_file_sha)
    m = frozen_manifest_c4(cfg, rep, envv, sha, counts, block)
    m['manifest_kind'] = 'candidate_expected'
    return m


# ---------- recon term ----------------------------------------------------------
def _recon_terms(model, out, tensors, cfg):
    """Adapter call ONCE on the same forward's fused pre-HGT outputs.
    Returns None at lam_recon==0 (heads absent; branch skipped)."""
    lam = float(cfg['lam_recon'])
    if lam == 0.0:
        return None
    r = model.recon(out['zm_pre'], out['zd_pre'],
                    tensors['mi_target'], tensors['mi_valid'],
                    tensors['dis_target'], tensors['dis_valid'])
    return {'lam': lam, 'L_recon': r['l_recon'],
            'recon_sum_mi': float(r['masked_sum_mi'].detach()),
            'recon_sum_dis': float(r['masked_sum_dis'].detach()),
            'recon_mean_mi': float(r['masked_mean_mi'].detach()),
            'recon_mean_dis': float(r['masked_mean_dis'].detach()),
            'recon_valid_mi': r['valid_count_mi'],
            'recon_valid_dis': r['valid_count_dis'],
            'gram_mi': r['gram_mi'], 'gram_dis': r['gram_dis']}


# ---------- checkpoints ----------------------------------------------------------
def _ckpt_c4(model, opt, epoch, val_metric, G, ei, cfg, bank,
             bank_sha, tensors, recon_block):
    ck = s1r._ckpt(model, opt, epoch, val_metric, G, ei, cfg)
    ck['neg_bank'] = bank
    ck['neg_bank_sha'] = bank_sha
    if tensors is not None:
        ck['recon_targets'] = {k: tensors[k] for k in
                               ('mi_target', 'mi_valid',
                                'dis_target', 'dis_valid')}
        ck['recon_meta'] = {
            'mi_target_sha256': recon_block['mi_target_sha256'],
            'dis_target_sha256': recon_block['dis_target_sha256'],
            'mi_valid_count': recon_block['mi_valid_count'],
            'dis_valid_count': recon_block['dis_valid_count']}
    return ck


def _persist_targets(out_dir, artifacts):
    np.savez(os.path.join(out_dir, 'recon_targets.npz'),
             mi_target=artifacts['mi_target'],
             dis_target=artifacts['dis_target'],
             mi_valid_mask=artifacts['mi_valid_mask'],
             dis_valid_mask=artifacts['dis_valid_mask'],
             mi_row_coverage=artifacts['mi_row_coverage'],
             dis_row_coverage=artifacts['dis_row_coverage'])


# ---------- training -------------------------------------------------------------
def train_c4(cfg, data, out_dir, log=print, expected_manifest=None,
             meta_path=None, audit_path=None, data_file_sha=None):
    """Train C4 on caller-supplied fold. Same gates as C3 PLUS
    sidecar/target validation before model/output; nonfinite abort
    keeps last-good checkpoint and exports nothing."""
    import torch
    import losses
    import c4_model as c4m

    rep = preflight_c4(cfg, data)
    envv = env_check(cfg)
    bank, bank_sha, bank_counts = c2r.build_negative_bank(cfg, data)
    recon_block, artifacts = _resolve_recon(
        cfg, data, meta_path, audit_path, data_file_sha)
    frozen = frozen_manifest_c4(cfg, rep, envv, bank_sha, bank_counts,
                                recon_block)
    check_expected_manifest(frozen, expected_manifest)
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, 'manifest_frozen.json'), 'w',
              encoding='utf-8') as f:
        json.dump(frozen, f, indent=1, default=str)
    if artifacts is not None:
        _persist_targets(out_dir, artifacts)

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

    model = c4m.PaperGuidedC4(
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
        temperature_max=float(cfg['temperature_max']),
        recon_enabled=(float(cfg['lam_recon']) != 0.0),
        recon_seed=int(cfg['seed']))
    opt = s1r.build_optimizer(model, cfg)   # AFTER adapter registration
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
    aborted = None
    epoch = 0
    for epoch in range(1, int(cfg['max_epochs']) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        prev = s1r._flat_params(model).clone()
        try:
            out = model(inputs, G, ei, tp_t)
            l_type = losses.type_loss(out['logits'], y_tr, w_t)
            it = c3r._intra_terms(model, out, cfg)
            et = c3r._inter_terms(out, pos_t, neg_t, cfg)
            rt_ = _recon_terms(model, out, tensors, cfg)
            loss = l_type
            if it is not None:
                loss = loss + it['lam'] * it['L_intra']
            if et is not None:
                loss = loss + et['lam'] * et['L_inter']
            if rt_ is not None:
                loss = loss + rt_['lam'] * rt_['L_recon']
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
        upd = float((s1r._flat_params(model) - prev).norm())
        rec = {'epoch': epoch,
               'train_loss': float(loss.detach()),
               'L_type': float(l_type.detach()),
               'grad_norm': gnorm, 'update_norm': upd}
        if it is not None:
            rec['lam_intra_applied'] = it['lam']
            rec['L_intra'] = float(it['L_intra'].detach())
            rec['intra_mi_sum'] = it['intra_mi_sum']
            rec['intra_dis_sum'] = it['intra_dis_sum']
            rec['intra_mi_mean'] = it['intra_mi_mean']
            rec['intra_dis_mean'] = it['intra_dis_mean']
            rec['intra_n_mi'] = it['n_mi']
            rec['intra_n_dis'] = it['n_dis']
        if et is not None:
            rec['lam_inter_applied'] = et['lam']
            rec['L_inter'] = float(et['L_inter'].detach())
            rec['inter_info_sum'] = et['inter_info_sum']
            rec['inter_info_mean'] = et['inter_info_mean']
            rec['inter_pos_mean_sim'] = et['inter_pos_mean_sim']
            rec['inter_neg_mean_sim'] = et['inter_neg_mean_sim']
            rec['inter_margin_term'] = et['inter_margin_term']
            rec['inter_n_pos'] = et['inter_n_pos']
            rec['inter_n_neg'] = et['inter_n_neg']
        if rt_ is not None:
            rec['lam_recon_applied'] = rt_['lam']
            rec['L_recon'] = float(rt_['L_recon'].detach())
            rec['recon_sum_mi'] = rt_['recon_sum_mi']
            rec['recon_sum_dis'] = rt_['recon_sum_dis']
            rec['recon_mean_mi'] = rt_['recon_mean_mi']
            rec['recon_mean_dis'] = rt_['recon_mean_dis']
            rec['recon_valid_mi'] = rt_['recon_valid_mi']
            rec['recon_valid_dis'] = rt_['recon_valid_dis']
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
                torch.save(_ckpt_c4(model, opt, epoch,
                                    rec['val_macro_f1'], G, ei, cfg,
                                    neg_t, bank_sha, tensors,
                                    recon_block),
                           os.path.join(out_dir, 'ckpt_best.pt'))
        trajectory.append(rec)
        log('epoch %d loss %.5f %s' % (
            epoch, rec['train_loss'],
            ('val_f1 %.4f' % rec['val_macro_f1'])
            if 'val_macro_f1' in rec else ''))

    torch.save(_ckpt_c4(model, opt, epoch,
                        best['val_macro_f1'] if best else float('nan'),
                        G, ei, cfg, neg_t, bank_sha, tensors,
                        recon_block),
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
        meta = {'stage': 'C4', 'exports_from': exports_from,
                'lam_intra': float(cfg['lam_intra']),
                'intra_temperature': float(cfg['intra_temperature']),
                'intra_alpha': float(cfg['intra_alpha']),
                'lam_inter': float(cfg['lam_inter']),
                'inter_temperature': float(cfg['inter_temperature']),
                'inter_margin': float(cfg['inter_margin']),
                'inter_margin_coef': float(cfg['inter_margin_coef']),
                'lam_recon': float(cfg['lam_recon']),
                'reduction': 'declared normalized engineering units, '
                'not paper-literal lambda equivalence',
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
        'intra': {'lam_intra': float(cfg['lam_intra']),
                  'temperature': float(cfg['intra_temperature']),
                  'alpha': float(cfg['intra_alpha']),
                  'reduction': '(mi.sum+dis.sum)/(n_mi+n_dis) - '
                               'reviewed C1 declared engineering unit'},
        'inter': {'lam_inter': float(cfg['lam_inter']),
                  'temperature': float(cfg['inter_temperature']),
                  'margin': float(cfg['inter_margin']),
                  'margin_coef': float(cfg['inter_margin_coef']),
                  'bank_sha256': bank_sha,
                  'reduction': 'info_mean + coef*margin - reviewed C2 '
                               'declared engineering unit'},
        'embeddings': 'intra: proj heads on out.views; inter: raw '
                      'fused PRE-HGT zm_pre/zd_pre; recon: adapter '
                      'heads on the SAME zm_pre/zd_pre; one forward, '
                      'each term added ONCE',
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


# ---------- CLI --------------------------------------------------------------------
def main(argv=None):
    """C4 freeze-then-train (same two-step as C3)."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--data', help='flat npz data bundle')
    p.add_argument('--meta', help='bundle meta.json sidecar '
                                  '(entity_names.dis, npz_sha256)')
    p.add_argument('--audit', help='audits.json sidecar '
                                   '(descriptor_alias)')
    p.add_argument('--out', required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--expected-manifest',
                   help='REQUIRED for training')
    p.add_argument('--emit-manifest',
                   help='preflight-only: write candidate manifest')
    a = p.parse_args(argv)
    try:
        cfg = json.load(open(a.config, encoding='utf-8'))
        data = s1r.load_data_npz(a.data) if a.data else {}
        data_file_sha = _sha_path(a.data) if a.data else None
        rep = preflight_c4(cfg, data)
        envv = env_check(cfg)
        _, bsha, bcnt = c2r.build_negative_bank(cfg, data)
        block, artifacts = _resolve_recon(
            cfg, data, a.meta, a.audit, data_file_sha)
        frozen = frozen_manifest_c4(cfg, rep, envv, bsha, bcnt, block)
        if a.preflight_only:
            frozen['manifest_kind'] = 'candidate_expected'
            if a.emit_manifest:
                with open(a.emit_manifest, 'w',
                          encoding='utf-8') as f:
                    json.dump(frozen, f, indent=1, default=str)
                if artifacts is not None:
                    np.savez(
                        os.path.splitext(a.emit_manifest)[0]
                        + '_targets.npz',
                        mi_target=artifacts['mi_target'],
                        dis_target=artifacts['dis_target'],
                        mi_valid_mask=artifacts['mi_valid_mask'],
                        dis_valid_mask=artifacts['dis_valid_mask'],
                        mi_row_coverage=artifacts['mi_row_coverage'],
                        dis_row_coverage=artifacts['dis_row_coverage'])
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
        m = train_c4(cfg, data, a.out,
                     expected_manifest=json.load(
                         open(a.expected_manifest, encoding='utf-8')),
                     meta_path=a.meta, audit_path=a.audit,
                     data_file_sha=data_file_sha)
        print('DONE best', m['best'], 'aborted', m['aborted'])
        return 2 if m['aborted'] else 0
    except (ValueError, KeyError, FileExistsError) as e:
        print('FAIL: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
