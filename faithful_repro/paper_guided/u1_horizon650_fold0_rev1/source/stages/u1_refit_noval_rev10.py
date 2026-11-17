# -*- coding: utf-8 -*-
"""U1_REFIT_NOVAL REV4 — hardened fixed-horizon no-validation refit.

Sibling revision of u1_refit_noval.py (preserved). Differences:
 - full sealed trajectory provenance restored: grad_norm, update_norm,
   unit-mass factors, per-term sums/means, dynamic refresh meta with
   producer/application phases + producer-Gram and old/new graph
   hashes (application phase honestly relabelled: no eval exists);
 - last-good checkpoint built from DEEP-COPIED tensors so a failed
   later epoch can never leak a stale-mutated state;
 - attempted vs completed epochs reported separately — a failed epoch
   is never labelled final;
 - post-optimizer parameter-finiteness gate;
 - strengthened preflight: sealed _pairs_checked bounds/integer/
   uniqueness gates, binary mask, U1-equivalent cfg validation
   (dynamic_period==5, dynamic_threshold==0.5, unit_mode, lam_recon
   consistency) minus all validation requirements;
 - optional externally pinned expected_source_hashes verified BEFORE
   any output is created.

No validation or test labels are consumed anywhere; cfg['refit_noval']
must be True; horizon is the externally declared cfg['max_epochs'].
"""
import argparse, copy, hashlib, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runner as s1r
import c2_runner as c2r
import c3_runner as c3r
import dynamic_graph as dg
import u1_runner as u1r              # read-only helper reuse
import reconstruction_targets as rt
from objective_units import combine_objectives

VIEW_KEYS = s1r.VIEW_KEYS
PROHIBITED_KEYS = {'val_pairs', 'y_val', 'test_pairs', 'y_test',
                   'test', 'outer_pairs', 'y_outer'}


def _fail(m):
    raise ValueError('gate failure: ' + m)


# ---------- loader: refuses any validation/test content ----------
def load_refit_npz(path, expect_sha=None):
    if expect_sha is not None:
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if got != expect_sha:
            _fail('refit npz sha mismatch')
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        bad = PROHIBITED_KEYS & keys
        if bad:
            _fail('prohibited held-out keys present: %s' % sorted(bad))
        if any(k.startswith('val_') or k.startswith('test_') or
               'outer' in k for k in keys):
            _fail('held-out-like keys present: %s' % sorted(keys))
        need = {'view_' + k for k in VIEW_KEYS} | {
            'train_mask', 'train_pairs', 'y_train'}
        if not need <= keys:
            _fail('missing keys %s' % sorted(need - keys))
        data = {'views': {}}
        for k in z.files:
            a = z[k]
            if a.dtype == object:
                _fail('object array refused: ' + k)
            if k.startswith('view_'):
                data['views'][k[len('view_'):]] = a
            else:
                data[k] = a
    return data


def effective_source_map():
    """Digests of the ACTUAL files of every module this runner
    executes: all sealed stage modules, the parent-level
    runner/losses/model/contracts/metrics (via
    imported_pure_hashes_u1), and this file."""
    m = dict(u1r.imported_pure_hashes_u1())
    m.update(u1r.source_hashes_u1())
    m['stages/u1_refit_noval_rev10.py'] = u1r._sha_path(
        os.path.abspath(__file__))
    for extra in ('selection_wrapper.py', 'semantic.py',
                  'functional.py',
                  'build_selection_binding_rev3.py'):
        cands = [os.path.join(os.path.dirname(
            os.path.abspath(__file__)), extra),
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), extra)]
        for c in cands:
            if os.path.exists(c):
                m['orchestrators/' + extra] = u1r._sha_path(c)
                break
    return m


def verify_source_pins(expected_source_hashes):
    """Externally pinned effective-code digests. REQUIRED — a refit
    run without pins is unbound. Verified BEFORE any output; the
    returned map must also be re-verified after work by the caller."""
    if not isinstance(expected_source_hashes, dict) \
            or not expected_source_hashes:
        _fail('expected_source_hashes is REQUIRED (nonempty dict)')
    actual = effective_source_map()
    diffs = {}
    for k, want in expected_source_hashes.items():
        if actual.get(k) != want:
            diffs[k] = (want, actual.get(k))
    missing = set(actual) - set(expected_source_hashes)
    if diffs or missing:
        _fail('source pin mismatch: %s unpinned=%s'
              % (diffs, sorted(missing)))
    return actual


# ---------- selection lineage (R1) --------------------------------------
_HEX = set('0123456789abcdef')


def _is_sha(s):
    return isinstance(s, str) and len(s) == 64 \
        and all(c in _HEX for c in s)


def _pinned_json(path, expect_sha, what):
    """Read a json file ONLY after verifying its exact byte SHA pin."""
    if not _is_sha(expect_sha):
        _fail('%s sha pin must be 64 lowercase hex' % what)
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if got != expect_sha:
        _fail('%s sha mismatch: %s != %s' % (what, got, expect_sha))
    try:
        obj = json.load(open(path, encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        _fail('%s is not valid json: %s' % (what, e))
    if not isinstance(obj, dict):
        _fail('%s must be a JSON object' % what)
    return obj


def verify_and_parse_selection_manifest(path, expect_sha):
    """Real U1 selection-run manifest.json, pin-verified before parse.
    Derives selected_epoch from the verified manifest's best.epoch —
    never a free caller integer. Requires a successful, non-aborted
    run; fails closed if a required identity cannot be proven."""
    m = _pinned_json(path, expect_sha, 'selection manifest')
    if m.get('stage') != 'U1':
        _fail('selection manifest stage must be U1, got %r'
              % m.get('stage'))
    if m.get('aborted') is not None:
        _fail('selection run aborted=%r — not a valid selection run'
              % m.get('aborted'))
    best = m.get('best')
    if not isinstance(best, dict) or type(best.get('epoch')) is not int \
            or best['epoch'] <= 0:
        _fail('selection manifest missing positive-int best.epoch')
    fe = m.get('final_epoch')
    if type(fe) is not int or isinstance(fe, bool) or fe <= 0:
        _fail('selection manifest missing positive-int final_epoch')
    cme = m.get('cfg', {}).get('max_epochs')
    if type(cme) is not int or isinstance(cme, bool):
        _fail('selection cfg missing int max_epochs')
    if not (best['epoch'] <= fe <= cme):
        _fail('impossible horizon: best %d > final %d > max %d'
              % (best['epoch'], fe, cme))
    if not isinstance(m.get('cfg'), dict):
        _fail('selection manifest missing cfg dict')
    if not isinstance(m.get('data_hashes'), dict):
        _fail('selection manifest missing data_hashes dict')
    return {'epoch': best['epoch'], 'cfg': m['cfg'],
            'data_hashes': m['data_hashes'], 'sha': expect_sha,
            'source_hashes': m.get('source_hashes') or {},
            'imported_pure_hashes':
                m.get('imported_pure_hashes') or {}}


def verify_and_parse_selection_binding(path, expect_sha, sel_sha):
    """Separately pinned binding sidecar — the declared contract that
    carries what the selection manifest cannot prove (fold, candidate,
    phase). Fields must be EXACT and self-consistent; never inferred."""
    b = _pinned_json(path, expect_sha, 'selection binding sidecar')
    want = {'selection_manifest_sha256', 'fold',
            'candidate_csv_sha256', 'phase'}
    missing = want - set(b)
    if missing:
        _fail('selection binding missing keys %s' % sorted(missing))
    # provenance must be EITHER the recomputed digests+counts schema
    # (build_selection_binding.py — preferred, machine-verifiable) OR
    # a nonempty attestation string (legacy REV4 contract). Digests are
    # validated structurally whenever present.
    # S2: producer-emitted digest schema is REQUIRED — no
    # attestation-string fallback on this path.
    if 'digests' in b or 'counts' in b or True:
        if b.get('stage') != 'SELECTION_BINDING_REV1':
            _fail('digest binding must be SELECTION_BINDING_REV1')
        cn = b.get('counts')
        if not isinstance(cn, dict) or any(
                type(cn.get(k)) is not int or cn[k] <= 0
                for k in ('inner_train', 'inner_val', 'test',
                          'refit_union')):
            _fail('binding counts must be positive ints '
                  '{inner_train,inner_val,test,refit_union}')
        if cn['refit_union'] != cn['inner_train'] + cn['inner_val']:
            _fail('binding counts: refit_union != train+val')
        dg = b.get('digests')
        if not isinstance(dg, dict) or not all(
                _is_sha(v) for v in dg.values()):
            _fail('binding digests must map names->64-hex sha')
        for k in ('sel_bundle_sha256', 'sel_meta_sha256',
                  'refit_bundle_sha256', 'refit_meta_sha256',
                  'candidate_manifest_sha256'):
            if not _is_sha(b.get(k)):
                _fail('digest binding missing pin ' + k)
        if type(b.get('selected_epoch')) is not int \
                or b['selected_epoch'] <= 0:
            _fail('digest binding missing positive selected_epoch')
    else:
        if not isinstance(b.get('non_test_pair_proof'), str) \
                or not b['non_test_pair_proof'].strip():
            _fail('binding lacks recomputed digests AND attestation '
                  'proof — cannot accept')
    if not _is_sha(b['selection_manifest_sha256']) \
            or b['selection_manifest_sha256'] != sel_sha:
        _fail('binding selection_manifest_sha256 does not match the '
              'verified selection manifest sha')
    if type(b['fold']) is not int or not (0 <= b['fold'] < 5):
        _fail('binding fold must be int in [0,5)')
    if not _is_sha(b['candidate_csv_sha256']):
        _fail('binding candidate_csv_sha256 must be 64 lowercase hex')
    if b['phase'] != 'selection':
        _fail('binding phase must be selection, got %r' % b['phase'])
    return b


def verify_and_parse_bundle_meta(path, expect_sha, binding):
    """Refit bundle's candidate_fold_data.meta.json, pin-verified then
    cross-checked against the verified binding: same fold, same
    candidate csv sha, refit phase vs selection phase."""
    m = _pinned_json(path, expect_sha, 'bundle meta')
    if m.get('stage') != 'CANDIDATE_FOLD_BUNDLE_REV3':
        _fail('bundle meta stage must be CANDIDATE_FOLD_BUNDLE_REV3')
    if m.get('phase') != 'refit':
        _fail('bundle meta phase must be refit, got %r'
              % m.get('phase'))
    if m.get('fold') != binding['fold']:
        _fail('bundle fold %r != bound selection fold %r'
              % (m.get('fold'), binding['fold']))
    pins = m.get('pins')
    if not isinstance(pins, dict):
        _fail('bundle meta missing pins dict')
    if pins.get('pair_split_candidate.csv') \
            != binding['candidate_csv_sha256']:
        _fail('bundle candidate sha %r != bound candidate sha %r'
              % (pins.get('pair_split_candidate.csv'),
                 binding['candidate_csv_sha256']))
    return m


CFG_DIFF_ALLOWED = {'stage', 'refit_noval', 'max_epochs'}


def check_cfg_compat(sel_cfg, refit_cfg):
    """Selection vs refit cfg compatibility: every key equal except the
    declared allowed set (stage/refit_noval are refit-only additions;
    max_epochs must equal the manifest-derived selected epoch, checked
    separately). No silent inference on other keys."""
    for k in set(sel_cfg) | set(refit_cfg):
        if k in CFG_DIFF_ALLOWED:
            continue
        if sel_cfg.get(k) != refit_cfg.get(k):
            _fail('cfg.%s incompatible: selection %r vs refit %r'
                  % (k, sel_cfg.get(k), refit_cfg.get(k)))
    if int(refit_cfg.get('max_epochs', -1)) <= 0:
        _fail('refit cfg.max_epochs must be positive int')
    return True


# ---------- data preflight: U1-equivalent minus validation ----------
def preflight_refit(cfg, data):
    rep = {'checks': []}
    def ok(n, c, d=''):
        rep['checks'].append((n, bool(c), d))
        if not c:
            raise ValueError('preflight_refit failed: %s %s' % (n, d))
    ok('cfg.refit_noval', cfg.get('refit_noval') is True)
    ok('cfg.stage', cfg.get('stage') == 'U1_REFIT_NOVAL',
       str(cfg.get('stage')))
    req = ('hidden', 'readout', 'proj', 'hgt_layers', 'hgt_heads',
           'n_types', 'hgt_dropout', 'readout_dropout', 'knn_k', 'lr',
           'weight_decay', 'max_epochs', 'eval_every', 'beta', 'seed',
           'leaky_slope', 'temperature_init', 'temperature_min',
           'temperature_max',
           'lam_intra', 'intra_temperature', 'intra_alpha',
           'lam_inter', 'inter_temperature', 'inter_margin',
           'inter_margin_coef', 'inter_cap_mult', 'inter_seed',
           'lam_recon', 'dynamic_enabled', 'dynamic_period',
           'dynamic_threshold', 'unit_mode')
    for k_ in req:
        ok('cfg.' + k_, k_ in cfg)
    ok('cfg.n_types==5', cfg['n_types'] == 5, str(cfg.get('n_types')))
    ok('cfg.beta in [0,1)', 0.0 <= cfg['beta'] < 1.0)
    ok('horizon declared', type(cfg['max_epochs']) is int
       and cfg['max_epochs'] > 0)
    ok('cfg.temp bounds', 0 < cfg['temperature_min']
       <= cfg['temperature_init'] <= cfg['temperature_max'])
    # U1-equivalent objective/dynamic validation (sealed preflight_u1
    # semantics, val requirements omitted)
    ok('lam_intra >=0', float(cfg['lam_intra']) >= 0.0
       and np.isfinite(float(cfg['lam_intra'])))
    ok('intra_temperature >0', 0.0 < float(cfg['intra_temperature'])
       and np.isfinite(float(cfg['intra_temperature'])))
    ok('intra_alpha in [0,1]',
       0.0 <= float(cfg['intra_alpha']) <= 1.0)
    ok('lam_inter >=0', float(cfg['lam_inter']) >= 0.0
       and np.isfinite(float(cfg['lam_inter'])))
    ok('inter_temperature >0', 0.0 < float(cfg['inter_temperature'])
       and np.isfinite(float(cfg['inter_temperature'])))
    ok('inter_margin >=0', 0.0 <= float(cfg['inter_margin'])
       and np.isfinite(float(cfg['inter_margin'])))
    ok('inter_margin_coef finite >=0',
       0.0 <= float(cfg['inter_margin_coef'])
       and np.isfinite(float(cfg['inter_margin_coef'])))
    ok('inter_cap_mult finite >=0', 0.0 <= float(cfg['inter_cap_mult'])
       and np.isfinite(float(cfg['inter_cap_mult'])))
    ok('lam_recon finite >=0', float(cfg['lam_recon']) >= 0.0
       and np.isfinite(float(cfg['lam_recon'])))
    ok('dynamic_enabled bool', isinstance(
        cfg.get('dynamic_enabled'), bool))
    ok('dynamic_period==5', type(cfg.get('dynamic_period')) is int
       and cfg['dynamic_period'] == 5)
    ok('dynamic_threshold==.5', cfg.get('dynamic_threshold') == .5)
    if cfg['dynamic_enabled']:
        ok('dynamic requires recon', float(cfg['lam_recon']) != 0.0)
    ok('unit_mode valid', cfg.get('unit_mode') in
       ('common_mass_sums', 'c5_means'))
    # prohibited content / known-key discipline mirrors sealed set
    bad = PROHIBITED_KEYS & set(data)
    ok('no held-out keys', not bad, str(sorted(bad)))
    allowed = set(s1r.ALLOWED_DATA_KEYS) - {'val_pairs', 'y_val'}
    extra = set(data) - allowed
    ok('known data keys', not extra, str(sorted(extra)))
    for k_ in ('views', 'train_mask', 'train_pairs', 'y_train'):
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
        ok('view %s rows' % name, a.shape[0] == want)
    tm = np.asarray(data['train_mask'])
    ok('mask shape', tm.shape == (n_mi, n_dis))
    ok('mask binary', np.isin(tm, (0, 1, False, True)).all())
    tp = s1r._pairs_checked(data['train_pairs'], 'train_pairs',
                            n_mi, n_dis, ok)
    ok('train pairs nonempty', tp.shape[0] > 0)
    exp = np.zeros((n_mi, n_dis), dtype=bool)
    exp[tp[:, 0], tp[:, 1]] = True
    ok('mask==pairs', (tm.astype(bool) == exp).all())
    yt = np.asarray(data['y_train'])
    ok('y_train shape', yt.shape == (tp.shape[0],
                                     int(cfg['n_types'])))
    ok('y_train binary+positive', np.isin(yt, (0, 1)).all()
       and (yt.sum(1) > 0).all())
    rep['data_hashes'] = {
        'train_mask': s1r._arr_sha(data['train_mask']),
        'train_pairs': s1r._arr_sha(tp),
        'y_train': s1r._arr_sha(yt)}
    rep['view_hashes'] = {n: s1r._arr_sha(v) for n, v in
                          views.items()}
    rep['n_pairs'] = {'train': int(tp.shape[0]), 'val': 0}
    rep['preflight'] = 'PASS'
    rep['stage'] = 'U1_REFIT_NOVAL'
    return rep


# ---------- refit expected manifest (B3) --------------------------------
REFIT_EXPECTED_KEYS = tuple(u1r.EXPECTED_MANIFEST_KEYS) + (
    'validation', 'refit_binding')


def expected_manifest_for_refit(cfg, data, binding,
                                meta_path=None, audit_path=None,
                                data_file_sha=None):
    """Emit a candidate expected manifest for a DECLARED refit run.
    `binding` pins the refit-specific contract: selection manifest sha,
    selected epoch (== cfg.max_epochs), consumed bundle sha and recon
    sidecar shas. Callers freeze this BEFORE any training and require
    it via expected_manifest=."""
    rep = preflight_refit(cfg, data)
    envv = u1r.env_check(cfg)
    _, sha, counts = c2r.build_negative_bank(cfg, data)
    block, _ = u1r._resolve_recon(cfg, data, meta_path, audit_path,
                                  data_file_sha)
    m = u1r.frozen_manifest_u1(cfg, rep, envv, sha, counts, block)
    m['stage'] = 'U1_REFIT_NOVAL'
    m['validation'] = 'none — fixed horizon, selection done'
    m['refit_binding'] = dict(binding)
    m['manifest_kind'] = 'candidate_expected_refit'
    return m


def check_expected_manifest_refit(frozen, expected):
    """Full-key expected-manifest check for refit — same discipline as
    the sealed checker but stage must be 'U1_REFIT_NOVAL' and the
    refit_binding block is compared in full (selection-manifest sha,
    selected epoch == cfg.max_epochs, bundle + sidecar shas)."""
    if not isinstance(expected, dict):
        raise ValueError('expected manifest must be a JSON object')
    missing = [k for k in REFIT_EXPECTED_KEYS if k not in expected]
    extra = [k for k in expected if k not in REFIT_EXPECTED_KEYS
             and k not in u1r.EXPECTED_MANIFEST_META_KEYS]
    if missing:
        raise ValueError('expected manifest missing keys: %s' % missing)
    if extra:
        raise ValueError('expected manifest unknown keys: %s' % extra)
    if expected.get('stage') != 'U1_REFIT_NOVAL':
        raise ValueError("stage must be U1_REFIT_NOVAL, got %r"
                         % (expected.get('stage'),))
    diffs = []
    def cmp(path, a, b):
        if a != b:
            diffs.append('%s: expected %r got %r' % (path, b, a))
    for key in REFIT_EXPECTED_KEYS:
        if key == 'env':
            ea, eb = expected['env'], frozen['env']
            if not isinstance(ea, dict) or not isinstance(eb, dict):
                diffs.append('env: expected a map')
                continue
            for k in set(ea) | set(eb):
                if k == 'platform':
                    continue
                cmp('env.' + k, eb.get(k), ea.get(k))
        elif key == 'refit_binding':
            ea, eb = expected['refit_binding'], \
                frozen.get('refit_binding', {})
            if not isinstance(ea, dict) or not isinstance(eb, dict):
                diffs.append('refit_binding: expected a map')
                continue
            for k in set(ea) | set(eb):
                cmp('refit_binding.' + k, eb.get(k), ea.get(k))
        else:
            cmp(key, frozen.get(key), expected.get(key))
    if diffs:
        raise ValueError('expected-manifest mismatch:\n'
                         + '\n'.join(diffs))


def _clone_ckpt(ck):
    """Deep copy so a later failed epoch can never mutate last-good."""
    import torch
    out = dict(ck)
    for k in ('model_state', 'opt_state', 'G', 'ei'):
        out[k] = {kk: (vv.clone() if hasattr(vv, 'clone')
                       else copy.deepcopy(vv))
                  for kk, vv in ck[k].items()}
    out['rng_torch'] = ck['rng_torch'].clone()
    out['rng_numpy'] = copy.deepcopy(ck['rng_numpy'])
    out['neg_bank'] = ck['neg_bank'].clone()
    if 'recon_targets' in ck:
        out['recon_targets'] = {k: v.clone() for k, v in
                                ck['recon_targets'].items()}
    return out


def _ckpt_refit(model, opt, epoch, G, ei, cfg, bank, bank_sha,
                tensors, recon_block, dynamic_state):
    """Mirror of s1r._ckpt + u1r._ckpt_u1 WITHOUT 'val_macro_f1' —
    omission cannot be misparsed as a fabricated metric."""
    import torch
    ck = {'epoch': int(epoch),
          'validation': 'none — fixed-horizon refit',
          'model_state': model.state_dict(),
          'opt_state': opt.state_dict(),
          'rng_torch': torch.get_rng_state(),
          'rng_numpy': np.random.get_state(),
          'G': dict(G), 'ei': dict(ei),
          'G_sha': {k: u1r._sha_bytes(v.cpu().numpy().tobytes())
                    for k, v in G.items()},
          'ei_sha': {str(k): u1r._sha_bytes(v.cpu().numpy().tobytes())
                     for k, v in ei.items()},
          'cfg': dict(cfg),
          'neg_bank': bank, 'neg_bank_sha': bank_sha,
          'dynamic_state': dynamic_state,
          'stage': 'U1_REFIT_NOVAL'}
    if tensors is not None:
        ck['recon_targets'] = {k: tensors[k] for k in
                               ('mi_target', 'mi_valid',
                                'dis_target', 'dis_valid')}
        ck['recon_meta'] = dict(recon_block)
        ck['recon_tensor_sha'] = {
            k: u1r._sha_bytes(v.numpy().tobytes())
            for k, v in ck['recon_targets'].items()}
    return ck


# ---------- fixed-horizon no-validation training ---------------------
def train_refit_noval(cfg, data, out_dir, log=print,
                      expected_manifest=None, meta_path=None,
                      audit_path=None, data_file_sha=None,
                      expected_source_hashes=None,
                      binding=None):
    """Same objective/update arithmetic and trajectory provenance as
    u1_runner.train_u1 with the validation path REMOVED: no evals, no
    best-epoch selection, fixed externally-declared horizon, final-
    horizon export only. On abort, exports ckpt_last_good.pt from the
    last COMPLETED epoch (deep-copied; never stale-mutated).

    REQUIRED bindings for a real run: expected_source_hashes (full
    effective-code map), expected_manifest (REFIT stage, checked via
    check_expected_manifest_refit), and binding = {selection_manifest_
    sha256, selected_epoch, bundle_npz_sha256, meta_sha256,
    audit_sha256}; selected_epoch must equal cfg['max_epochs']."""
    import torch
    import losses
    import c4_model as c4m

    rep = preflight_refit(cfg, data)
    verify_source_pins(expected_source_hashes)
    if binding is not None:
        if int(cfg['max_epochs']) != binding.get('selected_epoch'):
            _fail('horizon %r != bound selected_epoch %r' %
                  (cfg['max_epochs'], binding.get('selected_epoch')))
    envv = u1r.env_check(cfg)
    bank, bank_sha, bank_counts = c2r.build_negative_bank(cfg, data)
    recon_block, artifacts = u1r._resolve_recon(
        cfg, data, meta_path, audit_path, data_file_sha)
    frozen = u1r.frozen_manifest_u1(
        cfg, rep, envv, bank_sha, bank_counts, recon_block)
    frozen['stage'] = 'U1_REFIT_NOVAL'
    frozen['validation'] = 'none — fixed horizon, selection done'
    if binding is not None:
        frozen['refit_binding'] = dict(binding)
    if expected_manifest is not None:
        check_expected_manifest_refit(frozen, expected_manifest)
    os.makedirs(out_dir, exist_ok=False)
    _atomic_json(frozen, os.path.join(out_dir,
                                      'manifest_frozen.json'))
    if artifacts is not None:
        u1r._persist_targets(out_dir, artifacts)

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
    opt = s1r.build_optimizer(model, cfg)
    tp = s1r._pairs_checked_noerr(data['train_pairs'])
    tp_t = torch.tensor(tp, dtype=torch.long, device=device)
    y_tr = np.asarray(data['y_train'], dtype=np.float64)
    w = losses.class_weights_train(y_tr, beta=float(cfg['beta']))
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    unit_mass = float((torch.tensor(y_tr, dtype=torch.float32)
                       * w_t.view(1, -1)).sum())
    pos_t = torch.tensor(np.asarray(data['train_mask']).astype(bool),
                         device=device)
    neg_t = torch.tensor(bank, device=device)
    tensors = None
    if artifacts is not None:
        tensors = {'mi_target': torch.tensor(
            artifacts['mi_target'], dtype=torch.float32, device=device),
            'mi_valid': torch.tensor(artifacts['mi_valid_mask'],
                                     device=device),
            'dis_target': torch.tensor(artifacts['dis_target'],
                                       dtype=torch.float32,
                                       device=device),
            'dis_valid': torch.tensor(artifacts['dis_valid_mask'],
                                      device=device)}

    trajectory, dynamic_history = [], []
    dynamic_state = None
    aborted, attempted = None, 0
    completed, last_good = 0, None
    epoch = 0
    for epoch in range(1, int(cfg['max_epochs']) + 1):
        attempted = epoch
        model.train()
        opt.zero_grad(set_to_none=True)
        prev = s1r._flat_params(model).clone()
        try:
            out = model(inputs, G, ei, tp_t)
            l_type = losses.type_loss(out['logits'], y_tr, w_t)
            it = c3r._intra_terms(model, out, cfg)
            et = u1r._inter_terms_u1(out, pos_t, neg_t, cfg)
            rt_ = u1r._recon_terms(model, out, tensors, cfg)
            loss = combine_objectives(l_type, it, et, rt_, unit_mass,
                                      cfg)
        except (ValueError, FloatingPointError) as e:
            aborted = ('nonfinite input/logits/term rejected at epoch '
                       '%d: %s' % (epoch, e))
            break
        if not torch.isfinite(loss):
            aborted = 'nonfinite loss at epoch %d' % epoch
            break
        loss.backward()
        gnorm, bad_grad = 0.0, False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_grad = True
                break
            gnorm += float(p.grad.norm()) ** 2
        if bad_grad:
            aborted = 'nonfinite grad at epoch %d' % epoch
            break
        gnorm = gnorm ** 0.5
        opt.step()
        if not all(torch.isfinite(p).all()
                   for p in model.parameters()):
            aborted = ('nonfinite params after optimizer at epoch %d'
                       % epoch)
            break
        refresh_meta = None
        if cfg['dynamic_enabled'] and dg.should_update(epoch):
            try:
                old_g_sha = {k: u1r._sha_bytes(v.numpy().tobytes())
                             for k, v in G.items()}
                old_ei_sha = {str(k): u1r._sha_bytes(
                    v.numpy().tobytes()) for k, v in ei.items()}
                dynamic_state = dg.refresh_similarity_graphs(
                    rt_['gram_mi'].detach(), rt_['gram_dis'].detach(),
                    ei, epoch, k=int(cfg['knn_k']),
                    threshold=cfg['dynamic_threshold'])
                refresh_meta = dict(dynamic_state['meta'])
                refresh_meta.update({
                    'producer_epoch': epoch,
                    'producer_phase':
                        'training_forward_pre_optimizer',
                    'application_phase':
                        'post_optimizer_pre_next_epoch',
                    'producer_gram_mi_sha256': u1r._sha_bytes(
                        rt_['gram_mi'].detach().numpy().tobytes()),
                    'producer_gram_dis_sha256': u1r._sha_bytes(
                        rt_['gram_dis'].detach().numpy().tobytes()),
                    'input_G_sha': old_g_sha,
                    'input_ei_sha': old_ei_sha})
                dynamic_state['meta'] = refresh_meta
                G, ei = (dynamic_state['G'],
                         dynamic_state['edge_index_dict'])
                dynamic_history.append(refresh_meta)
            except (ValueError, FloatingPointError) as e:
                aborted = ('dynamic refresh rejected at epoch %d: %s'
                           % (epoch, e))
                break
        upd = float((s1r._flat_params(model) - prev).norm())
        rec = {'epoch': epoch,
               'train_loss': float(loss.detach()),
               'L_type': float(l_type.detach()),
               'grad_norm': gnorm, 'update_norm': upd}
        if cfg['unit_mode'] == 'common_mass_sums':
            rec['unit_mass'] = unit_mass
            rec['unit_intra_factor'] = (len(inputs['mi_v1'])
                                        + len(inputs['dis_v1'])) \
                / unit_mass
            rec['unit_inter_info_factor'] = int(pos_t.sum()) \
                / unit_mass
            rec['unit_inter_margin_factor'] = 1. / unit_mass
            rec['unit_recon_factor'] = (
                rt_['recon_valid_mi'] + rt_['recon_valid_dis']
            ) / unit_mass if rt_ is not None else 0.
        if cfg['dynamic_enabled']:
            rec['graph_epoch'] = (dynamic_state['meta']['epoch']
                                  if dynamic_state else 0)
            rec['dynamic_refreshed'] = refresh_meta is not None
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
        trajectory.append(rec)
        completed = epoch
        last_good = _clone_ckpt(_ckpt_refit(
            model, opt, epoch, G, ei, cfg, neg_t, bank_sha, tensors,
            recon_block, dynamic_state))
        log('epoch %d loss %.5f' % (epoch, rec['train_loss']))

    result = {'stage': 'U1_REFIT_NOVAL',
              'completed_epochs': completed,
              'horizon': int(cfg['max_epochs']),
              'aborted': aborted,
              'attempted_epoch': attempted if aborted else None,
              'validation_consumed': False,
              'dynamic_history': dynamic_history}
    if aborted is None:
        _atomic_save(last_good, os.path.join(out_dir,
                                             'ckpt_final.pt'))
    elif last_good is not None:
        last_good['exports_from_completed_epoch'] = completed
        _atomic_save(last_good, os.path.join(out_dir,
                                             'ckpt_last_good.pt'))
    _atomic_json(trajectory, os.path.join(out_dir,
                                          'trajectory_noval.json'))
    # post-work pin re-verification — source drift mid-run is detected
    verify_source_pins(expected_source_hashes)
    _atomic_json({'frozen': frozen, 'result': result},
                 os.path.join(out_dir, 'manifest_noval.json'))
    return result


def _atomic_save(obj, path):
    """torch.save is non-atomic; stage to <path>.part then rename."""
    import torch
    part = path + '.part'
    torch.save(obj, part)
    os.replace(part, path)


def _atomic_json(obj, path):
    """Atomic json write: <path>.part -> os.replace."""
    part = path + '.part'
    with open(part, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(part, path)


def load_refit_ckpt(path, expect_sha):
    """Trusted checkpoint load (R3): verify the exact external byte SHA
    BEFORE any deserialization, then verify every recorded semantic
    hash (G/ei/bank/recon tensors) against the actual tensors —
    a corrupted checkpoint is refused even when its file pin matches.
    Returns the verified checkpoint dict."""
    import torch
    if not _is_sha(expect_sha):
        _fail('ckpt sha pin must be 64 lowercase hex')
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if got != expect_sha:
        _fail('ckpt sha mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        _fail('ckpt not a dict')
    if ck.get('stage') != 'U1_REFIT_NOVAL':
        _fail('ckpt stage must be U1_REFIT_NOVAL, got %r'
              % ck.get('stage'))
    if 'val_macro_f1' in ck:
        _fail('refit ckpt must not carry val_macro_f1')
    if type(ck.get('epoch')) is not int or ck['epoch'] <= 0:
        _fail('ckpt epoch must be positive int')
    if not isinstance(ck.get('cfg'), dict):
        _fail('ckpt missing cfg dict')
    want_g = {k: u1r._sha_bytes(v.cpu().numpy().tobytes())
              for k, v in ck['G'].items()}
    if ck.get('G_sha') != want_g:
        _fail('ckpt G does not match recorded G_sha')
    want_ei = {str(k): u1r._sha_bytes(v.cpu().numpy().tobytes())
               for k, v in ck['ei'].items()}
    if ck.get('ei_sha') != want_ei:
        _fail('ckpt ei does not match recorded ei_sha')
    if ck.get('neg_bank_sha') != u1r._sha_bytes(
            ck['neg_bank'].cpu().numpy().tobytes()):
        _fail('ckpt neg_bank does not match recorded neg_bank_sha')
    if 'recon_tensor_sha' in ck:
        for k, want in ck['recon_tensor_sha'].items():
            got_k = u1r._sha_bytes(
                ck['recon_targets'][k].cpu().numpy().tobytes())
            if got_k != want:
                _fail('ckpt recon target %s sha mismatch' % k)
    return ck


# ---------- stage-aware consumer contract (M3) ---------------------------
def restore_refit_state(model, opt, checkpoint):
    """Consumer contract for U1_REFIT_NOVAL checkpoints — the ONLY
    sanctioned load path for future authorized outer evaluation.
    Refuses any other stage; restores model/opt/RNG/graph/bank and
    returns the restored graph state. Consumption by a scorer still
    requires that scorer to supply its OWN outer-pair list; nothing
    here reads or implies held-out labels."""
    import torch
    if checkpoint.get('stage') != 'U1_REFIT_NOVAL':
        raise ValueError('not a U1_REFIT_NOVAL checkpoint: %r'
                         % (checkpoint.get('stage'),))
    if 'val_macro_f1' in checkpoint:
        raise ValueError('refit ckpt must not carry val_macro_f1')
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['opt_state'])
    torch.set_rng_state(checkpoint['rng_torch'])
    np.random.set_state(checkpoint['rng_numpy'])
    return {'G': checkpoint['G'], 'ei': checkpoint['ei'],
            'epoch': checkpoint['epoch'],
            'dynamic_state': checkpoint.get('dynamic_state'),
            'neg_bank_sha': checkpoint.get('neg_bank_sha'),
            'cfg': checkpoint['cfg']}


def forward_refit(model, inputs, G, ei, pairs):
    """Fixed-checkpoint inference for an authorized outer scorer:
    eval-mode, no_grad forward on a caller-supplied pair list.
    The caller owns the pair list and any label bookkeeping — this
    function never sees labels."""
    import torch
    model.eval()
    tp = torch.tensor(np.asarray(pairs), dtype=torch.long)
    with torch.no_grad():
        out = model(inputs, G, ei, tp)
    if not torch.isfinite(out['logits']).all():
        raise FloatingPointError('nonfinite refit forward logits')
    return out['logits'].cpu().numpy()


# ---------- authorized driver (M1+M2+R1) ---------------------------------
def main(argv=None):
    """Refit driver. Exit 0 success / 3 gate refusal / 4 aborted run.
    Selection lineage is VERIFIED, not caller-asserted: the selection
    manifest FILE is read only after its exact sha pin is verified,
    selected_epoch is DERIVED from best.epoch (never a free integer),
    fold/candidate identity is cross-checked against the pinned bundle
    meta, and refit cfg must equal the selection cfg outside the
    declared allowed set."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='refit bundle npz (train-only)')
    ap.add_argument('--data-sha', help='bundle sha pin')
    ap.add_argument('--sel-bundle', help='selection bundle npz FILE')
    ap.add_argument('--sel-bundle-sha', help='selection bundle pin')
    ap.add_argument('--sel-meta', help='selection meta FILE')
    ap.add_argument('--sel-meta-sha', help='selection meta pin')
    ap.add_argument('--candidate-csv', help='candidate csv FILE')
    ap.add_argument('--candidate-csv-sha', help='candidate csv pin')
    ap.add_argument('--candidate-manifest',
                    help='candidate manifest FILE')
    ap.add_argument('--candidate-manifest-sha',
                    help='candidate manifest pin')
    ap.add_argument('--precontract',
                    help='pre-declared selection contract json FILE')
    ap.add_argument('--precontract-sha', help='precontract pin')
    ap.add_argument('--success-record',
                    help='wrapper-emitted success record FILE')
    ap.add_argument('--success-record-sha',
                    help='success record pin')
    ap.add_argument('--bundle-meta',
                    help='candidate_fold_data.meta.json of the bundle')
    ap.add_argument('--bundle-meta-sha', help='bundle meta sha pin')
    ap.add_argument('--cfg', help='refit cfg json')
    ap.add_argument('--expected-manifest',
                    help='emitted refit expected manifest json')
    ap.add_argument('--meta', help='recon meta sidecar')
    ap.add_argument('--audit', help='recon audit sidecar')
    ap.add_argument('--meta-sha', help='meta sidecar pin')
    ap.add_argument('--audit-sha', help='audit sidecar pin')
    ap.add_argument('--selection-manifest',
                    help='U1 selection run manifest.json FILE')
    ap.add_argument('--selection-manifest-sha',
                    help='exact sha of the selection manifest FILE')
    ap.add_argument('--selection-binding',
                    help='pinned binding sidecar json FILE')
    ap.add_argument('--selection-binding-sha',
                    help='exact sha of the binding sidecar FILE')
    ap.add_argument('--source-hashes',
                    help='json file of expected_source_hashes')
    ap.add_argument('--emit-source-map', action='store_true',
                    help='print effective source map and exit')
    ap.add_argument('--emit-manifest', action='store_true',
                    help='emit expected manifest to --out and exit')
    ap.add_argument('--out', help='output dir (must not exist)')
    args = ap.parse_args(argv)

    try:
        if args.emit_source_map:
            print(json.dumps(effective_source_map(), indent=1,
                             sort_keys=True))
            return 0
        if args.emit_manifest and args.expected_manifest:
            _fail('--emit-manifest and --expected-manifest are '
                  'mutually exclusive')
        req = ['data', 'data_sha', 'cfg', 'source_hashes', 'out',
               'selection_manifest', 'selection_manifest_sha',
               'selection_binding', 'selection_binding_sha',
               'bundle_meta', 'bundle_meta_sha', 'sel_bundle',
               'sel_bundle_sha', 'sel_meta', 'sel_meta_sha',
               'candidate_csv', 'candidate_csv_sha',
               'candidate_manifest', 'candidate_manifest_sha',
               'precontract', 'precontract_sha',
               'success_record', 'success_record_sha']
        if not args.emit_manifest:
            req.append('expected_manifest')
        for r in req:
            if getattr(args, r) is None:
                _fail('--%s is required' % r.replace('_', '-'))
        # verified selection lineage BEFORE any data consumption
        sel = verify_and_parse_selection_manifest(
            args.selection_manifest, args.selection_manifest_sha)
        bind = verify_and_parse_selection_binding(
            args.selection_binding, args.selection_binding_sha,
            args.selection_manifest_sha)
        if 'selected_epoch' in bind \
                and bind['selected_epoch'] != sel['epoch']:
            _fail('binding selected_epoch %d != manifest-derived '
                  'epoch %d' % (bind['selected_epoch'], sel['epoch']))
        # R2-closure: sel-side FILES verified byte-wise and sel-side
        # digests recomputed — not just caller-pin comparison.
        _require_sha_file(args.sel_bundle, args.sel_bundle_sha,
                          'sel bundle')
        _require_sha_file(args.sel_meta, args.sel_meta_sha,
                          'sel meta')
        _require_sha_file(args.candidate_csv, args.candidate_csv_sha,
                          'candidate csv')
        _require_sha_file(args.candidate_manifest,
                          args.candidate_manifest_sha,
                          'candidate manifest')
        pc = _pinned_json(args.precontract, args.precontract_sha,
                          'precontract')
        if pc.get('stage') != 'PRESELECTION_CONTRACT':
            _fail('precontract stage wrong')
        if pc.get('sel_bundle_sha256') != args.sel_bundle_sha:
            _fail('precontract sel bundle pin mismatch')
        if pc.get('sel_cfg_sha256') != u1r._sha_bytes(
                json.dumps(sel['cfg'], sort_keys=True,
                           default=str).encode()):
            _fail('precontract cfg pin does not match manifest cfg')
        # rev8: exact key-set equality to manifest source map incl.
        # imported_pure_hashes — subset maps refuse.
        smap = pc.get('sel_source_map')
        if not isinstance(smap, dict):
            _fail('precontract missing sel_source_map')
        mh = dict(sel.get('source_hashes') or {})
        mh.update(sel.get('imported_pure_hashes') or {})
        if set(smap) != set(mh):
            _fail('precontract source map keys != manifest keys '
                  '(missing %s extra %s)'
                  % (sorted(set(mh) - set(smap)),
                     sorted(set(smap) - set(mh))))
        for k, v in smap.items():
            if mh.get(k) != v:
                _fail('precontract source pin %s != manifest '
                      'source_hashes' % k)
        # rev8: success record — wrapper-emitted, sha-verified file.
        sr = _pinned_json(args.success_record,
                          args.success_record_sha, 'success record')
        if sr.get('stage') != 'SELECTION_SUCCESS_RECORD':
            _fail('success record stage wrong')
        if sr.get('aborted') is not None:
            _fail('success record shows aborted selection run')
        if sr.get('precontract_sha256') != args.precontract_sha:
            _fail('success record does not pin this precontract')
        if sr.get('selection_manifest_sha256')                 != args.selection_manifest_sha:
            _fail('success record does not pin this manifest')
        if sr.get('sel_bundle_sha256') != args.sel_bundle_sha:
            _fail('success record sel bundle pin mismatch')
        if sr.get('best_epoch') != sel['epoch']:
            _fail('success record best_epoch != manifest best.epoch')
        sdata = load_refit_npz.__self__ if False else None
        with np.load(args.sel_bundle, allow_pickle=False) as z:
            sb = {k: z[k] for k in z.files}
        dg = bind['digests']
        stp = np.asarray(sb['train_pairs'], dtype=np.int64)
        svp = np.asarray(sb['val_pairs'], dtype=np.int64)
        if dg.get('sel_train_pairs_sorted') != u1r._sha_bytes(
                np.ascontiguousarray(np.asarray(sorted(
                    map(tuple, stp.tolist())), dtype=np.int64)
                    ).tobytes()):
            _fail('sel train pairs digest does not recompute')
        if dg.get('sel_val_pairs_sorted') != u1r._sha_bytes(
                np.ascontiguousarray(np.asarray(sorted(
                    map(tuple, svp.tolist())), dtype=np.int64)
                    ).tobytes()):
            _fail('sel val pairs digest does not recompute')
        if dg.get('sel_train_mask') != u1r._sha_bytes(
                np.ascontiguousarray(np.asarray(
                    sb['train_mask']).astype(bool)).tobytes()):
            _fail('sel train mask digest does not recompute')
        if bind['counts']['inner_train'] != len(stp)                 or bind['counts']['inner_val'] != len(svp):
            _fail('binding sel counts != actual arrays')
        # R2: recompute view + y digests vs the MANIFEST's sealed
        # hashes — sel feature/label consumption runner-verified.
        sdh = sel.get('data_hashes') or {}
        svh_path = args.selection_manifest
        svh = _pinned_json(svh_path, args.selection_manifest_sha,
                           'sel manifest reparse').get('view_hashes')
        for sk in ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2'):
            if svh.get(sk) != u1r._sha_bytes(np.ascontiguousarray(
                    np.asarray(sb['view_' + sk])).tobytes()):
                _fail('manifest view_hashes.%s does not recompute '
                      'from actual sel bundle' % sk)
        for yk in ('y_train', 'y_val'):
            if sdh.get(yk) != u1r._sha_bytes(np.ascontiguousarray(
                    np.asarray(sb[yk])).tobytes()):
                _fail('manifest data_hashes.%s does not recompute'
                      % yk)
        for pk, arr in (('train_pairs', stp), ('val_pairs', svp),
                        ('train_mask',
                         np.asarray(sb['train_mask']).astype(bool))):
            if sdh.get(pk) != u1r._sha_bytes(
                    np.ascontiguousarray(np.asarray(arr)).tobytes()):
                _fail('manifest data_hashes.%s does not recompute'
                      % pk)
        # R4: derive val_mask from val_pairs + compare binding digest
        svm = np.zeros_like(np.asarray(sb['train_mask']))
        for i, j in svp.tolist():
            svm[i, j] = 1
        if 'sel_val_mask' not in dg:
            _fail('binding missing sel_val_mask digest')
        if dg['sel_val_mask'] != u1r._sha_bytes(
                np.ascontiguousarray(svm.astype(bool)).tobytes()):
            _fail('binding sel_val_mask digest does not recompute')
        # R3: parse sha-verified sel meta — fold/phase/candidate
        sm = _pinned_json(args.sel_meta, args.sel_meta_sha,
                          'sel meta')
        if sm.get('stage') != 'CANDIDATE_FOLD_BUNDLE_REV3'                 or sm.get('phase') != 'selection':
            _fail('sel meta stage/phase wrong')
        if sm.get('fold') != bind['fold']:
            _fail('sel meta fold != binding fold')
        if sm.get('pins', {}).get('pair_split_candidate.csv')                 != bind['candidate_csv_sha256']:
            _fail('sel meta candidate pin != binding')
        # S3: digest binding internals verified against CURRENT
        # inputs — a structurally valid but stale/forged binding fails.
        for bk, av in (('sel_bundle_sha256', args.sel_bundle_sha),
                       ('sel_meta_sha256', args.sel_meta_sha),
                       ('candidate_csv_sha256',
                        args.candidate_csv_sha),
                       ('candidate_manifest_sha256',
                        args.candidate_manifest_sha),
                       ('refit_bundle_sha256', args.data_sha),
                       ('refit_meta_sha256', args.bundle_meta_sha)):
            if bind.get(bk) != av:
                _fail('binding %s does not match current input pin'
                      % bk)
        if bind['counts']['refit_union'] is not None:
            pass  # verified post-load below
        bundle_meta = verify_and_parse_bundle_meta(
            args.bundle_meta, args.bundle_meta_sha, bind)
        cfg = json.load(open(args.cfg, encoding='utf-8'))
        if not isinstance(cfg, dict):
            _fail('cfg must be a JSON object')
        check_cfg_compat(sel['cfg'], cfg)
        if int(cfg['max_epochs']) != sel['epoch']:
            _fail('cfg.max_epochs %d != manifest-derived selected '
                  'epoch %d' % (int(cfg['max_epochs']), sel['epoch']))
        _require_sha_file(args.data, args.data_sha, 'bundle npz')
        exp_src = _pinned_json(args.source_hashes,
                               _sha_lenient_file(args.source_hashes),
                               'source-hashes json')
        exp_mf = (json.load(open(args.expected_manifest,
                                 encoding='utf-8'))
                  if not args.emit_manifest else None)
        for path, pin, nm in ((args.meta, args.meta_sha, 'meta'),
                              (args.audit, args.audit_sha, 'audit')):
            if path is not None:
                _require_sha_file(path, pin, nm + ' sidecar')
        data = load_refit_npz(args.data, expect_sha=args.data_sha)
        # S3b: recompute refit digests/counts from the ACTUAL arrays
        tp = np.asarray(data['train_pairs'], dtype=np.int64)
        if bind['counts']['refit_union'] != len(tp):
            _fail('binding refit_union %d != actual train_pairs %d'
                  % (bind['counts']['refit_union'], len(tp)))
        dg = bind['digests']
        want_rp = u1r._sha_bytes(np.ascontiguousarray(
            np.asarray(sorted(map(tuple, tp.tolist())),
                               dtype=np.int64)).tobytes())
        if dg.get('refit_train_pairs_sorted') != want_rp:
            _fail('binding refit pairs digest does not recompute')
        want_rm = u1r._sha_bytes(np.ascontiguousarray(
            np.asarray(data['train_mask']).astype(bool)).tobytes())
        if dg.get('refit_train_mask') != want_rm:
            _fail('binding refit mask digest does not recompute')
        binding = {'selection_manifest_sha256':
                   args.selection_manifest_sha,
                   'selected_epoch': sel['epoch'],
                   'fold': bind['fold'],
                   'candidate_csv_sha256':
                   bind['candidate_csv_sha256'],
                   'selection_binding_sha256':
                   args.selection_binding_sha,
                   'bundle_meta_sha256': args.bundle_meta_sha,
                   'bundle_npz_sha256': args.data_sha,
                   'meta_sha256': args.meta_sha,
                   'audit_sha256': args.audit_sha}
        if args.emit_manifest:
            m = expected_manifest_for_refit(
                cfg, data, binding, args.meta, args.audit,
                args.data_sha)
            verify_source_pins(exp_src)
            if os.path.exists(args.out):
                _fail('emit target exists — refusing to clobber: '
                      + args.out)
            parent = os.path.dirname(os.path.abspath(args.out))
            os.makedirs(parent, exist_ok=True)
            _atomic_json(m, args.out)
            return 0
        if os.path.exists(args.out):
            _fail('output dir exists — refusing: ' + args.out)
        # R7: selection-era sel_source_map must equal the
        # refit-era exp_src on shared keys — drifted sealed modules
        # between the two runs refuse (orchestrators/* are
        # refit-era-only additions, exempt).
        for k, v in smap.items():
            if k in exp_src and exp_src[k] != v:
                _fail('source parity: %s differs selection-era vs '
                      'refit-era' % k)
        res = train_refit_noval(
            cfg, data, args.out, expected_manifest=exp_mf,
            meta_path=args.meta, audit_path=args.audit,
            data_file_sha=args.data_sha,
            expected_source_hashes=exp_src, binding=binding)
        return 4 if res['aborted'] is not None else 0
    except (ValueError, FloatingPointError, TypeError, OSError,
            KeyError, AttributeError,
            json.JSONDecodeError) as e:
        print('refusal: %s' % e, file=sys.stderr)
        return 3


def sel_manifest_src(sel):
    """source_hashes carried by the parsed selection manifest —
    the parsed dict keeps only epoch/cfg/data_hashes, so re-derive
    from the raw manifest is done in caller; here sel is the parsed
    return; manifest source map is fetched from binding path again."""
    return sel.get('source_hashes') or {}


def _sha_lenient_file(path):
    """sha of a file we authored ourselves (source-hashes json) — used
    only to feed _pinned_json's pin argument; content validity is then
    checked by verify_source_pins against effective_source_map()."""
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _require_sha_file(path, expect, what):
    if not _is_sha(expect):
        _fail('%s sha pin must be 64 lowercase hex' % what)
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if got != expect:
        _fail('%s sha mismatch' % what)


if __name__ == '__main__':
    sys.exit(main() or 0)
