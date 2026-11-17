# -*- coding: utf-8 -*-
"""Candidate-split fold adapter REV2 — runner-consumable, fail-closed.

Fixes over REV1 (coordinator review):
 * emits the runner.py FLAT schema (view_mi_v1/.../view_dis_v2,
   train_mask, train_pairs, val_pairs, y_train, y_val) — no object
   arrays, loadable via runner.load_data_npz(allow_pickle=False);
 * ALL external SHA pins REQUIRED and verified BEFORE consuming data:
   candidate csv, candidate manifest, strict npz, imported deps
   (semantic/functional/contracts), MeSH inputs; the trusted manifest is
   checked to SEAL the csv + strict npz (manifest.output_sha256 /
   manifest.input_sha256 must equal the pinned+measured hashes);
 * candidate content checks: record uniqueness, nonempty group keys,
   index bounds vs strict dims, completeness vs the manifest's OWN
   declared gates (positive_universe / pair_groups_total read from the
   sealed manifest, not hardcoded);
 * atomic two-file publication: stage in <out>.part/ then rename; any
   error removes the staging dir — no partial artifacts, no clobber;
 * post-build input re-verification (mutation gate);
 * phases explicit: selection -> nonempty inner_val exported (runner
   preflight requires nonempty val); refit -> val keys OMITTED entirely
   (no fabricated validation; consumption needs a separately-authorized
   no-validation fixed-horizon runner — declared blocker, see report).

Held-out-label absence is proven by SET-EQUALITY between exported pair
lists and the pinned candidate's non-test segment membership — not by
key names alone. Y is read only at chosen train/val cells.

No balancing/selection/stopping uses labels. No edits to sealed stages.
"""
import argparse, csv, hashlib, json, os, shutil, sys, tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PHASES = ('selection', 'refit')
EXPECTED_COLS = (['mir_idx', 'dis_idx', 'group_key']
                 + [f'seg_fold{f}' for f in range(5)])
DEP_FILES = ('semantic.py', 'functional.py', 'contracts.py')


def _fail(msg):
    raise ValueError('gate failure: ' + msg)


def _sha_f(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _require_sha(path, expect, what):
    if expect is None:
        _fail('%s sha pin is REQUIRED but missing' % what)
    got = _sha_f(path)
    if got != expect:
        _fail('%s sha mismatch: %s != %s' % (what, got, expect))
    return got


def load_strict_pinned(path, expect_sha):
    """Strict npz contains OBJECT arrays (entity names, 'extra' dict) —
    allow_pickle=False fails on the real artifact. Policy: verify the
    EXACT byte pin BEFORE any deserialization; only then np.load with
    allow_pickle=True on the verified bytes; extract only the needed
    keys; validate names are plain strings and shapes match the matrix
    dims. Returns plain (non-object) structures only."""
    _require_sha(path, expect_sha, 'strict npz (pre-deserialize pin)')
    z = np.load(path, allow_pickle=True)   # justified: bytes pinned above
    need = ('Y', 'sim_m1', 'sim_d1', 'mir_names', 'dis_names')
    for k in need:
        if k not in z.files:
            _fail('strict npz missing key ' + k)
    Y = np.asarray(z['Y'])
    if Y.dtype == object or Y.ndim != 3 or not np.isfinite(Y).all():
        _fail('Y invalid')
    n_m, n_d, C = Y.shape
    def _names(k, n):
        out = z[k].tolist()
        if len(out) != n or not all(isinstance(x, str) and x
                                    for x in out):
            _fail('%s not a nonempty string list of len %d' % (k, n))
        return out
    mir_names = _names('mir_names', n_m)
    dis_names = _names('dis_names', n_d)
    if not np.isin(Y, (0, 1)).all():
        _fail('Y not binary — _multihot thresholds Y>0 silently '
              'otherwise')
    sims = {}
    for k, n in (('sim_m1', n_m), ('sim_d1', n_d)):
        a = np.asarray(z[k])
        if a.dtype == object or a.shape != (n, n) \
                or not np.isfinite(a).all():
            _fail('%s invalid' % k)
        if not np.allclose(a, a.T, rtol=1e-5, atol=1e-8):
            _fail('%s not symmetric' % k)
        sims[k] = a
    return {'Y': Y, 'sim_m1': sims['sim_m1'], 'sim_d1': sims['sim_d1'],
            'mir_names': mir_names, 'dis_names': dis_names}


# ---------- candidate gates ----------
def _stat_expect(mf, fold):
    """Read the sealed candidate manifest's stat_gates with exact
    schema+type validation (B1: real manifest carries 'stat_gates',
    NOT 'gates'). Returns the completeness expectations plus the
    declared per-fold inner/test counts for gate checks."""
    sg = mf.get('stat_gates')
    if not isinstance(sg, dict):
        _fail('manifest missing stat_gates dict')
    for k in ('positive_universe', 'pair_groups_total'):
        if type(sg.get(k)) is not int or sg[k] <= 0:
            _fail('stat_gates.%s must be a positive int' % k)
    fk = 'fold%d_test_cells' % fold
    if type(sg.get(fk)) is not int or sg[fk] <= 0:
        _fail('stat_gates.%s must be a positive int' % fk)
    ig = sg.get('inner_groups')
    if not isinstance(ig, dict):
        _fail('stat_gates.inner_groups must be a dict')
    g = ig.get(str(fold))
    if not isinstance(g, dict) or any(
            type(g.get(k)) is not int or g[k] <= 0
            for k in ('train_groups', 'inner_val_groups')):
        _fail('stat_gates.inner_groups[%d] invalid' % fold)
    return {'positive_universe': sg['positive_universe'],
            'pair_groups_total': sg['pair_groups_total'],
            'fold_test_cells': sg[fk],
            'train_groups': g['train_groups'],
            'inner_val_groups': g['inner_val_groups']}


def _fold_counts_gate(recs, fold, expect):
    """Verify the declared per-fold counts against the actual records
    (B1 fold-level binding): test-cell count for this fold, inner-train
    and inner-val group counts."""
    test_cells = sum(1 for r in recs if r['seg'][fold] == 'test')
    if test_cells != expect['fold_test_cells']:
        _fail('fold %d test cells %d != declared %d'
              % (fold, test_cells, expect['fold_test_cells']))
    groups = {}
    for r in recs:
        groups.setdefault(r['grp'], set()).add(r['seg'][fold])
    # declared train_groups = all NON-test groups (inner_train+inner_val)
    it = sum(1 for s in groups.values() if 'test' not in s)
    iv = sum(1 for s in groups.values()
             if 'inner_val' in s and 'test' not in s)
    if it != expect['train_groups']:
        _fail('fold %d non-test groups %d != declared %d'
              % (fold, it, expect['train_groups']))
    if iv != expect['inner_val_groups']:
        _fail('fold %d inner-val groups %d != declared %d'
              % (fold, iv, expect['inner_val_groups']))


def load_candidate_csv(path, n_mi, n_d, expect):
    """Schema, order, uniqueness, nonempty keys, bounds, completeness.
    `expect` = {'positive_universe': int, 'pair_groups_total': int}
    taken from the sealed candidate manifest."""
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows:
        _fail('empty candidate csv')
    if list(rows[0].keys()) != EXPECTED_COLS:
        _fail('candidate csv schema drift')
    seen, groups = set(), set()
    recs = []
    for r in rows:
        k = (int(r['mir_idx']), int(r['dis_idx']))
        if k in seen:
            _fail('duplicate record %s' % (k,))
        seen.add(k)
        g = r['group_key']
        if not isinstance(g, str) or not g.strip():
            _fail('empty group key')
        groups.add(g)
        if not (0 <= k[0] < n_mi and 0 <= k[1] < n_d):
            _fail('record out of bounds %s' % (k,))
        seg = [r[f'seg_fold{f}'] for f in range(5)]
        if not set(seg) <= {'test', 'inner_train', 'inner_val'}:
            _fail('unknown segment label %s' % seg)
        recs.append({'mir': k[0], 'dis': k[1], 'grp': g, 'seg': seg})
    keys = [(r['mir'], r['dis']) for r in recs]
    if keys != sorted(keys):
        _fail('candidate record order drift')
    if expect:
        if len(recs) != expect['positive_universe']:
            _fail('record count %d != manifest positive_universe %d'
                  % (len(recs), expect['positive_universe']))
        if len(groups) != expect['pair_groups_total']:
            _fail('group count %d != manifest pair_groups_total %d'
                  % (len(groups), expect['pair_groups_total']))
    return recs


def select_fold_pairs(records, fold, phase):
    if phase not in PHASES:
        _fail('unknown phase %r' % phase)
    if not (0 <= fold < 5):
        _fail('fold out of range')
    grp_seg = {}
    for r in records:
        grp_seg.setdefault(r['grp'], set()).add(r['seg'][fold])
    if any(len(s) > 1 for s in grp_seg.values()):
        _fail('pair-group straddles segments')
    tr = [(r['mir'], r['dis']) for r in records
          if r['seg'][fold] == 'inner_train']
    iv = [(r['mir'], r['dis']) for r in records
          if r['seg'][fold] == 'inner_val']
    te = {(r['mir'], r['dis']) for r in records
          if r['seg'][fold] == 'test'}
    if not tr or (phase == 'selection' and not iv):
        _fail('empty required segment for phase %s' % phase)
    if set(tr) & set(iv) or (set(tr) | set(iv)) & te:
        _fail('segment overlap incl. test leakage')
    if phase == 'selection':
        return sorted(set(tr)), sorted(set(iv))
    return sorted(set(tr) | set(iv)), None    # refit: val OMITTED


# ---------- bundle ----------
def build_flat(Y, train_pairs, val_pairs, S_seq, S_gene,
               functional_fn, semantic_fn):
    """Runner-flat schema. S_func rebuilt from train mask only."""
    n_m, n_d, C = Y.shape
    A = np.zeros((n_m, n_d), dtype=np.int8)
    for (i, j) in train_pairs:
        A[i, j] = 1
    S_func = np.asarray(functional_fn(A.astype(np.float64)))
    S_sem = np.asarray(semantic_fn())
    flat = {
        'view_mi_v1': np.hstack([S_seq, A.astype(np.float64)]),
        'view_mi_v2': np.hstack([S_func, A.astype(np.float64)]),
        'view_dis_v1': np.hstack([S_gene, A.T.astype(np.float64)]),
        'view_dis_v2': np.hstack([S_sem, A.T.astype(np.float64)]),
        'train_mask': A,
        'train_pairs': np.asarray(train_pairs, dtype=np.int64),
        'y_train': _multihot(Y, train_pairs, C)}
    for k, v in flat.items():
        a = np.asarray(v)
        if a.dtype == object:
            _fail('object array refused: ' + k)
        if k.startswith('view_') and not np.isfinite(a).all():
            _fail('nonfinite view ' + k)
    if val_pairs is not None:
        flat['val_pairs'] = np.asarray(val_pairs, dtype=np.int64)
        flat['y_val'] = _multihot(Y, val_pairs, C)
    return flat


def _multihot(Y, pairs, C):
    out = np.zeros((len(pairs), C), dtype=np.int8)
    for r, (i, j) in enumerate(pairs):
        out[r] = (Y[i, j] > 0).astype(np.int8)
    return out


# ---------- refit data-gate (no-validation variant, honest scope) ------
def prefit_gate(flat, n_types):
    """Runner-preflight-equivalent DATA checks for the refit bundle:
    identical to runner.preflight's data section minus val-pair
    requirements. Selection bundles use runner.preflight itself; this
    exists only because refit intentionally has no validation split."""
    tp, tm, yt = (np.asarray(flat['train_pairs']),
                  np.asarray(flat['train_mask']),
                  np.asarray(flat['y_train']))
    if tp.ndim != 2 or tp.shape[1] != 2 or tp.shape[0] == 0:
        _fail('train_pairs invalid')
    if len(set(map(tuple, tp.tolist()))) != len(tp):
        _fail('duplicate train pairs')
    n_mi, n_d = tm.shape
    if tp[:, 0].max() >= n_mi or tp[:, 1].max() >= n_d or tp.min() < 0:
        _fail('train pair out of bounds')
    if not np.isin(tm, (0, 1)).all():
        _fail('train_mask nonbinary')
    exp = np.zeros_like(tm, dtype=bool)
    exp[tp[:, 0], tp[:, 1]] = True
    if not (tm.astype(bool) == exp).all():
        _fail('train_mask != train_pairs cells')
    if yt.shape != (len(tp), n_types) or not np.isin(yt, (0, 1)).all():
        _fail('y_train invalid')
    if not (yt.sum(1) > 0).all():
        _fail('y_train rows need >=1 positive channel')
    if 'val_pairs' in flat or 'y_val' in flat:
        _fail('refit bundle must not carry validation')
    if 'test_pairs' in flat or 'y_test' in flat:
        _fail('held-out keys present')
    return True


def normalize_dtypes(flat):
    """M4: sealed-builder dtype parity — views float32, y_* float64.
    Mask int8 and pairs int64 are already correct from build_flat."""
    for k in list(flat):
        if k.startswith('view_'):
            flat[k] = flat[k].astype(np.float32)
        elif k.startswith('y_'):
            flat[k] = flat[k].astype(np.float64)
    return flat


# ---------- atomic publication ----------
def publish(staging_dir, out_dir, flat, meta, dis_names, audits):
    """Stage npz+manifest+recon sidecars then atomic rename; error ->
    staging removed. Sidecars are _resolve_recon-consumable:
    recon_meta.json seals the published npz's sha + entity_names.dis,
    recon_audits.json carries descriptor_alias regenerated from the
    pinned semantic run (B2)."""
    if os.path.exists(out_dir):
        _fail('occupied output namespace ' + out_dir)
    os.makedirs(staging_dir, exist_ok=False)
    try:
        npz_path = os.path.join(staging_dir,
                                'candidate_fold_data.npz')
        np.savez(npz_path, **flat)
        meta = dict(meta)
        meta['export_hashes'] = {
            k: _sha_arr(v) for k, v in flat.items()}
        with open(os.path.join(staging_dir,
                               'candidate_fold_data.meta.json'),
                  'w') as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(staging_dir, 'recon_meta.json'),
                  'w') as f:
            json.dump({'npz_sha256': _sha_f(npz_path),
                       'entity_names': {'dis': list(dis_names)}},
                      f, indent=1)
        with open(os.path.join(staging_dir, 'recon_audits.json'),
                  'w') as f:
            json.dump(audits, f, indent=1)
        os.rename(staging_dir, out_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# ---------- production entry ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--strict', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--phase', choices=PHASES, required=True)
    ap.add_argument('--mesh-xlsx', required=True)
    ap.add_argument('--meshtoid', required=True)
    ap.add_argument('--mtrees', required=True)
    ap.add_argument('--out', required=True)
    for p in ('candidate-csv-sha', 'candidate-manifest-sha',
              'strict-sha', 'semantic-sha', 'functional-sha',
              'contracts-sha', 'mesh-xlsx-sha', 'meshtoid-sha',
              'mtrees-sha'):
        ap.add_argument('--' + p, required=True)
    args = ap.parse_args()

    ccsv = os.path.join(args.candidate, 'pair_split_candidate.csv')
    cmf = os.path.join(args.candidate, 'candidate_manifest.json')
    # 1. external pins verified BEFORE consuming any data
    pins = {
        ccsv: _require_sha(ccsv, args.candidate_csv_sha, 'candidate csv'),
        cmf: _require_sha(cmf, args.candidate_manifest_sha,
                          'candidate manifest'),
        args.strict: _require_sha(args.strict, args.strict_sha,
                                 'strict npz')}
    deps = {}
    for d in DEP_FILES:
        p = os.path.join(_HERE, d)
        deps[p] = _require_sha(
            p, getattr(args, d.replace('.py', '_sha').replace('-', '_')),
            'dep ' + d)
    mesh = {}
    for flag, p in (('mesh_xlsx_sha', args.mesh_xlsx),
                    ('meshtoid_sha', args.meshtoid),
                    ('mtrees_sha', args.mtrees)):
        mesh[p] = _require_sha(p, getattr(args, flag), 'mesh ' + p)

    # 2. trusted manifest must SEAL candidate csv + strict npz
    mf = json.load(open(cmf))
    outs = mf.get('output_sha256', {})
    ins = mf.get('input_sha256', {})
    if outs.get('pair_split_candidate.csv') != pins[ccsv]:
        _fail('manifest does not seal candidate csv')
    if ins.get('dataset_strict') != pins[args.strict]:
        _fail('manifest does not seal strict npz')

    # 3. strict loaded AFTER pins via pin-before-deserialize loader
    strict = load_strict_pinned(args.strict, args.strict_sha)
    Y = strict['Y']
    n_m, n_d = Y.shape[0], Y.shape[1]
    expect = _stat_expect(mf, args.fold)
    recs = load_candidate_csv(ccsv, n_m, n_d, expect)
    _fold_counts_gate(recs, args.fold, expect)
    tr, va = select_fold_pairs(recs, args.fold, args.phase)

    # 4. build (imports AFTER dep pins verified)
    import semantic, functional
    ms = semantic.load_mesh_source(args.mesh_xlsx, args.meshtoid,
                                   args.mtrees)
    sem = semantic.semantic_similarity(list(strict['dis_names']), ms,
                                       multi_descriptor='expand')
    flat = build_flat(Y, tr, va, strict['sim_m1'], strict['sim_d1'],
                      lambda A: functional.functional_similarity(
                          A, sem).sim,
                      lambda: sem.S)
    flat = normalize_dtypes(flat)
    if args.phase == 'refit':
        prefit_gate(flat, int(Y.shape[2]))
    # B2: descriptor_alias regenerated from THIS run's sem object —
    # binds sidecar to the pinned semantic.py + MeSH inputs, not to a
    # stale fold-era audits.json. coverage False => uncovered row.
    uncovered = [{'row': int(i), 'name': strict['dis_names'][i]}
                 for i in range(n_d) if not sem.coverage[i]]
    audits = {'descriptor_alias': {
        'n_dis': n_d, 'n_uncovered_rows': len(uncovered),
        'uncovered_rows': uncovered,
        'generated_by': 'adapter sem.coverage (semantic.py pinned)',
        'provenance_note': 'descriptor-alias only; heldout_invariance/'
                           'train_sensitivity sections of any reused '
                           'fold-era audits.json do NOT apply'}}

    # 5. mutation gate — every pinned input re-verified post-build
    for p, h in list(pins.items()) + list(deps.items()) \
            + list(mesh.items()):
        if _sha_f(p) != h:
            _fail('input mutated during build: ' + p)

    stage = args.out + '.part'
    publish(stage, args.out, flat, {
        'stage': 'CANDIDATE_FOLD_BUNDLE_REV3', 'fold': args.fold,
        'phase': args.phase, 'candidate_dir': args.candidate,
        'pins': {os.path.basename(k): v for k, v in
                 list(pins.items()) + list(deps.items())
                 + list(mesh.items())},
        'n_train': len(tr), 'n_val': (len(va) if va else 0),
        'dtypes': 'views float32, y_* float64, mask int8, pairs int64 '
                  '(sealed-builder parity)',
        'heldout_absence_proof':
            'exported pairs set-equal pinned candidate non-test '
            'segments; outer-test Y never read',
        'refit_consumer':
            ('u1_refit_noval_rev3.load_refit_npz + train_refit_noval'
             if args.phase == 'refit'
             else 'runner.load_data_npz + preflight')},
            strict['dis_names'], audits)


if __name__ == '__main__':
    main()
