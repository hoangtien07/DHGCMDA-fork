# -*- coding: utf-8 -*-
"""Pre-selection contract + sealed-train_u1 wrapper. Hashes everything
BEFORE the run (bundle, metas, cfg, exact effective source map incl.
imported_pure_hashes, __file__ origins), emits an immutable no-clobber
PRESELECTION_CONTRACT, runs sealed train_u1, re-verifies sources
post-run, then emits SUCCESS_RECORD binding contract+manifest+bundle
shas. Exit 0 ok / 3 refusal / 4 aborted."""
import argparse, hashlib, importlib, json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import u1_runner as u1r


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _fail(m):
    raise ValueError(m)


def _srcmap():
    """Exact effective map incl. parent imported modules — key-set
    equality to the sealed U1 contract's own maps."""
    m = dict(u1r.imported_pure_hashes_u1())
    m.update(u1r.source_hashes_u1())
    return m


def _origins(map_keys):
    """Resolve each pinned module name to its loaded __file__ and
    re-hash — wrong/swapped origins refuse."""
    out = {}
    roots = [os.path.dirname(_HERE), _HERE]
    for k in map_keys:
        rel = k.replace('stages/', '').replace('paper_guided/',
                                               '')
        cand = [os.path.join(r, rel) for r in roots]
        hit = next((c for c in cand if os.path.exists(c)), None)
        if hit is None:
            _fail('no file for pinned module ' + k)
        out[k] = _sha(hit)
    return out


def _ajson(o, p):
    if os.path.exists(p):
        _fail('refuse clobber ' + p)
    part = p + '.part'
    json.dump(o, open(part, 'w'), indent=1, sort_keys=True,
              default=str)
    os.replace(part, p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in ('bundle', 'meta', 'audit', 'cfg'):
        ap.add_argument('--' + f, required=True)
        ap.add_argument('--' + f + '-sha', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    try:
        for f in ('bundle', 'meta', 'audit', 'cfg'):
            if _sha(getattr(args, f)) != getattr(args, f + '_sha'):
                _fail(f + ' sha mismatch')
        cfg = json.load(open(args.cfg))
        smap = _srcmap()
        origins = _origins(smap.keys())
        if origins != smap:
            _fail('loaded module origin != pinned source map: '
                  + repr({k: (origins[k], smap[k]) for k in smap
                          if origins[k] != smap[k]}))
        data = {}
        with np.load(args.bundle, allow_pickle=False) as z:
            for k in z.files:
                data[k] = z[k]
        views = {k[5:]: data.pop(k) for k in list(data)
                 if k.startswith('view_')}
        data['views'] = views
        pcp = args.out + '.precontract.json'
        _ajson({'stage': 'PRESELECTION_CONTRACT',
                'sel_bundle_sha256': args.bundle_sha,
                'sel_meta_sha256': args.meta_sha,
                'sel_audit_sha256': args.audit_sha,
                'sel_cfg_sha256': hashlib.sha256(json.dumps(
                    cfg, sort_keys=True, default=str).encode())
                .hexdigest(),
                'sel_source_map': smap}, pcp)
        pcsha = _sha(pcp)
        exp = u1r.expected_manifest_for_u1(cfg, data, args.meta,
                                           args.audit,
                                           args.bundle_sha)
        res = u1r.train_u1(cfg, data, args.out, log=lambda *a: 0,
                           expected_manifest=exp,
                           meta_path=args.meta,
                           audit_path=args.audit,
                           data_file_sha=args.bundle_sha)
        for k, v in _srcmap().items():
            if smap.get(k) != v:
                _fail('post-run source drift: ' + k)
        mfp = os.path.join(args.out, 'manifest.json')
        mf = json.load(open(mfp))
        _ajson({'stage': 'SELECTION_SUCCESS_RECORD',
                'precontract_sha256': pcsha,
                'selection_manifest_sha256': _sha(mfp),
                'sel_bundle_sha256': args.bundle_sha,
                'aborted': mf.get('aborted'),
                'best_epoch': mf['best']['epoch']},
               args.out + '.success_record.json')
        return 4 if mf.get('aborted') else 0
    except (ValueError, TypeError, OSError, KeyError,
            json.JSONDecodeError) as e:
        print('refusal: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
