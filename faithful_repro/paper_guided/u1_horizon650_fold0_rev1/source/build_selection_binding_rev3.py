# -*- coding: utf-8 -*-
"""Binding producer: recomputes selection->refit provenance from pinned
artifacts — NO attestation strings. Verifies exact external SHA before
reading every input; derives fold membership from the pinned candidate
CSV (grouped by pair-group, atomic); compares the selection bundle's
train/val pair arrays + masks and its manifest data_hashes against the
candidate; compares the refit bundle's train pairs to the candidate
non-test union; proves outer-test disjointness; and re-verifies every
pin post-work. Emits an atomic, no-clobber hash-pinned binding.json
consumable by u1_refit_noval_rev4 (--selection-binding).

Outer-test labels are NEVER read: only pair-index arrays and masks are
compared; no y_* array is loaded for test pairs (refit bundle has none
by construction; selection y_* is read only to recompute the
manifest's own declared data_hashes).

Exit 0 emit / 3 refusal."""
import argparse, hashlib, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import build_candidate_fold_features_rev3 as A


def _fail(m):
    raise ValueError(m)


def _pj(path, sha, what):
    A._require_sha(path, sha, what)
    return json.load(open(path, encoding='utf-8'))


def _pairs_set(a):
    a = np.asarray(a)
    return set(map(tuple, a.tolist()))


def _sorted_pairs_sha(s):
    arr = np.asarray(sorted(s), dtype=np.int64)
    return A._sha_arr(arr)


def _npz_keys(path, sha, keys):
    A._require_sha(path, sha, 'bundle npz')
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f, h in [('candidate_csv', 'candidate csv'),
                 ('candidate_manifest', 'candidate manifest'),
                 ('sel_bundle', 'selection bundle npz'),
                 ('sel_meta', 'selection bundle meta json'),
                 ('sel_manifest', 'selection-run manifest json'),
                 ('refit_bundle', 'refit bundle npz'),
                 ('refit_meta', 'refit bundle meta json')]:
        ap.add_argument('--' + f.replace('_', '-'))
        ap.add_argument('--' + f.replace('_', '-') + '-sha')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    try:
        for f in ('candidate_csv', 'candidate_manifest', 'sel_bundle',
                  'sel_meta', 'sel_manifest', 'refit_bundle',
                  'refit_meta'):
            if getattr(args, f) is None or getattr(
                    args, f + '_sha') is None:
                _fail('--%s and its sha are required'
                      % f.replace('_', '-'))
        if os.path.exists(args.out):
            _fail('binding target exists — no clobber: ' + args.out)
        fold = args.fold

        cmf = _pj(args.candidate_manifest, args.candidate_manifest_sha,
                  'candidate manifest')
        outs = cmf.get('output_sha256', {})
        if outs.get('pair_split_candidate.csv') \
                != args.candidate_csv_sha:
            _fail('candidate manifest does not seal candidate csv')
        smf = _pj(args.sel_manifest, args.sel_manifest_sha,
                  'selection manifest')
        if smf.get('stage') != 'U1':
            _fail('selection manifest stage must be U1')
        if smf.get('aborted') is not None:
            _fail('selection run aborted')
        best = smf.get('best')
        if not isinstance(best, dict) or type(best.get('epoch')) \
                is not int or best['epoch'] <= 0:
            _fail('selection manifest missing positive best.epoch')
        # R1: epoch schema REQUIRED unconditionally — absent or
        # non-int fields refuse; bound reflects U1 semantics
        # (best <= final <= max_epochs, final may stop early).
        fe = smf.get('final_epoch')
        if type(fe) is not int or isinstance(fe, bool) or fe <= 0:
            _fail('selection manifest missing positive-int '
                  'final_epoch')
        sc = smf.get('cfg')
        if not isinstance(sc, dict) or type(sc.get('max_epochs'))                 is not int or isinstance(sc['max_epochs'], bool):
            _fail('selection cfg missing int max_epochs')
        if not (best['epoch'] <= fe <= sc['max_epochs']):
            _fail('impossible horizon: best.epoch %d > final_epoch '
                  '%d > max_epochs %d'
                  % (best['epoch'], fe, sc['max_epochs']))
        dh = smf.get('data_hashes')
        if not isinstance(dh, dict):
            _fail('selection manifest missing data_hashes')

        for mp, msha, phase in ((args.sel_meta, args.sel_meta_sha,
                                 'selection'),
                                (args.refit_meta, args.refit_meta_sha,
                                 'refit')):
            m = _pj(mp, msha, phase + ' bundle meta')
            if m.get('stage') != 'CANDIDATE_FOLD_BUNDLE_REV3':
                _fail('%s bundle meta stage wrong' % phase)
            if m.get('phase') != phase:
                _fail('%s bundle meta phase wrong' % phase)
            if m.get('fold') != fold:
                _fail('%s bundle meta fold != --fold' % phase)
            if m['pins']['pair_split_candidate.csv'] \
                    != args.candidate_csv_sha:
                _fail('%s bundle meta candidate pin mismatch' % phase)

        sel = _npz_keys(args.sel_bundle, args.sel_bundle_sha,
                        ['train_pairs', 'val_pairs', 'train_mask',
                         'y_train', 'y_val',
                         'view_mi_v1', 'view_mi_v2',
                         'view_dis_v1', 'view_dis_v2'])
        # S1: REV3 adapter emits NO val_mask — derive val cells from
        # pinned val_pairs (never invent serialized fields).
        sel['val_mask'] = np.zeros_like(sel['train_mask'])
        for i, j in np.asarray(sel['val_pairs']).tolist():
            sel['val_mask'][i, j] = 1
        ref = _npz_keys(args.refit_bundle, args.refit_bundle_sha,
                        ['train_pairs', 'train_mask'])
        n_m, n_d = sel['train_mask'].shape
        expect = A._stat_expect(cmf, fold)
        recs = A.load_candidate_csv(args.candidate_csv, n_m, n_d,
                                    expect)
        A._fold_counts_gate(recs, fold, expect)
        tr_c, iv_c = A.select_fold_pairs(recs, fold, 'selection')
        ru_c, _ = A.select_fold_pairs(recs, fold, 'refit')
        te = {(r['mir'], r['dis']) for r in recs
              if r['seg'][fold] == 'test'}

        # 1. selection bundle pair arrays == candidate fold membership
        if _pairs_set(sel['train_pairs']) != set(tr_c):
            _fail('selection train_pairs != candidate inner_train')
        if _pairs_set(sel['val_pairs']) != set(iv_c):
            _fail('selection val_pairs != candidate inner_val')
        # 1b. S7: selection masks must be binary before hashing
        for nm, m in (('train', sel['train_mask']),
                      ('val', sel['val_mask'])):
            if not np.isin(np.asarray(m), (0, 1)).all():
                _fail('selection %s_mask not binary' % nm)
        # 2. masks exactly the pair cells
        if set(map(tuple, np.argwhere(
                sel['train_mask'].astype(bool)).tolist())) != set(tr_c):
            _fail('selection train_mask != candidate inner_train')
        if set(map(tuple, np.argwhere(
                sel['val_mask'].astype(bool)).tolist())) != set(iv_c):
            _fail('selection val_mask != candidate inner_val')
        # 3. refit train pairs == non-test union, mask consistent
        if _pairs_set(ref['train_pairs']) != set(ru_c):
            _fail('refit train_pairs != candidate non-test union')
        if set(map(tuple, np.argwhere(
                ref['train_mask'].astype(bool)).tolist())) \
                != set(ru_c):
            _fail('refit train_mask != candidate non-test union')
        # 4. test disjointness (labels never touched)
        for nm, ps in (('selection', set(tr_c) | set(iv_c)),
                       ('refit', set(ru_c))):
            if ps & te:
                _fail('%s bundle overlaps outer-test pairs' % nm)
        # 5. selection manifest data_hashes == recomputed (proves the
        #    selection RUN consumed these exact arrays)
        recomputed = {
            'train_mask': A._sha_arr(
                sel['train_mask'].astype(bool)),
            'train_pairs': A._sha_arr(np.asarray(sel['train_pairs'])),
            'val_pairs': A._sha_arr(np.asarray(sel['val_pairs'])),
            'y_train': A._sha_arr(np.asarray(sel['y_train'])),
            'y_val': A._sha_arr(np.asarray(sel['y_val']))}
        matched = []
        for k, v in dh.items():
            if k in recomputed:
                if v != recomputed[k]:
                    _fail('manifest data_hashes.%s mismatch — selection '
                          'did not consume this bundle' % k)
                matched.append(k)
        need = {'train_mask', 'train_pairs', 'val_pairs',
                'y_train', 'y_val'}
        if not need <= set(matched):
            _fail('manifest data_hashes missing/unsigned sealed keys '
                  '%s' % sorted(need - set(matched)))
        vh = smf.get('view_hashes')
        if not isinstance(vh, dict):
            _fail('selection manifest missing view_hashes')
        for sk in ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2'):
            want = vh.get(sk)
            if want is None:
                _fail('manifest view_hashes missing ' + sk)
            if want != A._sha_arr(np.asarray(sel['view_' + sk])):
                _fail('manifest view_hashes.%s mismatch — selection '
                      'consumed different features' % sk)
            matched.append('view_' + sk)

        pins_all = {'candidate_csv': args.candidate_csv_sha,
                    'candidate_manifest': args.candidate_manifest_sha,
                    'sel_bundle': args.sel_bundle_sha,
                    'sel_meta': args.sel_meta_sha,
                    'sel_manifest': args.sel_manifest_sha,
                    'refit_bundle': args.refit_bundle_sha,
                    'refit_meta': args.refit_meta_sha}
        # post-work re-verify of every pin
        for f, s in pins_all.items():
            A._require_sha(getattr(args, f), s, f)

        binding = {
            'stage': 'SELECTION_BINDING_REV1',
            'selection_manifest_sha256': args.sel_manifest_sha,
            'fold': fold, 'phase': 'selection',
            'candidate_csv_sha256': args.candidate_csv_sha,
            'candidate_manifest_sha256':
                args.candidate_manifest_sha,
            'sel_bundle_sha256': args.sel_bundle_sha,
            'sel_meta_sha256': args.sel_meta_sha,
            'refit_bundle_sha256': args.refit_bundle_sha,
            'refit_meta_sha256': args.refit_meta_sha,
            'selected_epoch': best['epoch'],
            'counts': {'inner_train': len(tr_c), 'inner_val': len(iv_c),
                       'test': len(te), 'refit_union': len(ru_c)},
            'digests': {
                'sel_train_pairs_sorted': _sorted_pairs_sha(tr_c),
                'sel_val_pairs_sorted': _sorted_pairs_sha(iv_c),
                'refit_train_pairs_sorted': _sorted_pairs_sha(ru_c),
                'sel_train_mask': A._sha_arr(
                    sel['train_mask'].astype(bool)),
                'sel_val_mask': A._sha_arr(
                    sel['val_mask'].astype(bool)),
                'refit_train_mask': A._sha_arr(
                    ref['train_mask'].astype(bool))},
            'manifest_data_hashes_match': matched,
            'outer_test_disjoint': True,
            'proof': 'recomputed membership + digests from pinned '
                     'candidate csv; attestation-free'}
        part = args.out + '.part'
        with open(part, 'w', encoding='utf-8') as f:
            json.dump(binding, f, indent=1, sort_keys=True)
        os.replace(part, args.out)
        return 0
    except (ValueError, TypeError, OSError, KeyError, IndexError,
            json.JSONDecodeError) as e:
        print('refusal: %s' % e, file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main() or 0)
