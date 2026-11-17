# -*- coding: utf-8 -*-
"""READ-ONLY real preflight REV3. ALL pins REQUIRED as full 64-hex via
--pins JSON: {name: {path, sha256}} for dataset_strict, candidate_csv,
candidate_manifest, each MeSH input, plus 'source_map' ({file: sha})
and 'expected_stat_gates' (optional). Verifies byte shas, manifest
output/input seals, stat_gates schema+types, fold counts/group
disjointness recomputed from the candidate csv alone, executed source
map == pinned map AND every module's resolved __file__ re-hashes.
Exit 0 only if every check passes; prints PREFLIGHT READY only then.
Read-only: no builds/labels/models/mutation."""
import csv, hashlib, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

def fail(msg, errs):
    errs.append(msg)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pins', required=True,
                    help='json: {name:{path,sha256}, "source_map":{..}}')
    a = ap.parse_args(argv)
    errs = []
    try:
        pins = json.load(open(a.pins, encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        print('refusal: unreadable pins: %s' % e)
        return 1
    if not isinstance(pins, dict):
        print('refusal: pins not a json object'); return 1
    req_files = ['dataset_strict', 'candidate_csv',
                 'candidate_manifest', 'mesh_hmdd_xlsx',
                 'mesh_toid_csv', 'mesh_mtrees_bin']
    files = {k: v for k, v in pins.items()
             if isinstance(v, dict) and 'path' in v}
    for k in req_files:
        if k not in files:
            fail('pins missing required file %s' % k, errs)
    shas = {}
    for k, v in files.items():
        w = v.get('sha256')
        if not (isinstance(w, str) and len(w) == 64
                and w == w.lower()
                and all(c in '0123456789abcdef' for c in w)):
            fail('%s pin missing/invalid 64-hex' % k, errs)
            continue
        p = v['path']
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        if not os.path.exists(p):
            fail('%s missing: %s' % (k, p), errs)
            continue
        got = sha(p)
        if got != w:
            fail('%s sha mismatch %s != %s' % (k, got, w), errs)
        shas[k] = got
    # manifest seals
    if 'candidate_manifest' in shas and 'candidate_csv' in shas:
        mp = files['candidate_manifest']['path']
        if not os.path.isabs(mp):
            mp = os.path.join(ROOT, mp)
        try:
            cmf = json.load(open(mp, encoding='utf-8'))
            outs = cmf.get('output_sha256', {})
            ins = cmf.get('input_sha256', {})
            if outs.get('pair_split_candidate.csv') \
                    != shas['candidate_csv']:
                fail('manifest output seal != candidate csv sha',
                     errs)
            if 'dataset_strict' in shas and ins.get(
                    'dataset_strict') != shas['dataset_strict']:
                fail('manifest input seal != dataset_strict sha',
                     errs)
        except (OSError, json.JSONDecodeError) as e:
            fail('manifest unreadable: %s' % e, errs)
            cmf = None
    else:
        cmf = None
    # candidate csv counts + group atomicity recomputed
    if 'candidate_csv' in shas and isinstance(cmf, dict):
        cp = files['candidate_csv']['path']
        if not os.path.isabs(cp):
            cp = os.path.join(ROOT, cp)
        try:
            rows = list(csv.DictReader(open(cp, encoding='utf-8')))
            seg = {}
            grp_seg = {}
            for r in rows:
                g = r['group_key']
                for f in range(5):
                    s = r['seg_fold%d' % f]
                    if s not in ('test', 'inner_train',
                                 'inner_val'):
                        fail('unknown seg label %r' % s, errs)
                    seg.setdefault((f, s), 0)
                    seg[(f, s)] += 1
                    grp_seg.setdefault((g, f), set()).add(s)
            seen_pairs = set()
            for r in rows:
                pr = (r['mir_idx'], r['dis_idx'])
                if pr in seen_pairs:
                    fail('duplicate csv record %s' % (pr,), errs)
                    break
                seen_pairs.add(pr)
                if not r['group_key'].strip():
                    fail('empty group_key', errs); break
            for (g, f), ss in grp_seg.items():
                if len(ss) > 1:
                    fail('group %s straddles segments fold %d'
                         % (g, f), errs)
                    break
            if cmf.get('all_gates_pass') is not True:
                fail('manifest all_gates_pass is not true', errs)
            bg = cmf.get('bool_gates')
            if not isinstance(bg, dict) or not bg:
                fail('manifest missing nonempty bool_gates', errs)
            else:
                for k, v in bg.items():
                    if v is not True:
                        fail('bool_gates.%s != true' % k, errs)
            sg = cmf.get('stat_gates')
            if not isinstance(sg, dict):
                fail('manifest missing stat_gates', errs)
                sg = {}
            for k in ('positive_universe', 'pair_groups_total'):
                v = sg.get(k)
                if type(v) is not int or isinstance(v, bool) \
                        or v <= 0:
                    fail('stat_gates.%s missing/not positive int'
                         % k, errs)
            if type(sg.get('positive_universe')) is int \
                    and sg['positive_universe'] != len(rows):
                fail('manifest universe %d != csv rows %d'
                     % (sg['positive_universe'], len(rows)), errs)
            n_grp = len({r['group_key'] for r in rows})
            if type(sg.get('pair_groups_total')) is int \
                    and sg['pair_groups_total'] != n_grp:
                fail('pair_groups_total %d != csv groups %d'
                     % (sg['pair_groups_total'], n_grp),
                     errs)
            ig = sg.get('inner_groups')
            if not isinstance(ig, dict):
                fail('stat_gates.inner_groups missing', errs)
                ig = {}
            for f in range(5):
                want = sg.get('fold%d_test_cells' % f)
                if type(want) is not int or isinstance(want, bool) \
                        or want <= 0:
                    fail('fold%d_test_cells bad' % f, errs)
                elif want != seg.get((f, 'test'), 0):
                    fail('fold%d test cells %d != csv %d'
                         % (f, want, seg.get((f, 'test'), 0)), errs)
                gi = ig.get(str(f))
                if not isinstance(gi, dict):
                    fail('inner_groups[%d] missing' % f, errs)
                    continue
                gfv = {}
                for r in rows:
                    gfv.setdefault(r['group_key'],
                                   r['seg_fold%d' % f])
                vals = list(gfv.values())
                tr_g = sum(1 for v in vals
                           if v in ('inner_train', 'inner_val'))
                iv_g = vals.count('inner_val')
                for kk, want_n in (('train_groups', tr_g),
                                   ('inner_val_groups', iv_g)):
                    v2 = gi.get(kk)
                    if type(v2) is not int or isinstance(v2, bool):
                        fail('inner_groups[%d].%s bad' % (f, kk),
                             errs)
                    elif v2 != want_n:
                        fail('fold%d %s %d != csv %d'
                             % (f, kk, v2, want_n), errs)
        except (OSError, KeyError) as e:
            fail('candidate csv unreadable: %s' % e, errs)
    # executed source map == pinned map, __file__ origins re-hash
    smap = pins.get('source_map')
    if not isinstance(smap, dict) or not smap:
        fail('pins missing nonempty source_map', errs)
    else:
        sys.path.insert(0, os.path.join(_HERE, 'stages'))
        sys.path.insert(0, _HERE)
        try:
            import u1_refit_noval_rev10 as R
            cur = R.effective_source_map()
            if set(cur) != set(smap):
                fail('source map key-set differs (missing %s extra %s)'
                     % (sorted(set(smap) - set(cur)),
                        sorted(set(cur) - set(smap))), errs)
            else:
                for k, want in smap.items():
                    if cur.get(k) != want:
                        fail('source pin mismatch %s' % k, errs)
                        break
                else:
                    # actual __file__ origin resolution: map key ->
                    # candidate paths under paper_guided[/stages],
                    # re-hash the file it resolves to
                    bad = []
                    for k, want in smap.items():
                        rel = k.split('/')[-1]
                        cands = [os.path.join(_HERE, 'stages', rel),
                                 os.path.join(_HERE, rel)]
                        hit = next((c for c in cands
                                    if os.path.exists(c)), None)
                        if hit is None:
                            bad.append('origin-missing ' + k)
                        elif sha(hit) != want:
                            bad.append('origin-hash ' + k)
                    for b in bad[:3]:
                        fail('module origin: ' + b, errs)
        except Exception as e:
            fail('source map generation failed: %s' % e, errs)
    if errs:
        for e in errs:
            print('REFUSE: ' + e)
        print('PREFLIGHT FAILED (%d refusals)' % len(errs))
        return 1
    print('PREFLIGHT READY')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
