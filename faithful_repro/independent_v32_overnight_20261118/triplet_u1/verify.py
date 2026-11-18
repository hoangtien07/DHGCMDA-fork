# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 artifact verification (AMENDMENT B3).

Independent re-checks that do NOT trust the run: contract hashes
unchanged, manifest PASS + 650 epochs, eval negatives all Y==0 and
disjoint from test, and every AUROC/AP/F1@0.5 in eval_full.json
recomputed from eval_arrays.npz (indices/truth/scores) with a fresh
sklearn call. Writes verification_full.json. Synthetic metric fixtures
are unit-level only and are NOT counted here."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVER = HERE.parent
PKG = OVER.parents[1] / 'faithful_repro' / \
    'colab_package_f0_selection_20261115'
STRICT = PKG / 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
sys.path.insert(0, str(HERE))
import split as S                       # noqa: E402
from eval_cvtriplet import binary_metrics  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2)
        f.write('\n')


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main(key):
    checks = []

    def ok(name, cond, detail=''):
        checks.append({'name': name, 'pass': bool(cond),
                       'detail': str(detail)})
        if not cond:
            raise ValueError('verify fail ' + name)

    pf = read(HERE / 'preflight.json')
    e = next(x for x in pf['entries'] if x['key'] == key)
    fold = e['fold']
    run = HERE / 'runs' / key
    comp = read(run / 'completion.json')
    ok('completion PASS', comp['status'] == 'PASS')
    ok('manifest sha stable',
       sha(run / 'out' / 'manifest.json') == comp['manifest_sha256'])
    man = read(run / 'out' / 'manifest.json')
    ok('final_epoch 650', man['final_epoch'] == 650)
    ok('not aborted', man['aborted'] is None)
    ok('exist lam pinned', man['exist_supervision']['lam_exist'] == 0.3)
    ok('trajectory finite', all(
        np.isfinite(r['train_loss']) for r in man['trajectory']))
    ev = read(run / 'eval_full.json')
    ok('eval PASS', ev['status'] == 'PASS')
    arr = np.load(run / 'eval_arrays.npz')
    ok('arrays hash present', len(arr.files) >= 35, len(arr.files))

    mf = read(PKG / 'package_manifest.json')['files']
    strict_sha = mf[
        'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz']
    ok('strict pin', sha(STRICT) == strict_sha)
    z = np.load(STRICT, allow_pickle=True)
    Y = np.asarray(z['Y']).astype(bool)
    seg = S.fold_segments(Y, fold)
    test_set = S.triplet_set(seg['test'])
    mask = S.pair_mask(*Y.shape[:2],
                       S.pairs_of(seg['inner_train']))
    unk = S.unknown_universe(Y)

    # --- artifact-level metric recompute: Task T ---
    for rep_i, row in enumerate(ev['task_T']['rows']):
        idx = arr[f'T_rep{rep_i}_idx']
        y = arr[f'T_rep{rep_i}_y']
        ok(f'T rep{rep_i} idx==pos+test-neg',
           len(idx) == len(y) and y.sum() == len(seg['test']))
        negs = idx[y == 0]
        poss = idx[y == 1]
        ok(f'T rep{rep_i} negs all Y==0',
           not Y[negs[:, 0], negs[:, 1], negs[:, 2]].any())
        ok(f'T rep{rep_i} negs disjoint test',
           not S.triplet_set(negs) & test_set)
        ok(f'T rep{rep_i} poss == test set',
           S.triplet_set(poss) == test_set)
        m = binary_metrics(y, arr[f'T_rep{rep_i}_s_u1x'])
        ok(f'T rep{rep_i} auroc recompute',
           close(m['auroc'], row['auroc']), (m['auroc'], row['auroc']))
        ok(f'T rep{rep_i} ap recompute',
           close(m['average_precision'], row['average_precision']))
        ok(f'T rep{rep_i} f1 recompute',
           close(m['f1_at_0_5'], row['f1_at_0_5']))
        codes = arr[f'T_rep{rep_i}_neg_codes'][y == 0]
        ok(f'T rep{rep_i} strata sum',
           int((codes == 0).sum() + (codes == 1).sum()
               + (codes == 2).sum()) == len(negs))
        for pool in ('mask_zero_pairs', 'all_unknown'):
            bl_row = ev['baseline_logistic'][pool]['rows'][rep_i]
            m_b = binary_metrics(
                y, arr[f'T_rep{rep_i}_s_bl_{pool}'])
            ok(f'T rep{rep_i} bl[{pool}] auroc recompute',
               close(m_b['auroc'], bl_row['auroc']))
    # --- Task P ---
    pz_set = set(map(tuple, np.argwhere(~Y.any(axis=2)).tolist()))
    for rep_i, row in enumerate(ev['task_P']['rows']):
        idx = arr[f'P_rep{rep_i}_idx']
        y = arr[f'P_rep{rep_i}_y']
        negs = idx[y == 0]
        poss = idx[y == 1]
        ok(f'P rep{rep_i} negs pair-zero',
           all(tuple(t) in pz_set for t in negs.tolist()))
        ok(f'P rep{rep_i} poss unseen-pair',
           all(Y[i, j].any() and not mask[i, j]
               for i, j in poss.tolist()))
        m = binary_metrics(y, arr[f'P_rep{rep_i}_s_u1x'])
        ok(f'P rep{rep_i} auroc recompute',
           close(m['auroc'], row['auroc']))
    # --- canary on the real bundle ---
    bdir = HERE / 'bundles' / f'fold{fold}'
    dd = np.load(bdir / 'candidate_fold_data.npz')
    tp, ytr = dd['train_pairs'], dd['y_train']
    row_of = {(i, j): r for r, (i, j) in enumerate(tp.tolist())}
    held = np.concatenate([seg['inner_val'], seg['test']])
    leaked = sum(1 for (i, j, k) in held.tolist()
                 if (i, j) in row_of and ytr[row_of[(i, j)], k])
    ok('canary train', leaked == 0, str(leaked))
    vp, yv = dd['val_pairs'], dd['y_val']
    vrow = {(i, j): r for r, (i, j) in enumerate(vp.tolist())}
    leaked_v = sum(1 for (i, j, k) in
                   np.concatenate([seg['inner_train'],
                                   seg['test']]).tolist()
                   if (i, j) in vrow and yv[vrow[(i, j)], k])
    ok('canary val', leaked_v == 0, str(leaked_v))

    report = {'status': 'PASS', 'checks': checks, 'key': key,
              'n_checks': len(checks)}
    write(run / 'verification_full.json', report)
    print('VERIFY PASS', len(checks), 'checks', key)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='fold0_seed1234')
    main(ap.parse_args().key)
