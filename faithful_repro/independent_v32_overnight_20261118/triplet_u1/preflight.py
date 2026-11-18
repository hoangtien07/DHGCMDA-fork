# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 preflight — pin checks + fixtures + contracts.

Gates (fail-closed): protocol hash, all namespace source hashes,
strict npz + bundle + sidecar hashes per fold, fixtures all PASS,
held-out label canary re-verified on REAL bundles, contract cfg +
expected manifest + pinned neg-supervision index written per (fold,seed).
Read-only on every artifact outside this namespace.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric

HERE = Path(__file__).resolve().parent
OVER = HERE.parent
ROOT = OVER.parents[1]
PKG = ROOT / 'faithful_repro' / 'colab_package_f0_selection_20261115'
STRICT = PKG / 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
PKG_MF = PKG / 'package_manifest.json'
SRC = ROOT / 'faithful_repro' / 'paper_guided' / \
    'u1_horizon650_fold0_rev1' / 'source'
for _p in (str(SRC), str(SRC / 'stages'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import split as S                       # noqa: E402
import runner as s1r                    # noqa: E402
import fixtures                         # noqa: E402
import train_u1x                        # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
_arr = lambda a: hashlib.sha256(
    np.ascontiguousarray(a).tobytes()).hexdigest()


def write(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=1, allow_nan=False)
        f.write('\n')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.__version__.split('+')[0] == '2.5.1' \
        and torch_geometric.__version__ == '2.6.1' \
        and np.__version__ == '1.26.4' \
        and not torch.cuda.is_available()
    fx = fixtures.run_all()

    src_files = ['split.py', 'fixtures.py', 'model_u1x.py',
                 'losses_u1x.py', 'train_u1x.py', 'eval_cvtriplet.py',
                 'baseline_logistic.py', 'run_eval.py', 'train.py',
                 'build_bundles.py', 'preflight.py', 'verify.py',
                 'run_5x3.py']
    src_hashes = {n: sha(HERE / n) for n in src_files}

    mf = read(PKG_MF)['files']
    strict_sha = mf['inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz']
    assert sha(STRICT) == strict_sha
    z = np.load(STRICT, allow_pickle=True)
    Y = np.asarray(z['Y']).astype(bool)

    base_cfg = read(OVER / 'bank1_linux' / 'contracts' /
                    'fold0_seed1234' / 'cfg.json')
    # production cfg = U1 unchanged + existence keys; cap10 (negative
    # verdict on cap1) — pin, no sweep.
    base_cfg['inter_cap_mult'] = 10
    base_cfg['stage'] = 'U1X'
    base_cfg['revision'] = 'REV2-cvtriplet-existence'
    base_cfg['lam_exist'] = 0.3
    notes = base_cfg.setdefault('notes', {})
    notes['u1x'] = ('U1 objective unchanged; + lam_exist*focal existence'
                    ' on mask cells + pinned 10x mask-zero subsample; '
                    'sanity fold0 only, not a result')

    entries = []
    (HERE / 'contracts').mkdir(exist_ok=True)
    for fold in range(5):
        bdir = HERE / 'bundles' / f'fold{fold}'
        rep = read(HERE / 'build_report.json')
        erow = next(e for e in rep['entries'] if e['fold'] == fold)
        assert sha(bdir / 'candidate_fold_data.npz') == \
            erow['bundle_sha256']
        sc = read(bdir / 'split_sidecar.json')
        data = s1r.load_data_npz(str(bdir / 'candidate_fold_data.npz'))
        # canary on REAL bundle: no held-out bit set in y_train/y_val
        seg = S.fold_segments(Y, fold)
        held = np.concatenate([seg['inner_val'], seg['test']])
        tp = data['train_pairs']
        ytr = np.asarray(data['y_train'])
        row_of = {(i, j): r for r, (i, j) in
                  enumerate(tp.tolist())}
        leaked = 0
        for (i, j, k) in held.tolist():
            if (i, j) in row_of and ytr[row_of[(i, j)], k]:
                leaked += 1
        assert leaked == 0, ('held-out label leaked', fold, leaked)
        vp = data['val_pairs']
        vrow = {(i, j): r for r, (i, j) in enumerate(vp.tolist())}
        yv = np.asarray(data['y_val'])
        leaked_v = sum(1 for (i, j, k) in
                       np.concatenate([seg['inner_train'],
                                       seg['test']]).tolist()
                       if (i, j) in vrow and yv[vrow[(i, j)], k])
        assert leaked_v == 0
        for seed in (1234, 42, 2025):
            cfg = dict(base_cfg)
            cfg['seed'] = cfg['inter_seed'] = seed
            neg_idx, neg_sha = S.neg_supervision_index(
                data['train_mask'], len(tp), seed)
            exp = train_u1x.expected_manifest_for_u1x(
                cfg, data,
                meta_path=str(bdir / 'recon_meta.json'),
                audit_path=str(bdir / 'recon_audits.json'),
                data_file_sha=erow['bundle_sha256'],
                neg_exist_idx=neg_idx, neg_exist_sha=neg_sha)
            key = f'fold{fold}_seed{seed}'
            c = HERE / 'contracts' / key
            c.mkdir(exist_ok=True)
            write(c / 'cfg.json', cfg)
            write(c / 'expected.json', exp)
            np.save(c / 'neg_exist_idx.npy', neg_idx)
            entries.append({
                'key': key, 'fold': fold, 'seed': seed,
                'bundle': str(bdir.resolve()),
                'bundle_sha256': erow['bundle_sha256'],
                'neg_exist_sha256': neg_sha,
                'n_neg_exist': int(len(neg_idx)),
                'n_train_pairs': sc['n_train_pairs'],
                'n_val_pairs': sc['n_val_pairs'],
                'straddle_innerval_dropped':
                    sc['straddle_innerval_dropped'],
                'test_pos_pair_seen': sc['test_pos_pair_seen'],
                'test_pos_pair_unseen': sc['test_pos_pair_unseen'],
                'cfg_sha256': sha(c / 'cfg.json'),
                'expected_sha256': sha(c / 'expected.json')})
    out = {'status': 'PASS',
           'protocol_sha256': sha(OVER / 'PROTOCOL_CVTRIPLET_U1_REV2.md'),
           'design_review_sha256': sha(OVER / 'DESIGN_REVIEW.md'),
           'source_hashes': src_hashes,
           'fixtures': fx,
           'strict_sha256': strict_sha,
           'entries': entries,
           'preflight_script_sha256': sha(__file__)}
    write(HERE / 'preflight.json', out)
    print('PREFLIGHT PASS', len(entries), 'entries;', len(fx),
          'fixtures', flush=True)


if __name__ == '__main__':
    main()
