# -*- coding: utf-8 -*-
"""Build 5 triplet-split bundles for CVtriplet-U1 REV2.

Same view recipe as the sealed pair-split builder
(build_candidate_fold_features_rev3.py): [S_seq|A], [S_func(A)|A],
[S_gene|A^T], [S_sem|A^T] with A = train_mask over inner_train pairs.
S_func rebuilt train-only; S_sem from the pinned MeSH-2019 modules.
Split logic is triplet-level per split.py (NEW — the pinned builder's
pair-segment CSV cannot express straddled pairs). All external inputs
byte-pinned against colab_package package_manifest.json.

Outputs triplet_u1/bundles/fold{N}/ with candidate_fold_data.npz +
recon_meta.json + recon_audits.json + split_sidecar.json.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVER = HERE.parent
ROOT = OVER.parents[1]
PKG = ROOT / 'faithful_repro' / 'colab_package_f0_selection_20261115'
SRC = PKG / 'src'
sys.path[:0] = [str(SRC), str(HERE)]

import split as S                       # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_arr = lambda a: hashlib.sha256(
    np.ascontiguousarray(a).tobytes()).hexdigest()


def write(p, x):
    with open(p, 'x', encoding='utf-8') as f:
        json.dump(x, f, indent=1)
        f.write('\n')


def main():
    mf = json.loads((PKG / 'package_manifest.json')
                    .read_text(encoding='utf-8'))['files']
    required = {
        'strict': PKG / 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',
        'semantic': SRC / 'semantic.py',
        'functional': SRC / 'functional.py',
        'contracts': SRC / 'contracts.py',
        'mesh_xlsx': PKG / 'inputs/HMDDv3.xlsx',
        'meshtoid': PKG / 'inputs/meshToID.csv',
        'mtrees': PKG / 'inputs/mtrees2019.bin'}
    manifest_keys = {
        'strict': 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',
        'semantic': 'src/semantic.py',
        'functional': 'src/functional.py',
        'contracts': 'src/contracts.py',
        'mesh_xlsx': 'inputs/HMDDv3.xlsx',
        'meshtoid': 'inputs/meshToID.csv',
        'mtrees': 'inputs/mtrees2019.bin'}
    pins = {}
    for name, p in required.items():
        h = sha(p)
        assert h == mf[manifest_keys[name]], ('pin', name, h)
        pins[name] = h

    # pin-before-deserialize on the strict npz (object arrays inside)
    z = np.load(required['strict'], allow_pickle=True)
    Y = np.asarray(z['Y'])
    assert Y.dtype != object and np.isin(Y, (0, 1)).all()
    n_m, n_d, C = Y.shape
    S_seq = np.asarray(z['sim_m1'], dtype=np.float64)
    S_gene = np.asarray(z['sim_d1'], dtype=np.float64)
    dis_names = z['dis_names'].tolist()
    assert len(dis_names) == n_d

    import semantic                     # noqa: E402
    import functional                   # noqa: E402
    ms = semantic.load_mesh_source(str(required['mesh_xlsx']),
                                   str(required['meshtoid']),
                                   str(required['mtrees']))
    sem = semantic.semantic_similarity(dis_names, ms,
                                       multi_descriptor='expand')
    S_sem = np.asarray(sem.S)

    out_root = HERE / 'bundles'
    entries = []
    for fold in range(5):
        target = out_root / f'fold{fold}'
        assert not target.exists(), target
        seg = S.fold_segments(Y, fold)
        train_pairs = S.pairs_of(seg['inner_train'])
        A = S.pair_mask(n_m, n_d, train_pairs)
        val_pairs, kept_val, dropped = S.val_pairs_disjoint(
            seg['inner_val'], train_pairs)
        assert len(kept_val) + dropped == len(seg['inner_val'])
        y_train = S.multihot_masked(Y, train_pairs, seg['inner_train'])
        assert (y_train.sum(1) > 0).all()
        y_val = S.multihot_masked(Y, val_pairs, seg['inner_val'])
        assert (y_val.sum(1) > 0).all() or len(val_pairs) == 0
        S_func = np.asarray(
            functional.functional_similarity(
                A.astype(np.float64), sem).sim)
        flat = {
            'view_mi_v1': np.hstack([S_seq, A.astype(np.float64)]),
            'view_mi_v2': np.hstack([S_func, A.astype(np.float64)]),
            'view_dis_v1': np.hstack([S_gene, A.T.astype(np.float64)]),
            'view_dis_v2': np.hstack([S_sem, A.T.astype(np.float64)]),
            'train_mask': A,
            'train_pairs': train_pairs.astype(np.int64),
            'y_train': y_train.astype(np.float64),
            'val_pairs': val_pairs.astype(np.int64),
            'y_val': y_val.astype(np.float64)}
        for k, v in flat.items():
            assert np.asarray(v).dtype != object, k
            if k.startswith('view_'):
                assert np.isfinite(v).all(), k
        stage = Path(str(target) + '.part')
        stage.mkdir(parents=True)
        npz = stage / 'candidate_fold_data.npz'
        np.savez(npz, **{k: (v.astype(np.float32)
                             if k.startswith('view_') else v)
                         for k, v in flat.items()})
        uncovered = [{'row': int(i), 'name': dis_names[i]}
                     for i in range(n_d) if not sem.coverage[i]]
        write(stage / 'recon_meta.json',
              {'npz_sha256': sha(npz),
               'entity_names': {'dis': list(dis_names)}})
        write(stage / 'recon_audits.json',
              {'descriptor_alias': {
                  'n_dis': n_d,
                  'n_uncovered_rows': len(uncovered),
                  'uncovered_rows': uncovered,
                  'generated_by': 'triplet_u1 build (semantic.py pinned)',
                  'provenance_note': 'descriptor-alias only'}})
        mask_zero_pairs = np.argwhere(~A.astype(bool)).astype(np.int64)
        pos_strata = S.positive_strata(seg['test'], A)
        sidecar = {
            'fold': fold,
            'protocol': 'triplet-level, rng(0) permutation, '
                        'array_split 5 — same partition as '
                        'triplet_baseline.py',
            'n_test_triplets': int(len(seg['test'])),
            'n_inner_train_triplets': int(len(seg['inner_train'])),
            'n_inner_val_triplets': int(len(seg['inner_val'])),
            'n_train_pairs': int(len(train_pairs)),
            'n_val_pairs': int(len(val_pairs)),
            'straddle_innerval_dropped': int(dropped),
            'test_pos_pair_seen': int(pos_strata.sum()),
            'test_pos_pair_unseen': int((1 - pos_strata).sum()),
            'n_mask_zero_pairs': int(len(mask_zero_pairs)),
            'inner_val_kept_triplets': int(len(kept_val)),
            'hashes': {k: _arr(v) for k, v in flat.items()},
            'split_hashes': {kk: seg[kk] for kk in seg
                             if kk.endswith('_sha')},
            'inner_val_kept_sha256': _arr(kept_val)}
        write(stage / 'split_sidecar.json', sidecar)
        stage.rename(target)
        entries.append({'fold': fold, 'dir': str(target),
                        'bundle_sha256': sha(target /
                                             'candidate_fold_data.npz'),
                        'sidecar': sidecar})
        print('BUILT fold%d: train_pairs=%d val_pairs=%d '
              'innerval_drop=%d test=%d seen=%d'
              % (fold, len(train_pairs), len(val_pairs), dropped,
                 len(seg['test']), int(pos_strata.sum())), flush=True)

    for name, p in required.items():
        assert sha(p) == pins[name], ('mutation', name)
    write(HERE / 'build_report.json',
          {'status': 'PASS', 'pins': pins, 'entries': entries})
    print('BUILD PASS', flush=True)


if __name__ == '__main__':
    main()
