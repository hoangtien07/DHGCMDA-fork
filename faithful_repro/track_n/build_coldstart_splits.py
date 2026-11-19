# -*- coding: utf-8 -*-
"""Track N: build cold-start split indices (entity-level) + support
audit + provenance canary. Code-only; produces frozen JSON+sha256.
Run: .venv-bank1/bin/python build_coldstart_splits.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TENSOR = HERE.parent / 'data' / \
    'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
OUT = HERE / 'splits'
SEEDS = (1234, 42, 2025)


def main():
    z = np.load(TENSOR, allow_pickle=True)
    Y = z['Y'].astype(int)
    M, D, T = Y.shape
    OUT.mkdir(exist_ok=True)
    manifest = {'tensor_sha256': hashlib.sha256(
        TENSOR.read_bytes()).hexdigest(),
        'seeds': SEEDS, 'tracks': {}}
    for track, n_ent, axis in (('mirna_cold', M, 0),
                               ('disease_cold', D, 1)):
        manifest['tracks'][track] = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_ent)
            folds = np.array_split(perm, 5)
            rec = {'folds': [], 'audit': []}
            for f, ents in enumerate(folds):
                mask = np.zeros((M, D), bool)
                idx = np.array(sorted(ents))
                if axis == 0:
                    mask[idx, :] = True
                else:
                    mask[:, idx] = True
                test_pairs = np.argwhere(mask & (Y.sum(-1) > 0))
                train_pairs = np.argwhere(~mask & (Y.sum(-1) > 0))
                rec['folds'].append({
                    'held_out_entities': idx.tolist(),
                    'test_pairs': test_pairs.tolist(),
                    'train_pairs': train_pairs.tolist()})
                yt = Y.sum(-1) > 0
                rec['audit'].append({
                    'fold': f, 'n_entities': int(len(idx)),
                    'n_test_pairs': int((mask & yt).sum()),
                    'n_train_pairs': int((~mask & yt).sum()),
                    'test_type_support': Y[mask].sum(0).tolist(),
                    'train_type_support': Y[~mask].sum(0).tolist()})
            blob = json.dumps(rec, sort_keys=True).encode()
            fn = OUT / f'{track}_seed{seed}.json'
            fn.write_bytes(blob)
            manifest['tracks'][track][str(seed)] = {
                'file': fn.name,
                'sha256': hashlib.sha256(blob).hexdigest()}
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    # entity-level disjointness check
    for track in manifest['tracks']:
        for seed in SEEDS:
            rec = json.loads(
                (OUT / f'{track}_seed{seed}.json').read_text())
            ents = [set(f['held_out_entities']) for f in rec['folds']]
            for i in range(5):
                for j in range(i + 1, 5):
                    assert not (ents[i] & ents[j]), (track, seed, i, j)
                tr = {tuple(p) for p in rec['folds'][i]['train_pairs']}
                te = {tuple(p) for p in rec['folds'][i]['test_pairs']}
                assert not (tr & te), (track, seed, i)
            print(track, seed, 'disjoint OK; support:',
                  rec['audit'][0]['test_type_support'])
    print('MANIFEST written')


if __name__ == '__main__':
    main()
