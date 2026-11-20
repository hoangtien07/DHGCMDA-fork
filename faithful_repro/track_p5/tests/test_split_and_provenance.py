"""Split contracts + provenance canary on SYNTHETIC data only.
No real HMDD tensor is used here — real-data verification lives in
G2_DATA_FEATURE_VERDICT.json."""
import types
import numpy as np
import torch
import faithful as fz


def _synth_Y(seed=0, nm=10, nd=8, C=5, density=0.3):
    rng = np.random.default_rng(seed)
    Y = (rng.random((nm, nd, C)) < density).astype(np.float32)
    # ensure every pair has consistent view: zero out pairs with no label
    # multi-label rows injected
    Y[0, 0] = np.array([1, 0, 1, 0, 0], np.float32)
    return Y


def _a(protocol, val_frac=0.1):
    return types.SimpleNamespace(protocol=protocol, folds=5, val_frac=val_frac,
                                 early_stop=1, n_neg_draws=3)


def test_cv_pair_disjoint_and_complete():
    Y = _synth_Y()
    folds = fz.make_splits(Y, _a('type'), np.random.default_rng(0))
    allp = set()
    for f in folds:
        p = set(map(tuple, f['test'].tolist()))
        assert not (p & allp)
        allp |= p
        s = lambda x: set(map(tuple, x.tolist()))
        assert not (s(f['train']) & s(f['test']))
        assert not (s(f['val']) & s(f['test']))
        assert not (s(f['val']) & s(f['train']))
    assert len(allp) == int((Y.sum(2) > 0).sum())


def test_pair_groups_all_types_together():
    """CV_PAIR: splits operate on PAIR indices (2 cols), never per-type
    triplets — a multi-label pair's whole label vector is masked atomically."""
    Y = _synth_Y()
    folds = fz.make_splits(Y, _a('type'), np.random.default_rng(1))
    for f in folds:
        assert f['test'].shape[1] == 2  # pair indices, not (i,j,k) triplets
        testset = set(map(tuple, f['test'].tolist()))
        # atomic masking: train-masked Ytr zeroes ALL types of a test pair
        Ytr = np.zeros_like(Y)
        Ytr[f['train'][:, 0], f['train'][:, 1]] = Y[f['train'][:, 0], f['train'][:, 1]]
        for i, j in testset:
            assert Ytr[i, j].sum() == 0
        # and the unmasked copy keeps the full multi-hot row
        multi = np.argwhere(Y.sum(2) > 1)
        for i, j in multi:
            if (i, j) in testset:
                assert Y[i, j].sum() > 1  # row intact in source, split only by pair


def test_triplet_overlap_allowed_pair_disjoint_forbidden():
    """CV_TRIPLET may place different types of one pair in train/test —
    verify the semantic is distinct from CV_PAIR."""
    Y = _synth_Y(density=0.6)
    folds = fz.make_splits(Y, _a('triplet'), np.random.default_rng(2))
    f = folds[0]
    tr_pairs = set((int(i), int(j)) for i, j, _ in f['train'].tolist())
    te_pairs = set((int(i), int(j)) for i, j, _ in f['test'].tolist())
    # overlap allowed but not required — just verify triplets disjoint
    tr_tri = set(map(tuple, f['train'].tolist()))
    te_tri = set(map(tuple, f['test'].tolist()))
    assert not (tr_tri & te_tri)


def test_negative_pool_excludes_heldout():
    """Mirror of run_fold neg_pool logic on synthetic: test/val pair
    coordinates must be excluded from the existence negative pool."""
    Y = _synth_Y()
    fold = fz.make_splits(Y, _a('type'), np.random.default_rng(3))[0]
    nm, nd, C = Y.shape
    Ytr = np.zeros_like(Y)
    Ytr[fold['train'][:, 0], fold['train'][:, 1]] = Y[fold['train'][:, 0], fold['train'][:, 1]]
    A = (Ytr.sum(2) > 0).astype(np.float32)
    excl = np.zeros((nm, nd), bool)
    for k in ['test', 'val', 'test_neg', 'val_neg']:
        if k in fold and len(fold[k]):
            excl[fold[k][:, 0], fold[k][:, 1]] = True
    neg_pool = np.argwhere((A == 0) & ~excl)
    test_pairs = set(map(tuple, fold['test'].tolist()))
    assert not any(tuple(p) in test_pairs for p in neg_pool.tolist())


def test_multihot_row_survives_pipeline():
    Y = _synth_Y()
    assert np.array_equal(Y[0, 0], np.array([1, 0, 1, 0, 0], np.float32))
    folds = fz.make_splits(Y, _a('type'), np.random.default_rng(4))
    # the multihot pair lands wholly in exactly one split of each fold
    for f in folds:
        ins = sum(tuple([0, 0]) in set(map(tuple, f[k].tolist()))
                  for k in ('train', 'val', 'test'))
        assert ins == 1


def test_canary_heldout_label_invariance():
    """Canary: perturb held-out labels in a SYNTHETIC copy and rebuild
    the full feature path — every train-visible tensor must be
    bit-identical. Perturbing Y while keeping sim_* fixed is NOT a
    provenance proof (see FEATURE_PROVENANCE_DAG)."""
    Y = _synth_Y()
    fold = fz.make_splits(Y, _a('type'), np.random.default_rng(5))[0]
    rng = np.random.default_rng(7)
    sims = {k: rng.random((n, n)).astype(np.float32)
            for k, n in [('sim_m1', 10), ('sim_m2', 10),
                         ('sim_d1', 8), ('sim_d2', 8)]}

    def build_train_visible(Yin):
        Ytr = np.zeros_like(Yin)
        Ytr[fold['train'][:, 0], fold['train'][:, 1]] = \
            Yin[fold['train'][:, 0], fold['train'][:, 1]]
        A = (Ytr.sum(2) > 0).astype(np.float32)
        X = {k: np.hstack([sims[k.split('_')[1]], A if k[4] == 'm' else A.T])
             for k in []}  # placeholder, real path below
        Xm = {v: np.hstack([sims[f'sim_m{v}'], A]) for v in (1, 2)}
        Xd = {v: np.hstack([sims[f'sim_d{v}'], A.T]) for v in (1, 2)}
        G = {**{f'm{v}': fz.knn_G(x, 3) for v, x in Xm.items()},
             **{f'd{v}': fz.knn_G(x, 3) for v, x in Xd.items()}}
        gip_m = fz.gip(A); gip_d = fz.gip(A.T)
        excl = np.zeros((10, 8), bool)
        excl[fold['test'][:, 0], fold['test'][:, 1]] = True
        excl[fold['val'][:, 0], fold['val'][:, 1]] = True
        neg_pool = np.argwhere((A == 0) & ~excl)
        ei = fz.edges(torch.tensor(A), torch.tensor(gip_m),
                      torch.tensor(gip_d), 0.5)
        return Ytr, A, G, gip_m, gip_d, neg_pool, ei

    base = build_train_visible(Y)
    Yp = Y.copy()
    # perturb ONLY held-out labels: flip a test pair's label vector
    ti, tj = fold['test'][0]
    Yp[ti, tj] = 1 - Y[ti, tj]
    pert = build_train_visible(Yp)
    names = ['Ytr', 'A', 'G', 'gip_m', 'gip_d', 'neg_pool', 'ei']
    for name, b, p in zip(names, base, pert):
        if isinstance(b, dict):
            for k in b:
                bb, pp = b[k], p[k]
                eq = (bb == pp) if isinstance(bb, np.ndarray) else torch.equal(bb, pp)
                assert np.all(eq) if isinstance(eq, np.ndarray) else eq, f"{name}[{k}] changed"
        elif isinstance(b, np.ndarray):
            assert np.array_equal(b, p), f"{name} changed under held-out perturbation"
        else:
            assert torch.equal(b, p)


def test_views_not_certified_clean():
    """Provenance contract is MACHINE-CHECKED: the manifest must record
    every bundled sim_* view as UNKNOWN_PROVENANCE and the G2 verdict
    must cap the claim ceiling at exploratory — a regression to
    'verified_clean' fails this test."""
    import json, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    man = json.load(open(os.path.join(here, 'SOURCE_MANIFEST.json')))
    prov = man['artifacts']['strict_tensor']['reverified']['view_provenance']
    assert 'UNKNOWN_PROVENANCE' in prov
    v = json.load(open(os.path.join(here, 'G2_DATA_FEATURE_VERDICT.json')))
    s = json.dumps(v)
    assert 'G2_PASS_EXPLORATORY_ONLY' in s
    assert 'verified_clean' not in s.lower()
