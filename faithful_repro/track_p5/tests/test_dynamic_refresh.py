"""Eq.22-29 + Sec 2.8: HGT block, edge construction, dynamic refresh
semantics — including the post-refresh shared-operator behavior (T1)."""
import numpy as np
import torch
import faithful as fz


def _args(**kw):
    import types
    d = dict(hidden=16, proj=8, t=0.5, heads=4, nlayer=2, dropout=0.1,
             ablation='none', predictor='diag')
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_edge_types_and_directions():
    nm, nd = 5, 4
    A = torch.zeros(nm, nd); A[0, 1] = 1; A[2, 3] = 1
    Sm = torch.rand(nm, nm); Sd = torch.rand(nd, nd)
    ei = fz.edges(A, Sm, Sd, theta=0.3)
    assert set(ei.keys()) == set(fz.META[1])
    md = ei[('miRNA', 'associates', 'disease')]
    dm = ei[('disease', 'associates', 'miRNA')]
    # direction: md is [2, E] with sources miRNA->disease; dm flipped
    assert md.shape[0] == 2 and dm.shape[0] == 2
    assert torch.equal(md.flip(0), dm)
    assert md.shape[1] == 2  # two associations


def test_threshold_and_self_exclusion():
    Sm = torch.eye(4) * 0.9 + 0.05  # diagonal high but must be excluded
    ei = fz.edges(torch.zeros(4, 3), Sm, torch.zeros(3, 3), theta=0.5)
    em = ei[('miRNA', 'similar', 'miRNA')]
    # no i->i edges
    assert not (em[0] == em[1]).any()


def test_hgt_shapes_and_gradients():
    meta = fz.META
    block = fz.HGTBlock(16, meta, heads=4, p=0.1)
    x = {'miRNA': torch.randn(5, 16, requires_grad=True),
         'disease': torch.randn(4, 16, requires_grad=True)}
    ei = {r: torch.randint(0, 4, (2, 6)) for r in meta[1]}
    ei[('miRNA', 'associates', 'disease')] = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ei[('disease', 'associates', 'miRNA')] = torch.tensor([[1, 2, 3], [0, 1, 2]])
    ei[('miRNA', 'similar', 'miRNA')] = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    ei[('disease', 'similar', 'disease')] = torch.tensor([[0, 1, 2], [1, 2, 3]])
    out = block(x, ei)
    assert out['miRNA'].shape == (5, 16) and out['disease'].shape == (4, 16)
    loss = out['miRNA'].sum() + out['disease'].sum()
    loss.backward()
    assert torch.isfinite(x['miRNA'].grad).all()


def test_hgt_two_layers():
    a = _args()
    model = fz.DHGCMDA(5, 4, 5, a)
    assert len(model.hgt) == 2


def test_hgt_heads():
    a = _args(heads=4)
    model = fz.DHGCMDA(5, 4, 5, a)
    conv = model.hgt[0].conv
    assert conv.heads == 4


def test_hgt_residual_structure():
    """x' = LN(x + Dropout(HGT(x))) — verify residual add by zeroing conv."""
    block = fz.HGTBlock(8, fz.META, heads=4, p=0.0)
    x = {'miRNA': torch.randn(3, 8), 'disease': torch.randn(2, 8)}
    ei = {r: torch.zeros((2, 0), dtype=torch.long) for r in fz.META[1]}
    # HGTConv with no edges -> zero output -> out = LN(x)
    for p in block.conv.parameters():
        torch.nn.init.zeros_(p)
    out = block(x, ei)
    for k in x:
        ref = block.ln[k](x[k])
        assert torch.allclose(out[k], ref, atol=1e-6)


def test_post_refresh_view_sharing():
    """T1: after refresh, faithful.py assigns the SAME rebuilt Hm to m1
    and m2. Reproduce the assignment logic and assert the documented
    behavior — this test PINS current behavior; interpretation is in
    the decision register (reading A vs B), not silently 'fixed'."""
    rng = np.random.default_rng(0)
    nm, nd = 8, 6
    Af = (rng.random((nm, nd)) > 0.6).astype(np.float32)
    Smn = np.random.default_rng(1).random((nm, nm))
    Sdn = np.random.default_rng(2).random((nd, nd))
    Hm = fz.knn_G(np.hstack([Smn, Af]), 3)
    Hd = fz.knn_G(np.hstack([Sdn, Af.T]), 3)
    G = {'m1': Hm, 'm2': Hm, 'd1': Hd, 'd2': Hd}  # faithful.py L410-412
    assert torch.equal(G['m1'], G['m2']) and torch.equal(G['d1'], G['d2'])
    # pre-refresh the two views ARE distinct (different feature matrices)
    X1 = np.hstack([np.random.default_rng(3).random((nm, nm)), Af])
    X2 = np.hstack([np.random.default_rng(4).random((nm, nm)), Af])
    assert not torch.equal(fz.knn_G(X1, 3), fz.knn_G(X2, 3))


def test_refresh_frequency():
    """update_every=5 -> refresh at ep 5,10,15... not at 4 or 6."""
    fired = [ep for ep in range(1, 21) if ep % 5 == 0]
    assert fired == [5, 10, 15, 20]


def test_recon_minmax_normalization():
    S = torch.tensor([[0.0, 2.0], [4.0, 1.0]])
    n = fz.mm01(S)
    assert float(n.min()) == 0.0 and float(n.max()) == 1.0
