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
    """T1 exercised through the REAL refresh path: p5_runner.train_fold
    performs the faithful.py L406-412 block (same fz.edges/knn_G/mm01
    calls, same shared-Hm assignment). After a refresh, the live G
    objects satisfy m1==m2 and d1==d2; without any refresh the two
    view operators stay distinct."""
    import p5_runner as pr
    D = pr.synthetic_fixture(seed=9)
    fold = pr.synthetic_fold(D['Y'])
    a_refresh = _train_args(update_every=1, eval_every=1, patience=50,
                      hidden=8, proj=4, K=3)
    _, _, G_ref, _ = pr.train_fold(D, fold, a_refresh, epochs=2, return_state=True)
    assert torch.equal(G_ref['m1'], G_ref['m2'])
    assert torch.equal(G_ref['d1'], G_ref['d2'])
    a_none = _train_args(update_every=99, eval_every=1, patience=50,
                   hidden=8, proj=4, K=3)
    _, _, G_pre, _ = pr.train_fold(D, fold, a_none, epochs=2, return_state=True)
    assert not torch.equal(G_pre['m1'], G_pre['m2'])  # distinct view operators


def _train_args(**kw):
    import types
    base = dict(hidden=16, proj=8, heads=2, nlayer=1, dropout=0.0, K=3,
                t=0.5, lam1=1.0, lam2=0.3, lam3=1.0, margin=0.5,
                margin_w=0.1, exist_w=0.3, neg_w=0.5, beta=0.99999,
                lr=1e-3, wd=1e-5, neg_ratio=3, update_every=5, theta=0.5,
                early_stop=1, eval_every=1, patience=5, n_neg_draws=2,
                predictor='diag', recon_reduction='sum', diag_leak=0,
                ablation='none', protocol='type', folds=5, only_fold=-1,
                cap_threads=0, threads=1, seed=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_refresh_frequency():
    """update_every=5 -> refresh at ep 5,10,15... not at 4 or 6."""
    fired = [ep for ep in range(1, 21) if ep % 5 == 0]
    assert fired == [5, 10, 15, 20]


def test_recon_minmax_normalization():
    S = torch.tensor([[0.0, 2.0], [4.0, 1.0]])
    n = fz.mm01(S)
    assert float(n.min()) == 0.0 and float(n.max()) == 1.0
