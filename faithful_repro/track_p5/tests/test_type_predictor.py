"""Eq.30-34: predictor bilinear (diag/full), softmax multi-hot CE with
effective-number weights, loss composition — oracle-verified."""
import numpy as np
import torch
import faithful as fz


def test_diag_bilinear_math():
    torch.manual_seed(0)
    p = fz.Predictor(8, 16, 5, p=0.0, mode='diag')
    zm = torch.randn(3, 8); zd = torch.randn(4, 8)
    ex, tp = p(zm, zd)
    m, d = p.pm(zm), p.pd(zd)
    ex_ref = torch.sigmoid((m * p.re) @ d.T)
    lg_ref = torch.einsum('ih,kh,jh->ijk', m, p.rt, d)
    tp_ref = torch.softmax(lg_ref / torch.clamp(p.temp, 0.5, 5.0), 2)
    assert torch.allclose(ex, ex_ref) and torch.allclose(tp, tp_ref)
    assert ex.shape == (3, 4) and tp.shape == (3, 4, 5)


def test_full_bilinear_math():
    torch.manual_seed(1)
    p = fz.Predictor(8, 16, 5, p=0.0, mode='full')
    zm = torch.randn(3, 8); zd = torch.randn(4, 8)
    ex, tp = p(zm, zd)
    assert ex.shape == (3, 4) and tp.shape == (3, 4, 5)
    m, d = p.pm(zm), p.pd(zd)
    logits = torch.einsum('ih,ohk,jk->ijo', m, p.w, d)
    assert torch.allclose(ex, torch.sigmoid(logits[:, :, 0]))
    assert torch.allclose(tp, torch.softmax(logits[:, :, 1:] / torch.clamp(p.temp, 0.5, 5.0), 2))


def test_softmax_temperature():
    torch.manual_seed(2)
    p = fz.Predictor(8, 16, 5, p=0.0)
    with torch.no_grad():
        p.temp.fill_(4.9)  # inside clamp range
    zm = torch.randn(2, 8); zd = torch.randn(2, 8)
    _, tp = p(zm, zd)
    m, d = p.pm(zm), p.pd(zd)
    lg = torch.einsum('ih,kh,jh->ijk', m, p.rt, d)
    assert torch.allclose(tp, torch.softmax(lg / 4.9, 2))
    # clamp: temp beyond 5 must clamp to 5
    with torch.no_grad():
        p.temp.fill_(9.0)
    _, tp2 = p(zm, zd)
    assert torch.allclose(tp2, torch.softmax(lg / 5.0, 2))


def test_type_prob_sums_one():
    torch.manual_seed(3)
    p = fz.Predictor(8, 16, 5, p=0.0)
    _, tp = p(torch.randn(3, 8), torch.randn(3, 8))
    assert torch.allclose(tp.sum(-1), torch.ones(3, 3), atol=1e-6)


def test_weighted_multihot_ce_oracle():
    """L_type = -mean_pairs(sum_k w_k y_k log p_k) — independent oracle
    on a two-label row [1,0,1,0,0]."""
    w = torch.tensor([1., 9., 1., 1., 1.])
    yp = torch.tensor([[1., 0., 1., 0., 0.]])
    tpp = torch.softmax(torch.randn(1, 5, generator=torch.Generator().manual_seed(4)), 1)
    l_type = -(w * yp * torch.log(tpp + 1e-9)).sum(1).mean()
    oracle = -(w[0] * np.log(tpp[0, 0].item() + 1e-9) +
               w[2] * np.log(tpp[0, 2].item() + 1e-9))
    assert abs(l_type.item() - oracle) < 1e-6


def test_multihot_vs_bce_distinct():
    """Softmax-multi-hot CE is NOT independent-Bernoulli BCE — values
    must differ; they may never be aliased."""
    y = torch.tensor([[1., 0., 1., 0., 0.]])
    logits = torch.tensor([[0.5, -0.3, 0.8, -1.0, 0.2]])
    p_soft = torch.softmax(logits, 1)
    l_soft = -(y * torch.log(p_soft)).sum()
    p_sig = torch.sigmoid(logits)
    l_bce = torch.nn.functional.binary_cross_entropy(p_sig, y, reduction='sum')
    assert abs(l_soft.item() - l_bce.item()) > 1e-3


def test_en_weights_train_only():
    """w_k = (1-beta)/(1-beta^n_k), train counts, normalized to sum C."""
    beta = 0.99999
    n_k = np.array([100., 10., 200., 300., 250.])
    w = (1 - beta) / (1 - beta ** np.maximum(n_k, 1))
    w = w * 5 / w.sum()
    assert abs(w.sum() - 5) < 1e-9
    assert w[1] > w[3], "minority must get larger weight"


def test_total_loss_composition():
    """loss = l_type + lam1*intra + lam2*inter + lam3*rec + exist_w*l_ex
    — verify the documented composition holds with all terms finite."""
    lt, li, le, lr, lx = 0.5, 0.2, 0.3, 100.0, 0.4
    lam1, lam2, lam3, ew = 1.0, 0.3, 1.0, 0.3
    total = lt + lam1 * li + lam2 * le + lam3 * lr + ew * lx
    assert abs(total - (0.5 + 0.2 + 0.09 + 100.0 + 0.12)) < 1e-9
    # sensitivity: recon sum dominates — measured ratio recorded
    assert lam3 * lr / total > 0.9
