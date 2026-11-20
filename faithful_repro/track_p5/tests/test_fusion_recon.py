"""Eq.16-21: AVF fusion + reconstruction — shapes, gate semantics,
Gram/Frobenius scale and gradients on synthetic tensors."""
import numpy as np
import torch
import faithful as fz


def test_avf_gate_shapes():
    m = fz.AVF(16, r=4)
    assert m.W1.in_features == 16 and m.W1.out_features == 4
    assert m.W2.out_features == 16  # channel-wise gate
    z = m([torch.randn(5, 16), torch.randn(5, 16)])
    assert z.shape == (5, 16)


def test_avf_output_nonneg():
    torch.manual_seed(0)
    m = fz.AVF(8)
    z = m([torch.randn(6, 8), torch.randn(6, 8)])
    assert (z >= 0).all(), "ReLU output must be nonneg"


def test_avf_view_specific_output():
    """Distinct view inputs must produce different gated outputs when
    the gate weights are nonzero — proves dual-view paths differ."""
    torch.manual_seed(1)
    m = fz.AVF(8)
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    z_same = m([z1, z1.clone()])
    z_diff = m([z1, z2])
    assert not torch.allclose(z_same, z_diff)


def test_avf_beta_learnable():
    torch.manual_seed(2)
    m = fz.AVF(8)
    assert m.beta.requires_grad
    b = torch.softmax(m.beta, 0)
    assert abs(float(b.sum()) - 1.0) < 1e-6
    # changing beta changes output when views differ
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    out1 = m([z1, z2]).detach().clone()
    with torch.no_grad():
        m.beta[1] += 2.0
    out2 = m([z1, z2]).detach()
    assert not torch.allclose(out1, out2)


def test_recon_projection_width():
    m = fz.Recon(16)
    assert m.W1.out_features == 16 and m.W2.out_features == 8  # d -> d/2
    z = torch.randn(7, 16)
    zp = m.W2(torch.relu(m.W1(z)))
    assert zp.shape == (7, 8)


def test_gram_semantics_symmetric():
    torch.manual_seed(3)
    m = fz.Recon(8)
    S = m(torch.randn(9, 8))
    assert S.shape == (9, 9)
    assert torch.allclose(S, S.T, atol=1e-6)


def test_recon_loss_sum_vs_mean_scale():
    """sum is paper-literal; record the scale ratio so the dominance
    question is measured, not assumed."""
    torch.manual_seed(4)
    m = fz.Recon(8)
    S = m(torch.randn(10, 8))
    T = torch.randn(10, 10)
    l_sum = ((T - S) ** 2).sum()
    l_mean = ((T - S) ** 2).mean()
    ratio = (l_sum / l_mean).item()
    assert 50 < ratio < 200, f"expected ~n^2=100 ratio, got {ratio}"
    # gradient magnitudes scale by the same factor on the SAME input
    z = torch.randn(10, 8, requires_grad=True)
    m2 = fz.Recon(8)
    ((T - m2(z)) ** 2).sum().backward()
    g_sum = z.grad.norm().item()
    z.grad = None
    ((T - m2(z)) ** 2).mean().backward()
    g_mean = z.grad.norm().item()
    assert abs(g_sum / g_mean - 100.0) < 1e-3  # n^2 = 100


def test_recon_gradient_finite():
    torch.manual_seed(5)
    m = fz.Recon(8)
    z = torch.randn(6, 8, requires_grad=True)
    S = m(z)
    target = torch.randn(6, 6)
    loss = ((target - S) ** 2).sum()
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
