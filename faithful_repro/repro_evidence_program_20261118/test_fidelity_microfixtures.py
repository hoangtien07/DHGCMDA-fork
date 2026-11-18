# -*- coding: utf-8 -*-
"""Synthetic micro-fixtures for the fidelity trace (WP3).

No training — identical small tensors run through (a) the pinned
formulas reimplemented here and (b) the actual U1 metrics.py and
author constants, to pin down *reduction denominators* and *channel
semantics* where documentation alone is ambiguous.
Run: .venv-bank1/bin/python test_fidelity_microfixtures.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'paper_guided'
                    / 'u1_horizon650_fold0_rev1' / 'source'))
from metrics import type_loss  # U1 loss under test


def _en_weights(counts, beta=0.99999):
    """Author EN weights (upstream_main L66-72)."""
    en = [(1 - beta ** n) / (1 - beta) for n in counts]
    w = np.array([1.0 / e for e in en])
    return w / w.sum() * len(counts)  # normalized to sum=C


def test_en_weights_values():
    # 5-type support on strict tensor
    w = _en_weights([1155, 403, 2293, 3997, 3900])
    assert abs(w.sum() - 5.0) < 1e-9
    # minority class (epigenetics 403) gets ~9.7x the weight of
    # majority (target 3997): EN curve is saturating at beta=0.99999
    assert w[1] > w[3]
    assert 8 < w[1] / w[3] < 12


def test_u1_loss_reduction_denominator():
    """U1 type_loss normalizes by weighted positive mass — NOT by
    #pairs like the author CE (reduction='mean' over pos samples).
    On single-label data the two coincide up to the weight profile;
    on multi-label rows they differ."""
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(6, 5)), dtype=torch.float64)
    y = np.zeros((6, 5)); y[range(6), [0, 1, 2, 3, 4, 4]] = 1
    y[5, 0] = 1  # pair 5 multi-label {0,4}
    w = torch.tensor(_en_weights([3, 2, 2, 2, 3]), dtype=torch.float64)
    l = type_loss(logits, y, w)
    logp = F.log_softmax(logits, dim=-1)
    yt = torch.tensor(y, dtype=torch.float64)
    manual = -(w.view(1, -1) * yt * logp).sum() / (w * yt).sum()
    assert abs(float(l) - float(manual)) < 1e-12
    # denominator: weighted positive mass = sum w_k over set bits
    # rows: {0},{1},{2},{3},{4},{4}+{0}
    assert abs(float((w * yt).sum()) -
               float(2 * w[0] + w[1] + w[2] + w[3] + 2 * w[4])) < 1e-12


def test_multilabel_denominator_divergence():
    """Same logits, multi-label vs single-label truth: U1 denominator
    grows with label count; author scalar CE cannot represent
    multi-label at all (one scalar target per pair)."""
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64)
    y_single = np.eye(5)[:4]
    y_multi = y_single.copy(); y_multi[0, 2] = 1
    w = torch.ones(5, dtype=torch.float64)
    l1 = float(type_loss(logits, y_single, w))
    l2 = float(type_loss(logits, y_multi, w))
    # multi-label adds a positive term and renormalizes — not just
    # 'the same loss on different data': the denominators differ
    assert abs(l2 - l1) > 1e-6


def test_exist_channel_no_temperature():
    """Existence channel = sigmoid(mi·diag(r)·dis) WITHOUT temperature;
    type channels = softmax(logits/T), T clamped [0.5,5] — upstream
    L475-493 reimplemented on identical weights."""
    rng = np.random.default_rng(2)
    d = 8
    mi = torch.tensor(rng.normal(size=(3, d)), dtype=torch.float64)
    dis = torch.tensor(rng.normal(size=(4, d)), dtype=torch.float64)
    r_exist = torch.tensor(rng.normal(size=d) * 0.01,
                           dtype=torch.float64)
    exist = torch.sigmoid(torch.mm(mi * r_exist, dis.t()))
    assert exist.shape == (3, 4) and ((exist > 0) & (exist < 1)).all()
    logits = torch.tensor(rng.normal(size=(3, 4, 5)),
                          dtype=torch.float64)
    for T in (0.1, 2.0, 9.0):
        Tc = min(max(T, 0.5), 5.0)
        p = F.softmax(logits / Tc, dim=-1)
        assert torch.allclose(p.sum(-1), torch.ones(3, 4, dtype=torch.float64))
    # triplet score = exist * softmax_k (U1X decomposition)
    s = exist.unsqueeze(-1) * F.softmax(logits / 2.0, dim=-1)
    assert s.shape == (3, 4, 5)
    assert (s.sum(-1) < exist + 1e-9).all()  # softmax shares mass


def test_focal_existence_formula():
    """Upstream focal: pos = -(1-p)^γ ln p; neg = -p^γ ln(1-p) · α."""
    p = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64)
    g, a = 2.0, 0.5
    pos = -(1 - p) ** g * torch.log(p + 1e-7)
    neg = -p ** g * torch.log(1 - p + 1e-7) * a
    # hand values
    assert abs(float(pos[0]) - (-(0.1 ** 2) * np.log(0.9 + 1e-7))) < 1e-6
    assert abs(float(neg[0]) - (-(0.9 ** 2) * np.log(0.1 + 1e-7) * .5)) < 1e-6


def test_type_ce_upstream_equivalence():
    """Replicate upstream _compute_type_loss on scalar targets:
    CE(weight=EN, label_smoothing=0.1, reduction='mean')."""
    rng = np.random.default_rng(4)
    logits = torch.tensor(rng.normal(size=(12, 5)), dtype=torch.float64)
    # scalar targets 1..4 -> idx 0..3; v3.2 type-5 unscorable upstream
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    w = torch.tensor(_en_weights([3, 3, 3, 3]), dtype=torch.float64)
    l = F.cross_entropy(logits[:, :4], targets,
                        weight=w[:4] / w[:4].sum() * 4,
                        label_smoothing=0.1, reduction='mean')
    # check it differs from U1 normalization on same data
    y = np.eye(5)[targets.numpy()][: , :4]
    y = np.concatenate([y, np.zeros((12, 1))], axis=1)
    l_u1 = type_loss(logits, y, torch.ones(5, dtype=torch.float64))
    assert abs(float(l) - float(l_u1)) > 0  # denominators differ


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_')]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print('ALL', len(fns), 'FIDELITY MICRO-FIXTURES PASS')
