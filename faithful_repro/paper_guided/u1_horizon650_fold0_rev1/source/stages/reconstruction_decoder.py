# -*- coding: utf-8 -*-
"""Eq. 19 reconstruction decoder heads (R1 preparation stage).

Contract: see R1_TARGET_DECODER_CONTRACT.md in this directory. Provides
the literal bias-free two-linear projection head and a factory that
returns independent miRNA/disease heads in a ModuleDict. Initialization
is seeded inside torch.random.fork_rng so the caller's global RNG is
byte-preserved; the disabled path constructs nothing and draws no RNG.
CPU-only scope. No trainer integration, no IO besides in-memory
state_dict.
"""
import torch
from torch import nn
import torch.nn.functional as F

_SEED_OFFSET = 4001


def _check_dim(name, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError('%s must be an int, got %r' % (name, v))
    if v <= 0:
        raise ValueError('%s must be positive, got %d' % (name, v))


class ReconstructionHead(nn.Module):
    """Literal Eq. 19 head: W2 ReLU(W1 Z) with bias-free linears.

    Linear(hidden, hidden, bias=False) -> ReLU ->
    Linear(hidden, proj, bias=False). No dropout, normalization,
    sigmoid, clipping or identity-diagonal overwrite. Bias-free follows
    the displayed equation; widths/init are coordinator choices
    (future 256/64).
    """

    def __init__(self, hidden, proj):
        super().__init__()
        _check_dim('hidden', hidden)
        _check_dim('proj', proj)
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, proj, bias=False)

    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z)))


def make_reconstruction_heads(hidden, proj, seed, enabled=True):
    """Independent miRNA/disease Eq. 19 heads, or None when disabled.

    enabled=False returns None immediately — no ModuleDict, no head
    construction, zero RNG consumption. When enabled, both heads are
    initialized inside torch.random.fork_rng(devices=[]) under
    torch.manual_seed(seed + 4001), so the caller's global CPU RNG
    state is preserved exactly. Returns nn.ModuleDict({'mi': head,
    'dis': head}) with independent parameter storage.
    """
    if not isinstance(enabled, bool):
        raise TypeError('enabled must be bool, got %r' % (enabled,))
    if not enabled:
        return None
    _check_dim('hidden', hidden)
    _check_dim('proj', proj)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError('seed must be an int, got %r' % (seed,))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + _SEED_OFFSET)
        return nn.ModuleDict({'mi': ReconstructionHead(hidden, proj),
                              'dis': ReconstructionHead(hidden, proj)})
