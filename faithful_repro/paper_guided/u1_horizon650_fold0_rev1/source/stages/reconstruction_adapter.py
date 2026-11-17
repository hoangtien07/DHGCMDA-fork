# -*- coding: utf-8 -*-
"""Disabled-safe reconstruction composition adapter (R1 preparation).

Contract: see RECONSTRUCTION_ADAPTER_CONTRACT.md in this directory.
Registers the accepted factory's independent miRNA/disease Eq. 19
heads and composes them with the accepted masked raw-Gram loss.
Preparation only — no trainer integration, no file IO, no RNG in
forward, no labels, no second encoder pass.
"""
import torch
from torch import nn

from reconstruction import reconstruction_loss
from reconstruction_decoder import make_reconstruction_heads


class ReconstructionAdapter(nn.Module):
    """Compose Eq. 19 heads + masked reconstruction loss.

    enabled=True registers ModuleDict({'mi','dis'}) from the accepted
    seed+4001 fork_rng factory (parameters appear as
    heads.{mi,dis}.fc{1,2}.weight; global RNG preserved).
    enabled=False stores heads=None — no parameters, no RNG, no
    validation of unused dims/seed; forward returns None immediately.
    """

    def __init__(self, hidden, proj, seed, enabled=True):
        super().__init__()
        self.heads = make_reconstruction_heads(hidden, proj, seed,
                                               enabled)

    def _check_input(self, name, z):
        if not isinstance(z, torch.Tensor):
            raise TypeError('%s must be a torch tensor' % name)
        if z.dim() != 2:
            raise ValueError('%s must be 2-D [n, hidden], got %d dims'
                             % (name, z.dim()))
        w = self.heads['mi'].fc1.weight
        if z.shape[1] != w.shape[1]:
            raise ValueError('%s feature dim %d does not match head '
                             'hidden %d' % (name, z.shape[1], w.shape[1]))
        if z.dtype != w.dtype:
            raise ValueError('%s dtype %s does not match head weight '
                             'dtype %s' % (name, z.dtype, w.dtype))
        if z.device != w.device:
            raise ValueError('%s device %s does not match head weight '
                             'device %s' % (name, z.device, w.device))

    def forward(self, zm_pre=None, zd_pre=None, mi_target=None,
                mi_valid_mask=None, dis_target=None,
                dis_valid_mask=None):
        """One head application per modality + masked raw-Gram loss.

        Disabled: returns None immediately (no validation, no target
        access). Enabled: returns dict with 'l_recon' =
        (masked_sum_mi + masked_sum_dis) / (valid_count_mi +
        valid_count_dis) — valid-count-weighted global mean, unscaled
        (lambda is the caller's, added once) — plus per-modality
        masked_sum/masked_mean/valid_count and the full gram_mi/gram_dis
        matrices. Targets get no gradient; forward draws no RNG and
        mutates no input.
        """
        if self.heads is None:
            return None
        self._check_input('zm_pre', zm_pre)
        self._check_input('zd_pre', zd_pre)
        out_mi = reconstruction_loss(self.heads['mi'](zm_pre),
                                     mi_target, mi_valid_mask)
        out_dis = reconstruction_loss(self.heads['dis'](zd_pre),
                                      dis_target, dis_valid_mask)
        l_recon = ((out_mi['masked_sum'] + out_dis['masked_sum'])
                   / (out_mi['valid_count'] + out_dis['valid_count']))
        if not torch.isfinite(l_recon):
            raise ValueError('non-finite combined l_recon; rejecting '
                             'rather than emitting NaN/Inf')
        return {
            'l_recon': l_recon,
            'masked_sum_mi': out_mi['masked_sum'],
            'masked_mean_mi': out_mi['masked_mean'],
            'valid_count_mi': out_mi['valid_count'],
            'gram_mi': out_mi['gram'],
            'masked_sum_dis': out_dis['masked_sum'],
            'masked_mean_dis': out_dis['masked_mean'],
            'valid_count_dis': out_dis['valid_count'],
            'gram_dis': out_dis['gram'],
        }
