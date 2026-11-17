# -*- coding: utf-8 -*-
"""Pure masked similarity reconstruction loss (paper Eq. 19-21).

Contract: see RECONSTRUCTION_CONTRACT.md in this directory. Consumes an
already-projected embedding [n, d], a detached target [n, n] and an
explicit boolean valid mask [n, n]. Predicts the RAW Gram Z Z^T
(Eq. 20) — not cosine, no clipping, no forced diagonal. Contains no
parameters, no RNG, no IO.
"""
import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _validate(projected, target, valid_mask):
    for name, t in (('projected', projected), ('target', target),
                    ('valid_mask', valid_mask)):
        if not isinstance(t, torch.Tensor):
            raise TypeError('%s must be a torch tensor' % name)
        if t.dim() != 2:
            raise ValueError('%s must be 2-D, got %d dims'
                             % (name, t.dim()))
    n, d = projected.shape
    if n == 0 or d == 0:
        raise ValueError('projected must be non-empty [n, d]')
    if target.shape != (n, n):
        raise ValueError('target must be [%d, %d], got %s'
                         % (n, n, tuple(target.shape)))
    if valid_mask.shape != (n, n):
        raise ValueError('valid_mask must be [%d, %d], got %s'
                         % (n, n, tuple(valid_mask.shape)))
    if projected.dtype not in _SUPPORTED_DTYPES:
        raise ValueError('projected dtype must be float32/float64, got '
                         '%s' % projected.dtype)
    if target.dtype != projected.dtype:
        raise ValueError('dtype mismatch: projected %s vs target %s'
                         % (projected.dtype, target.dtype))
    if valid_mask.dtype != torch.bool:
        raise ValueError('valid_mask must be bool, got %s'
                         % valid_mask.dtype)
    if projected.device != target.device \
            or projected.device != valid_mask.device:
        raise ValueError('device mismatch: projected %s, target %s, '
                         'valid_mask %s' % (projected.device,
                                            target.device,
                                            valid_mask.device))
    if not torch.isfinite(projected).all():
        raise ValueError('projected contains NaN or Inf')
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise ValueError('valid_mask selects an empty set')
    if not torch.isfinite(target.detach()[valid_mask]).all():
        raise ValueError('target contains NaN or Inf on a valid '
                         'position')
    return valid_count


def reconstruction_loss(projected, target, valid_mask):
    """Masked raw-Gram reconstruction loss (paper Eq. 20-21).

    Parameters
    ----------
    projected : torch.Tensor [n, d]
        Already-projected embeddings Z_proj (Eq. 19 output) for one
        modality. float32 or float64.
    target : torch.Tensor [n, n]
        Similarity matrix S to reconstruct. Same device/dtype as
        `projected`. Must be finite on positions where `valid_mask` is
        True; entries under invalid positions may be NaN/missing and are
        never read into arithmetic. Detached internally even if supplied
        with requires_grad=True — never receives gradient.
    valid_mask : torch.BoolTensor [n, n]
        Explicit coverage mask, at least one True. The caller owns the
        symmetry/diagonal/coverage policy; nothing is inferred from
        zero-valued or NaN entries.

    Returns
    -------
    dict with keys:
        'gram'        : [n, n] tensor, projected @ projected.T (Eq. 20
                        raw Gram — not cosine, unclipped, no forced
                        diagonal), full and differentiable
        'valid_count' : Python int
        'masked_sum'  : scalar, sum over valid cells of
                        (gram - target)^2 — masked squared Frobenius
        'masked_mean' : scalar, masked_sum / valid_count

    Masking is applied by indexed gather BEFORE subtraction, squaring or
    summation: invalid cells never enter arithmetic, so NaN there cannot
    affect the value and advanced-indexing backward scatters exactly
    zero cotangent to invalid gram cells — neither value nor gradient
    depends on invalid entries. Non-finite gram, masked_sum or
    masked_mean (overflow) is rejected with ValueError rather than
    silently emitted. The function draws no random numbers.
    """
    valid_count = _validate(projected, target, valid_mask)
    gram = projected @ projected.T
    if not torch.isfinite(gram).all():
        raise ValueError('gram overflowed dtype range; rejecting '
                         'rather than emitting Inf')
    residual = gram[valid_mask] - target.detach()[valid_mask]
    masked_sum = (residual * residual).sum()
    masked_mean = masked_sum / valid_count
    for name, v in (('masked_sum', masked_sum),
                    ('masked_mean', masked_mean)):
        if not torch.isfinite(v):
            raise ValueError('non-finite %s (loss overflow); rejecting '
                             'rather than emitting Inf' % name)
    return {
        'gram': gram,
        'valid_count': valid_count,
        'masked_sum': masked_sum,
        'masked_mean': masked_mean,
    }
