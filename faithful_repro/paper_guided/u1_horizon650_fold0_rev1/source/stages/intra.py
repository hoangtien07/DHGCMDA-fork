# -*- coding: utf-8 -*-
"""Pure intra-modality contrastive loss (paper Eq. 6-9).

Contract: see INTRA_CONTRACT.md in this directory. Consumes already
projected embeddings [n, d]; contains no parameters, no RNG, no IO.
"""
import torch

_NORM_EPS = 1e-12
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _validate(z1, z2, temperature, alpha):
    if not (isinstance(z1, torch.Tensor) and isinstance(z2, torch.Tensor)):
        raise TypeError('z1 and z2 must be torch tensors')
    for name, z in (('z1', z1), ('z2', z2)):
        if z.dim() != 2:
            raise ValueError('%s must be 2-D [n, d], got %d dims'
                             % (name, z.dim()))
        if z.shape[0] == 0 or z.shape[1] == 0:
            raise ValueError('%s must be non-empty [n, d]' % name)
        if z.dtype not in _SUPPORTED_DTYPES:
            raise ValueError('%s dtype must be float32/float64, got %s'
                             % (name, z.dtype))
        if not torch.isfinite(z).all():
            raise ValueError('%s contains NaN or Inf' % name)
    if z1.shape != z2.shape:
        raise ValueError('shape mismatch: %s vs %s'
                         % (tuple(z1.shape), tuple(z2.shape)))
    if z1.dtype != z2.dtype:
        raise ValueError('dtype mismatch: %s vs %s'
                         % (z1.dtype, z2.dtype))
    if z1.device != z2.device:
        raise ValueError('device mismatch: %s vs %s'
                         % (z1.device, z2.device))
    if not isinstance(temperature, (int, float)):
        raise TypeError('temperature must be a real number')
    if not (temperature > 0 and temperature == temperature
            and temperature != float('inf')):
        raise ValueError('temperature must be positive and finite, got %r'
                         % temperature)
    if not isinstance(alpha, (int, float)):
        raise TypeError('alpha must be a real number')
    if not (0.0 <= alpha <= 1.0 and alpha == alpha):
        raise ValueError('alpha must be in [0, 1], got %r' % alpha)


def _direction_loss(anchor_norm, other_norm, temperature):
    """Per-anchor Eq. 6-7 loss: anchor rows are A[i], counterpart
    positive is B[i]; denominator over all B[j] plus all A[j], j != i.
    Returns [n] per-anchor losses (already includes -theta_pos).
    """
    n = anchor_norm.shape[0]
    cross = anchor_norm @ other_norm.T / temperature      # [n, n]
    same = anchor_norm @ anchor_norm.T / temperature       # [n, n]
    same.fill_diagonal_(float('-inf'))                     # exclude self
    logits = torch.cat([cross, same], dim=1)               # [n, 2n-1]
    denom = torch.logsumexp(logits, dim=1)                 # [n]
    pos = torch.diagonal(cross)                            # [n] theta_pos/t
    losses = denom - pos
    if not torch.isfinite(losses).all():
        raise ValueError('non-finite per-anchor loss under declared '
                         'temperature; rejecting rather than emitting '
                         'NaN/Inf')
    return losses


def _norm_rows(z):
    """Robust L2 row normalization: rescale by abs-max first so that
    finite-but-huge rows (norm overflowing the dtype range, e.g.
    [1e38, 1e38] float32) still yield the mathematically correct
    direction instead of collapsing silently to a zero row. A
    positive per-row scale factor cancels in cosine similarity, so
    this is exact for every nonzero finite row; zero rows remain
    zero under the 1e-12 clamp convention."""
    scale = z.abs().amax(dim=1, keepdim=True).clamp_min(_NORM_EPS)
    zs = z / scale                    # |zs| <= 1, zero rows stay zero
    return zs / zs.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)


def intra_view_loss(z1, z2, temperature=0.5, alpha=0.5):
    """Bidirectional intra-modality contrastive loss (paper Eq. 6-9).

    Parameters
    ----------
    z1, z2 : torch.Tensor [n, d]
        Projected embeddings for the same n entities from two views.
        Same shape, device, dtype (float32 or float64).
    temperature : float
        Positive finite temperature t (Eq. 6-7). Default 0.5.
    alpha : float in [0, 1]
        Weight of the z1-anchor direction (Eq. 9). Default 0.5.

    Returns
    -------
    dict with keys:
        'per_anchor_forward' : [n] losses for anchors in z1 (Eq. 6-7)
        'per_anchor_reverse' : [n] losses for anchors in z2
        'forward_sum'        : sum_i l_f(i)
        'reverse_sum'        : sum_i l_r(i)
        'sum'                : alpha*forward_sum + (1-alpha)*reverse_sum
        'mean'               : sum / n

    Conventions: zero-norm rows are normalized via clamp_min(1e-12),
    which yields all-zero logits; an ALL-ZERO ANCHOR row gets loss
    log(2n-1). Note this convention is per-anchor only — a zero
    NON-anchor row still contributes exp(0)=1 terms to every other
    anchor's denominator. n=1 gives an exact zero loss with zero
    gradients. Gradients flow to both inputs; the function draws no
    random numbers. Non-finite AGGREGATE values (forward_sum,
    reverse_sum, sum, mean) are rejected with ValueError even when
    per-anchor losses remain finite (e.g. extreme temperature).
    """
    _validate(z1, z2, temperature, alpha)
    t = float(temperature)
    z1n = _norm_rows(z1)
    z2n = _norm_rows(z2)
    lf = _direction_loss(z1n, z2n, t)
    lr = _direction_loss(z2n, z1n, t)
    n = z1.shape[0]
    forward_sum = lf.sum()
    reverse_sum = lr.sum()
    total = alpha * forward_sum + (1.0 - alpha) * reverse_sum
    mean = total / n
    for name, v in (('forward_sum', forward_sum),
                    ('reverse_sum', reverse_sum),
                    ('sum', total), ('mean', mean)):
        if not torch.isfinite(v):
            raise ValueError('non-finite aggregate loss %s under '
                             'declared temperature; rejecting rather '
                             'than emitting Inf' % name)
    return {
        'per_anchor_forward': lf,
        'per_anchor_reverse': lr,
        'forward_sum': forward_sum,
        'reverse_sum': reverse_sum,
        'sum': total,
        'mean': total / n,
    }
