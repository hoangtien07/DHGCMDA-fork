# -*- coding: utf-8 -*-
"""Pure inter-modality (cross-view) contrastive loss — paper Eq. 10-15.

Contract: see INTER_CONTRACT.md in this directory. Consumes fused
PRE-HGT embeddings ZM/ZD and explicit boolean positive/negative masks;
contains no parameters, no IO, no label access, and draws only from a
caller-seeded LOCAL generator inside the bank sampler (the loss itself
is fully deterministic).
"""
import torch

_NORM_EPS = 1e-12
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _norm_rows(z):
    """Abs-max-prescaled L2 row normalization (same convention as
    intra.py): exact direction for every finite nonzero row incl.
    float32-overflowing magnitudes; zero rows stay zero."""
    scale = z.abs().amax(dim=1, keepdim=True).clamp_min(_NORM_EPS)
    zs = z / scale
    return zs / zs.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)


def _as_bool_mask(m, name):
    if not isinstance(m, torch.Tensor):
        raise TypeError('%s must be a torch.Tensor' % name)
    if m.dim() != 2:
        raise ValueError('%s must be 2-D [n_mi, n_dis]' % name)
    if m.dtype == torch.bool:
        return m
    if m.dtype in (torch.float32, torch.float64):
        # 0/1 float masks acceptable ONLY if every entry is exactly 0/1
        if not torch.isfinite(m).all() or \
                not ((m == 0) | (m == 1)).all():
            raise ValueError('%s float mask must contain only 0/1'
                             % name)
        return m.bool()
    if m.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8):
        if not ((m == 0) | (m == 1)).all():
            raise ValueError('%s integer mask must contain only 0/1'
                             % name)
        return m.bool()
    raise ValueError('%s must be bool or 0/1 mask, got %s'
                     % (name, m.dtype))


def _finite_pos(v, name):
    if not isinstance(v, (int, float)):
        raise TypeError('%s must be a real number' % name)
    if not (v == v and v != float('inf') and v != float('-inf')):
        raise ValueError('%s must be finite, got %r' % (name, v))


def _validate(zm, zd, pos_mask, neg_mask, temperature, margin,
              margin_coef):
    for name, z in (('zm', zm), ('zd', zd)):
        if not isinstance(z, torch.Tensor):
            raise TypeError('%s must be a torch.Tensor' % name)
        if z.dim() != 2:
            raise ValueError('%s must be 2-D, got %d dims'
                             % (name, z.dim()))
        if z.shape[0] == 0 or z.shape[1] == 0:
            raise ValueError('%s must be non-empty [n, d]' % name)
        if z.dtype not in _SUPPORTED_DTYPES:
            raise ValueError('%s dtype must be float32/float64, got %s'
                             % (name, z.dtype))
        if not torch.isfinite(z).all():
            raise ValueError('%s contains NaN or Inf' % name)
    if zm.dtype != zd.dtype:
        raise ValueError('dtype mismatch: %s vs %s'
                         % (zm.dtype, zd.dtype))
    if zm.device != zd.device:
        raise ValueError('device mismatch: %s vs %s'
                         % (zm.device, zd.device))
    if zm.shape[1] != zd.shape[1]:
        raise ValueError('feature width mismatch: %d vs %d'
                         % (zm.shape[1], zd.shape[1]))
    pm = _as_bool_mask(pos_mask, 'pos_mask')
    nm = _as_bool_mask(neg_mask, 'neg_mask')
    for name, m in (('pos_mask', pm), ('neg_mask', nm)):
        if m.device != zm.device:
            raise ValueError('%s device %s != zm device %s'
                             % (name, m.device, zm.device))
    want = (zm.shape[0], zd.shape[0])
    if pm.shape != want:
        raise ValueError('pos_mask shape %s != (n_mi,n_dis) %s'
                         % (tuple(pm.shape), want))
    if nm.shape != want:
        raise ValueError('neg_mask shape %s != (n_mi,n_dis) %s'
                         % (tuple(nm.shape), want))
    if (pm & nm).any():
        raise ValueError('pos_mask and neg_mask overlap: a cell cannot '
                         'be both positive and negative')
    _finite_pos(temperature, 'temperature')
    if temperature <= 0:
        raise ValueError('temperature must be positive, got %r'
                         % temperature)
    _finite_pos(margin, 'margin')
    if margin < 0:
        raise ValueError('margin must be >= 0, got %r' % margin)
    _finite_pos(margin_coef, 'margin_coef')
    if margin_coef < 0:
        raise ValueError('margin_coef must be >= 0, got %r'
                         % margin_coef)
    return pm, nm


def inter_modality_loss(zm, zd, pos_mask, neg_mask, temperature=0.5,
                        margin=0.5, margin_coef=0.1):
    """Cross-modality contrastive loss (paper Eq. 10-15).

    zm [n_mi,d], zd [n_dis,d]: fused PRE-HGT embeddings (Eq. 10 input).
    pos_mask/neg_mask: bool [n_mi, n_dis], disjoint. Denominator for
    each positive (i,j): {S_ij} + {S_ik : neg_mask[i,k]} — other
    positives excluded (Eq. 13). Margin = relu(margin - (mean_pos_sim -
    mean_neg_sim)) over raw S_cross at masked cells (Eq. 14).

    Returns dict with per_positive, info_sum, info_mean, pos_mean_sim,
    neg_mean_sim, margin_term, margin_coef, l_inter, n_pos, n_neg.
    l_inter = info_mean + margin_coef*margin_term (inner Eq. 15; the
    outer lam_inter is applied once by the caller). Empty pos or empty
    neg -> differentiable zero terms. All mean reductions are declared
    normalized engineering units.
    """
    pm, nm = _validate(zm, zd, pos_mask, neg_mask, temperature,
                       margin, margin_coef)
    t = float(temperature)
    n_pos = int(pm.sum().item())
    n_neg = int(nm.sum().item())

    # Branch on mask counts BEFORE any similarity work (also keeps
    # boundary paths independent of S magnitude). Live zero built from
    # single finite ELEMENTS (validated finite), so huge-but-finite
    # inputs cannot overflow it the way sum()*0 can.
    z0 = zm.reshape(-1)[0] * 0.0 + zd.reshape(-1)[0] * 0.0
    if n_pos == 0 or n_neg == 0:
        # Boundary: no InfoNCE anchor or no margin/denominator terms.
        # per_positive is a differentiable zero vector of length n_pos
        # (empty when n_pos==0) so per_positive.sum()==info_sum==0 on
        # BOTH boundaries. Every scalar is a live finite zero.
        per_pos = (z0.expand(n_pos) if n_pos > 0 else
                   torch.zeros(0, dtype=zm.dtype, device=zm.device)
                   + z0)
        return {'per_positive': per_pos,
                'info_sum': z0, 'info_mean': z0,
                'pos_mean_sim': z0, 'neg_mean_sim': z0,
                'margin_term': z0, 'margin_coef': float(margin_coef),
                'l_inter': z0, 'n_pos': n_pos, 'n_neg': n_neg}

    zn_m = _norm_rows(zm)
    zn_d = _norm_rows(zd)
    S = zn_m @ zn_d.T / t                          # [n_mi, n_dis]

    # InfoNCE per positive: denom_i = exp(S_ij) + sum_{k in neg_i}
    # exp(S_ik); l_ij = logaddexp(S_ij, lse over that row's negs).
    pos_rows = pm.any(dim=1)
    S_rows = S[pos_rows]                              # [k, n_dis]
    pm_rows = pm[pos_rows]
    nm_rows = nm[pos_rows]
    lse_rows = torch.logsumexp(
        torch.where(nm_rows, S_rows,
                    torch.full_like(S_rows, float('-inf'))),
        dim=1)                                        # [k]
    # per-positive: l_ij = log(denom) - S_ij, denom = e^S_ij + sum_negs
    lse_per_pos = torch.logaddexp(
        S_rows[pm_rows],
        lse_rows.unsqueeze(1).expand_as(S_rows)[pm_rows])
    per_positive = lse_per_pos - S_rows[pm_rows]

    info_sum = per_positive.sum()
    info_mean = info_sum / n_pos
    pos_mean_sim = S[pm].mean()
    neg_mean_sim = S[nm].mean()
    margin_term = torch.relu(
        float(margin) - (pos_mean_sim - neg_mean_sim))
    l_inter = info_mean + float(margin_coef) * margin_term

    for name, v in (('info_sum', info_sum), ('info_mean', info_mean),
                    ('pos_mean_sim', pos_mean_sim),
                    ('neg_mean_sim', neg_mean_sim),
                    ('margin_term', margin_term), ('l_inter', l_inter)):
        if not torch.isfinite(v):
            raise ValueError('non-finite aggregate %s under declared '
                             'temperature; rejecting rather than '
                             'emitting Inf/NaN' % name)
    return {'per_positive': per_positive,
            'info_sum': info_sum, 'info_mean': info_mean,
            'pos_mean_sim': pos_mean_sim,
            'neg_mean_sim': neg_mean_sim,
            'margin_term': margin_term,
            'margin_coef': float(margin_coef),
            'l_inter': l_inter, 'n_pos': n_pos, 'n_neg': n_neg}


def sample_negative_bank(pos_mask, seed, cap_multiplier=10):
    """Static per-anchor negative bank over A_train-zero cells.

    For each anchor row i: candidates = zero cells of pos_mask;
    k = min(n_candidates, cap_multiplier * n_pos_i) sampled uniformly
    WITHOUT replacement via an independent torch.Generator seeded by
    `seed` — global RNG untouched, bank bit-reproducible. Anchors with
    no positives sample nothing (k=0). Returns bool [n_mi, n_dis],
    asserted disjoint from pos_mask.

    These cells are ASSUMED negatives (unknown under train labels),
    not experimentally verified negatives — false-negative risk is
    disclosed by callers in manifests/reports.
    """
    pm = _as_bool_mask(pos_mask, 'pos_mask')
    if not isinstance(seed, int):
        raise TypeError('seed must be an int')
    if not (isinstance(cap_multiplier, (int, float))
            and cap_multiplier >= 0):
        raise ValueError('cap_multiplier must be >= 0')
    n_mi, n_dis = pm.shape
    g = torch.Generator().manual_seed(int(seed))
    bank = torch.zeros_like(pm)
    for i in range(n_mi):
        cand = torch.nonzero(~pm[i], as_tuple=False).flatten()
        n_pos_i = int(pm[i].sum().item())
        k = min(int(cand.numel()),
                int(cap_multiplier * n_pos_i))
        if k <= 0:
            continue
        perm = torch.randperm(cand.numel(), generator=g)[:k]
        bank[i, cand[perm]] = True
    if (bank & pm).any():                            # pragma: no cover
        raise AssertionError('internal: bank overlaps positives')
    return bank
