# -*- coding: utf-8 -*-
"""Pure paper §2.8 (Eq. 30-31) dynamic graph refresh (Stream D1).

Contract: see DYNAMIC_GRAPH_CONTRACT.md in this directory. Consumes
detached copies of the raw Grams produced by the accepted masked
reconstruction loss plus the existing four-relation edge_index_dict;
rebuilds binary similarity edges and KNN hypergraph Laplacians. No
trainer integration, no RNG, no file IO, no labels, no module-global
state. Self-contained: the Eq.4 Laplacian is reimplemented locally
(identical formula to the frozen runner's) rather than imported.
"""
import hashlib

import torch

_REL_SIM_MI = ('mir', 'sim', 'mir')
_REL_SIM_DIS = ('dis', 'sim', 'dis')
_REL_ASSOC = ('mir', 'assoc', 'dis')
_REL_REV = ('dis', 'rev_assoc', 'mir')
_RELATIONS = (_REL_SIM_MI, _REL_SIM_DIS, _REL_ASSOC, _REL_REV)
_G_KEYS = ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2')
_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SYM_RTOL = 1e-4
_SYM_ATOL = 1e-6
_UPDATE_PERIOD = 5


def _check_int(name, v, lo):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError('%s must be an int, got %r' % (name, v))
    if v < lo:
        raise ValueError('%s must be >= %d, got %d' % (name, lo, v))


def should_update(epoch, period=5):
    """End-of-epoch refresh schedule: True on positive multiples of
    period (epoch 0 False)."""
    _check_int('epoch', epoch, 0)
    _check_int('period', period, 1)
    return epoch > 0 and epoch % period == 0


def _check_gram(name, g):
    if not isinstance(g, torch.Tensor):
        raise TypeError('%s must be a torch tensor' % name)
    if g.dim() != 2 or g.shape[0] != g.shape[1]:
        raise ValueError('%s must be square 2-D [n, n], got %s'
                         % (name, tuple(g.shape)))
    if g.shape[0] == 0:
        raise ValueError('%s must be non-empty' % name)
    if g.dtype not in _SUPPORTED_DTYPES:
        raise ValueError('%s dtype must be float32/float64, got %s'
                         % (name, g.dtype))
    if g.device.type != 'cpu':
        raise ValueError('%s must be a CPU tensor, got %s'
                         % (name, g.device))
    g = g.detach()
    if not torch.isfinite(g).all():
        raise ValueError('%s contains NaN or Inf' % name)
    if not torch.allclose(g, g.T, rtol=_SYM_RTOL, atol=_SYM_ATOL):
        raise ValueError('%s is not symmetric within declared '
                         'tolerance' % name)
    return g.clone()


def _check_edges(edge_index_dict, n_mi, n_dis):
    if not isinstance(edge_index_dict, dict):
        raise TypeError('edge_index_dict must be a dict')
    if set(edge_index_dict.keys()) != set(_RELATIONS):
        raise ValueError('edge_index_dict must contain exactly the '
                         'four canonical relations')
    for rel in _RELATIONS:
        e = edge_index_dict[rel]
        if not isinstance(e, torch.Tensor):
            raise TypeError('%s edges must be a torch tensor' % (rel,))
        if e.dtype != torch.int64:
            raise ValueError('%s edges must be int64, got %s'
                             % (rel, e.dtype))
        if e.dim() != 2 or e.shape[0] != 2:
            raise ValueError('%s edges must be [2, E], got %s'
                             % (rel, tuple(e.shape)))
    bounds = {_REL_SIM_MI: (n_mi, n_mi), _REL_SIM_DIS: (n_dis, n_dis),
              _REL_ASSOC: (n_mi, n_dis), _REL_REV: (n_dis, n_mi)}
    for rel, (hi_s, hi_d) in bounds.items():
        e = edge_index_dict[rel]
        if e.numel() and (e.min() < 0 or e[0].max() >= hi_s
                          or e[1].max() >= hi_d):
            raise ValueError('%s has out-of-bounds endpoints' % (rel,))
    a, r = (edge_index_dict[_REL_ASSOC].detach(),
            edge_index_dict[_REL_REV].detach())
    if a.shape != r.shape or not torch.equal(a[0], r[1]) \
            or not torch.equal(a[1], r[0]):
        raise ValueError('rev_assoc edges must equal assoc edges '
                         'reversed column-wise, order preserved')


def _normalize_offdiag_minmax(g):
    """Off-diagonal minmax to [0,1] with abs-max prescale; constant or
    empty off-diag -> 0; diagonal set to 1 (center convention).
    Computed in float64, returned in the input dtype."""
    n = g.shape[0]
    gd = g.detach().double()
    off_mask = ~torch.eye(n, dtype=torch.bool)
    out = torch.zeros(n, n, dtype=torch.float64)
    if n > 1:
        off = gd[off_mask]
        amax = off.abs().max()
        if amax > 0:
            x = off / amax
            lo, hi = x.min(), x.max()
            if hi > lo:
                out[off_mask] = (x - lo) / (hi - lo)
    out.fill_diagonal_(1.0)
    return out.to(g.dtype)


def _sim_edges(norm, threshold):
    """Directed ordered pairs (i,j), i!=j, strict score>threshold,
    row-major (src asc, dst asc), int64 [2,E]."""
    n = norm.shape[0]
    nrm = norm.double()
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and nrm[i, j].item() > threshold:
                src.append(i)
                dst.append(j)
    if not src:
        return torch.empty(2, 0, dtype=torch.int64)
    return torch.tensor([src, dst], dtype=torch.int64)


def _knn_incidence(norm, k):
    """Column c = center c's hyperedge: top min(k, n-1) OTHER nodes by
    descending normalized score (ties -> ascending node index) + self.
    Direct ranking on the normalized similarity, not a second cosine."""
    n = norm.shape[0]
    kk = min(k, n - 1)
    nrm = norm.double()
    H = torch.zeros(n, n, dtype=torch.float64)
    for c in range(n):
        others = sorted((j for j in range(n) if j != c),
                        key=lambda j: (-nrm[c, j].item(), j))[:kk]
        for m in others + [c]:
            H[m, c] = 1.0
    return H


def _hypergraph_laplacian(H):
    """G = Dv^-1/2 H W De^-1 H^T Dv^-1/2, W=I (author Eq.4, identical
    to frozen runner.hypergraph_laplacian). Zero-degree rows -> zero.
    Computed in float64."""
    Hd = H.double()
    dv = Hd.sum(dim=1)
    de = Hd.sum(dim=0)
    dv2 = torch.where(dv > 0, dv.pow(-0.5), torch.zeros_like(dv))
    inv_de = torch.where(de > 0, 1.0 / de, torch.zeros_like(de))
    return (dv2[:, None] * Hd * inv_de[None, :]) @ Hd.T * dv2[None, :]


def _sha(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def refresh_similarity_graphs(gram_mi, gram_dis, edge_index_dict, epoch,
                              k=13, threshold=0.5):
    """Refresh similarity edges and KNN hypergraph Laplacians from the
    reconstructed Grams (paper §2.8 Eq. 30-31).

    Returns dict with normalized_mi/normalized_dis (off-diag minmax
    [0,1], diag 1 center convention), incidence_mi/incidence_dis (0/1
    KNN hyperedges), G = {mi_v1,mi_v2,dis_v1,dis_v2} (shared per
    modality, separate storage), a new edge_index_dict (refreshed sim
    edges + cloned assoc/rev in canonical order), and meta provenance
    (epoch, period, k/k_eff, threshold, policies, sha256 hashes, edge
    counts, sim-edge turnover diagnostics). All outputs detached; no
    RNG, mutation, IO or labels.
    """
    _check_int('epoch', epoch, 1)
    if epoch % _UPDATE_PERIOD != 0:
        raise ValueError('epoch %d is not a positive multiple of the '
                         'fixed update period %d' % (epoch,
                                                     _UPDATE_PERIOD))
    _check_int('k', k, 1)
    if isinstance(threshold, bool) \
            or not isinstance(threshold, (int, float)):
        raise TypeError('threshold must be a real number')
    if not (0.0 <= float(threshold) <= 1.0
            and float(threshold) == float(threshold)):
        raise ValueError('threshold must be in [0, 1], got %r'
                         % threshold)
    t = float(threshold)
    g_mi = _check_gram('gram_mi', gram_mi)
    g_dis = _check_gram('gram_dis', gram_dis)
    n_mi, n_dis = g_mi.shape[0], g_dis.shape[0]
    _check_edges(edge_index_dict, n_mi, n_dis)

    norm_mi = _normalize_offdiag_minmax(g_mi)
    norm_dis = _normalize_offdiag_minmax(g_dis)
    edges_mi = _sim_edges(norm_mi, t)
    edges_dis = _sim_edges(norm_dis, t)
    H_mi = _knn_incidence(norm_mi, k)
    H_dis = _knn_incidence(norm_dis, k)
    G_mi = _hypergraph_laplacian(H_mi).to(g_mi.dtype)
    G_dis = _hypergraph_laplacian(H_dis).to(g_dis.dtype)
    G = {'mi_v1': G_mi, 'mi_v2': G_mi.clone(),
         'dis_v1': G_dis, 'dis_v2': G_dis.clone()}
    H_mi = H_mi.to(g_mi.dtype)
    H_dis = H_dis.to(g_dis.dtype)
    new_ei = {
        _REL_SIM_MI: edges_mi,
        _REL_SIM_DIS: edges_dis,
        _REL_ASSOC: edge_index_dict[_REL_ASSOC].detach().clone(),
        _REL_REV: edge_index_dict[_REL_REV].detach().clone(),
    }
    old_mi = set(map(tuple,
                     edge_index_dict[_REL_SIM_MI].detach().T.tolist()))
    old_dis = set(map(tuple,
                      edge_index_dict[_REL_SIM_DIS].detach().T.tolist()))
    new_mi = set(map(tuple, edges_mi.T.tolist()))
    new_dis = set(map(tuple, edges_dis.T.tolist()))
    meta = {
        'epoch': epoch,
        'period': _UPDATE_PERIOD,
        'k': k,
        'k_eff_mi': min(k, n_mi - 1),
        'k_eff_dis': min(k, n_dis - 1),
        'threshold': t,
        'normalization': 'offdiag_minmax (abs-max prescale; '
                         'constant/empty offdiag -> 0)',
        'sharing': 'modality-shared G values, separate per-view '
                   'storage (declared engineering choice)',
        'diag_convention': 'normalized diagonal fixed to 1.0; raw '
                           'Grams untouched',
        'n_mi': n_mi,
        'n_dis': n_dis,
        'dtype_mi': str(g_mi.dtype),
        'dtype_dis': str(g_dis.dtype),
        'edges_sim_mi': int(edges_mi.shape[1]),
        'edges_sim_dis': int(edges_dis.shape[1]),
        'edges_assoc': int(new_ei[_REL_ASSOC].shape[1]),
        'edges_rev_assoc': int(new_ei[_REL_REV].shape[1]),
        'sim_mi_edges_added': len(new_mi - old_mi),
        'sim_mi_edges_removed': len(old_mi - new_mi),
        'sim_dis_edges_added': len(new_dis - old_dis),
        'sim_dis_edges_removed': len(old_dis - new_dis),
        'normalized_mi_sha256': _sha(norm_mi),
        'normalized_dis_sha256': _sha(norm_dis),
        'incidence_mi_sha256': _sha(H_mi),
        'incidence_dis_sha256': _sha(H_dis),
        'G_mi_v1_sha256': _sha(G['mi_v1']),
        'G_mi_v2_sha256': _sha(G['mi_v2']),
        'G_dis_v1_sha256': _sha(G['dis_v1']),
        'G_dis_v2_sha256': _sha(G['dis_v2']),
        'edges_sim_mi_sha256': _sha(edges_mi),
        'edges_sim_dis_sha256': _sha(edges_dis),
        'edges_assoc_sha256': _sha(new_ei[_REL_ASSOC]),
        'edges_rev_assoc_sha256': _sha(new_ei[_REL_REV]),
    }
    return {
        'normalized_mi': norm_mi,
        'normalized_dis': norm_dis,
        'incidence_mi': H_mi,
        'incidence_dis': H_dis,
        'G': G,
        'edge_index_dict': new_ei,
        'meta': meta,
    }
