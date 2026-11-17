# -*- coding: utf-8 -*-
"""Pure reconstruction target/mask builder (R1 preparation stage).

Contract: see R1_TARGET_DECODER_CONTRACT.md in this directory. Slices
the declared leading similarity blocks out of already-built augmented
views, derives coverage masks from train association support (miRNA)
and the declared descriptor audit (disease), and emits off-diagonal
valid cell masks plus provenance/count/hash metadata. No file IO, no
RNG, no labels, no pair lists, no mutation of inputs.
"""
import hashlib
from collections.abc import Mapping

import numpy as np

_VIEW_KEYS = ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2')
_SYM_ATOL = 1e-6


def _fail(msg):
    raise ValueError(msg)


def _require_mapping(x, name):
    if not isinstance(x, Mapping):
        raise TypeError('%s must be a mapping' % name)


def _validate(views, train_mask, disease_names, descriptor_audit):
    _require_mapping(views, 'views')
    if set(views.keys()) != set(_VIEW_KEYS):
        _fail('views must contain exactly keys %s, got %s'
              % (_VIEW_KEYS, sorted(views.keys())))
    if not isinstance(train_mask, np.ndarray):
        raise TypeError('train_mask must be a numpy array')
    if train_mask.dtype != np.bool_ or train_mask.ndim != 2:
        _fail('train_mask must be a 2-D bool array')
    n_mi, n_dis = train_mask.shape
    if n_mi == 0 or n_dis == 0:
        _fail('train_mask must be non-empty in both dims')
    for k in _VIEW_KEYS:
        v = views[k]
        if not isinstance(v, np.ndarray):
            raise TypeError('views[%r] must be a numpy array' % k)
        if v.ndim != 2:
            _fail('views[%r] must be 2-D' % k)
        if not np.issubdtype(v.dtype, np.floating):
            _fail('views[%r] must have a float dtype, got %s'
                  % (k, v.dtype))
    for k in ('mi_v1', 'mi_v2'):
        if views[k].shape != (n_mi, n_mi + n_dis):
            _fail('views[%r] shape must be (%d, %d), got %s'
                  % (k, n_mi, n_mi + n_dis, views[k].shape))
    for k in ('dis_v1', 'dis_v2'):
        if views[k].shape != (n_dis, n_dis + n_mi):
            _fail('views[%r] shape must be (%d, %d), got %s'
                  % (k, n_dis, n_dis + n_mi, views[k].shape))
    # exact binary association tails — guards the 411/271 slice order
    for k, want in (('mi_v1', train_mask), ('mi_v2', train_mask),
                    ('dis_v1', train_mask.T), ('dis_v2', train_mask.T)):
        lead = n_mi if k.startswith('mi') else n_dis
        tail = views[k][:, lead:]
        if not np.isin(tail, (0.0, 1.0)).all():
            _fail('views[%r] association tail must be binary' % k)
        if not np.array_equal(tail != 0, want):
            _fail('views[%r] association tail does not equal the '
                  'declared train mask' % k)
    names = list(disease_names)
    if len(names) != n_dis or any(not isinstance(s, str)
                                  for s in names):
        _fail('disease_names must be %d strings' % n_dis)
    _require_mapping(descriptor_audit, 'descriptor_audit')
    try:
        a_n_dis = descriptor_audit['n_dis']
        a_n_unc = descriptor_audit['n_uncovered_rows']
        a_rows = descriptor_audit['uncovered_rows']
    except KeyError as e:
        _fail('descriptor_audit missing key %s' % e)
    if a_n_dis != n_dis:
        _fail('descriptor_audit n_dis %r does not match mask width %d'
              % (a_n_dis, n_dis))
    if not isinstance(a_rows, (list, tuple)) or len(a_rows) != a_n_unc:
        _fail('descriptor_audit uncovered_rows length does not match '
              'n_uncovered_rows')
    uncovered = []
    seen = set()
    for e in a_rows:
        if not isinstance(e, Mapping) or 'row' not in e \
                or 'name' not in e:
            _fail('descriptor_audit uncovered_rows entries need '
                  'row and name')
        r, nm = e['row'], e['name']
        if not isinstance(r, (int, np.integer)) or isinstance(r, bool):
            _fail('descriptor_audit row %r is not an int' % (r,))
        if not (0 <= int(r) < n_dis):
            _fail('descriptor_audit row %d out of bounds' % int(r))
        if int(r) in seen:
            _fail('descriptor_audit duplicate row %d' % int(r))
        seen.add(int(r))
        if nm != names[int(r)]:
            _fail('descriptor_audit name %r at row %d does not equal '
                  'disease_names[%d]=%r' % (nm, int(r), int(r),
                                            names[int(r)]))
        uncovered.append(int(r))
    return n_mi, n_dis, names, uncovered


def _target(name, block):
    if not np.isfinite(block).all():
        _fail('%s target contains NaN or Inf' % name)
    if np.abs(block - block.T).max() > _SYM_ATOL:
        _fail('%s target is not symmetric within atol=%g'
              % (name, _SYM_ATOL))
    return np.array(block, copy=True)


def _valid_mask(coverage, name):
    n = coverage.shape[0]
    cell = coverage[:, None] & coverage[None, :]
    cell &= ~np.eye(n, dtype=bool)          # diagonal never supervised
    if not cell.any():
        _fail('%s valid cell set is empty' % name)
    return cell


def build_reconstruction_targets(views, train_mask, disease_names,
                                 descriptor_audit):
    """Build masked reconstruction targets + coverage metadata.

    Parameters
    ----------
    views : mapping
        Exactly keys 'mi_v1','mi_v2','dis_v1','dis_v2'; each a 2-D
        float ndarray of the augmented [similarity | association]
        layout (mi views [n_mi, n_mi+n_dis], dis views
        [n_dis, n_dis+n_mi]). Association tails must be binary and
        exactly equal the declared train mask / its transpose.
    train_mask : np.ndarray [n_mi, n_dis] bool
        Fold-train association mask — the only supervision-derived
        input; controls miRNA coverage.
    disease_names : sequence of str, length n_dis
        Row-aligned disease names (entity_names['dis']).
    descriptor_audit : mapping
        Declared Task3 descriptor audit with 'n_dis',
        'n_uncovered_rows' and 'uncovered_rows' entries {row, name}.
        Disease coverage = all rows except these indices; never
        inferred from zero rows.

    Returns
    -------
    dict with copied targets 'mi_target' (= mi_v2[:, :n_mi], train
    functional similarity) and 'dis_target' (= dis_v2[:, :n_dis], MeSH
    semantic similarity), 'mi_row_coverage'/'dis_row_coverage' [n] bool,
    'mi_valid_mask'/'dis_valid_mask' [n,n] bool (covered row AND
    covered column AND off-diagonal), counts, and 'meta' provenance
    (identities, shapes, dtypes, sha256 of targets/masks/train_mask,
    uncovered echo, symmetry atol).

    Rejects TypeError/ValueError on malformed views/mask/names/audit,
    tail drift, non-finite or asymmetric (atol 1e-6) targets, and empty
    valid sets. Does no IO, draws no random numbers, mutates no input.
    """
    n_mi, n_dis, names, uncovered = _validate(
        views, train_mask, disease_names, descriptor_audit)
    mi_target = _target('mi_v2 leading', views['mi_v2'][:, :n_mi])
    dis_target = _target('dis_v2 leading', views['dis_v2'][:, :n_dis])
    mi_cov = train_mask.any(axis=1)
    dis_cov = np.ones(n_dis, dtype=bool)
    dis_cov[np.asarray(uncovered, dtype=int)] = False
    mi_valid = _valid_mask(mi_cov, 'miRNA')
    dis_valid = _valid_mask(dis_cov, 'disease')
    sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
    return {
        'mi_target': mi_target,
        'dis_target': dis_target,
        'mi_row_coverage': mi_cov,
        'dis_row_coverage': dis_cov,
        'mi_valid_mask': mi_valid,
        'dis_valid_mask': dis_valid,
        'mi_valid_count': int(mi_valid.sum()),
        'dis_valid_count': int(dis_valid.sum()),
        'mi_covered_count': int(mi_cov.sum()),
        'dis_covered_count': int(dis_cov.sum()),
        'meta': {
            'mi_target_id': 'mi_v2[:, :n_mi] train functional '
                            'similarity (S_func, train-mask built)',
            'dis_target_id': 'dis_v2[:, :n_dis] MeSH-2019 semantic '
                             'similarity (S_sem)',
            'mi_coverage_policy': 'train_mask.any(axis=1)',
            'dis_coverage_policy': 'descriptor_audit complement '
                                   '(declared, not zero-row inferred)',
            'diagonal_policy': 'always excluded (fixed to 1.0 in '
                               'feature preparation)',
            'n_mi': n_mi,
            'n_dis': n_dis,
            'mi_target_dtype': str(mi_target.dtype),
            'dis_target_dtype': str(dis_target.dtype),
            'mi_target_sha256': sha(mi_target),
            'dis_target_sha256': sha(dis_target),
            'mi_valid_mask_sha256': sha(mi_valid),
            'dis_valid_mask_sha256': sha(dis_valid),
            'train_mask_sha256': sha(np.ascontiguousarray(train_mask)),
            'dis_uncovered_rows': uncovered,
            'symmetry_atol': _SYM_ATOL,
        },
    }
