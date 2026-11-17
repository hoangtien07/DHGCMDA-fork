# -*- coding: utf-8 -*-
"""Fold-label contract for the paper-guided track.

`train_associations` is the ONLY path by which label information may enter
feature building. It accepts the multi-hot label tensor and an explicit
train-pair list and returns a boolean train-only association mask plus
per-channel train counts. Callers pass the mask onward; feature builders
never receive Y or held-out indices, which makes held-out-label access
structurally impossible rather than policy-based.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainAssociations:
    """Train-only association view.

    mask:           (n_mirna, n_disease) bool — True only at train pairs.
    channel_counts: (n_channel,) int — positive-label count per type
                    channel, computed on train pairs ONLY.
    n_pairs:        number of train pairs (mask True count).
    """
    mask: np.ndarray
    channel_counts: np.ndarray
    n_pairs: int


def train_associations(Y, train_pairs):
    """Build the train-only association mask from Y and explicit pairs.

    Y:           (n_mirna, n_disease, C) multi-hot labels (any dtype).
    train_pairs: iterable of (mi, di) integer index pairs belonging to the
                 current fold's TRAIN segment. Held-out pairs must not be
                 included; this function does not validate fold identity —
                 the caller owns the split contract.

    Returns TrainAssociations. The mask has True exactly at train pairs
    where any channel is positive (train pairs are positive by contract;
    asserted). Channel counts count positive labels per channel across
    train pairs only.
    """
    Y = np.asarray(Y)
    if Y.ndim != 3:
        raise ValueError('Y must be (n_mirna, n_disease, n_channel)')
    if not np.isfinite(Y.astype(float)).all() \
            or not np.isin(Y, (0, 1, 0.0, 1.0)).all():
        raise ValueError('Y must be finite binary multi-hot')
    n_m, n_d, _ = Y.shape
    raw = np.asarray(list(train_pairs))
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
        raise ValueError('train_pairs must be a non-empty (n,2) list')
    if not np.isfinite(raw.astype(float)).all() \
            or (raw.astype(float) % 1 != 0).any():
        raise ValueError('train_pairs must be integer indices')
    pairs = raw.astype(np.int64)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('duplicate train pairs are rejected')
    if not ((pairs[:, 0] >= 0).all() and (pairs[:, 0] < n_m).all()
            and (pairs[:, 1] >= 0).all() and (pairs[:, 1] < n_d).all()):
        raise ValueError('train_pairs contain out-of-range indices')
    mask = np.zeros((n_m, n_d), dtype=bool)
    mask[pairs[:, 0], pairs[:, 1]] = True
    if not (Y[mask] != 0).any(axis=1).all():
        raise ValueError('train pair with all-zero label channels — '
                         'violates the positive-pair contract')
    counts = (Y[mask] != 0).sum(axis=0).astype(np.int64)
    return TrainAssociations(mask=mask, channel_counts=counts,
                             n_pairs=int(mask.sum()))


def multihot(Y, pairs):
    """(n_pairs, C) multi-hot label rows for an explicit pair list."""
    Y = np.asarray(Y)
    pairs = np.asarray(list(pairs), dtype=np.int64).reshape(-1, 2)
    return (Y[pairs[:, 0], pairs[:, 1]] != 0).astype(np.float64)


def class_weights_train(y_multi, beta=0.99999):
    """Effective-count class weights from TRAIN multi-hot rows only.

    w_k = (1 - beta) / (1 - beta**n_k), n_k = train positives of type k,
    then scaled so sum_k w_k = C (declared normalization). Channels with
    n_k == 0 get weight 0 (they contribute nothing anywhere anyway).
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError('beta must be in [0, 1)')
    y = np.asarray(y_multi, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] == 0:
        raise ValueError('y_multi must be a non-empty 2-D array')
    if not np.isin(y, (0, 1)).all():
        raise ValueError('y_multi must be binary multi-hot')
    n = y.sum(axis=0)
    if not n.any():
        raise ValueError('no positive labels — weights undefined')
    w = np.zeros(y.shape[1], dtype=np.float64)
    pos = n > 0
    w[pos] = (1.0 - beta) / (1.0 - np.power(beta, n[pos]))
    w *= y.shape[1] / w.sum()
    return w
