# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 split + eval-set spec (pure, deterministic).

Pinned semantics — see ../PROTOCOL_CVTRIPLET_U1_REV2.md:
  partition  = default_rng(0).permutation(|U|) -> array_split 5
               (IDENTICAL to ../triplet_baseline.py for comparability)
  inner_val  = default_rng(2000+fold).choice(train_all, ceil(0.1*n), replace=False)
  eval negs  = default_rng(s) s in {0..4}, equal-count, no replacement
  neg superv = default_rng(seed+77).choice(mask-zero, min(10*n_pos, avail), replace=False)
All index arrays returned sorted int64; nothing reads or mutates Y.
"""
import hashlib
import numpy as np

SPLIT_SEED = 0
INNER_SEED_BASE = 2000
INNER_VAL_FRAC = 0.1
NEG_SUPERV_MULT = 10
NEG_SUPERV_SEED_OFF = 77
EVAL_SEEDS = (0, 1, 2, 3, 4)
BASELINE_NEG_SEED_BASE = 3000


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def positive_universe(Y):
    pos = np.argwhere(np.asarray(Y).astype(bool)).astype(np.int64)
    return pos


def fold_partition(pos):
    order = np.random.default_rng(SPLIT_SEED).permutation(len(pos))
    return [pos[i] for i in np.array_split(order, 5)]


def _sort_rows(a):
    """Lexicographic row sort — np.sort(axis=0) sorts each COLUMN and
    would scramble (i,j,k) triples; unique(axis=0) returns sorted rows
    of the same set."""
    a = np.asarray(a)
    if len(a) == 0:
        return a.reshape(0, a.shape[-1]).astype(np.int64)
    return np.unique(a, axis=0)


def inner_split(train_all, fold):
    n_val = int(np.ceil(INNER_VAL_FRAC * len(train_all)))
    sel = np.random.default_rng(INNER_SEED_BASE + fold).choice(
        len(train_all), size=n_val, replace=False)
    inner_val = _sort_rows(train_all[sel])
    keep = np.ones(len(train_all), bool)
    keep[sel] = False
    inner_train = train_all[keep]
    return inner_train, inner_val


def fold_segments(Y, fold):
    pos = positive_universe(Y)
    parts = fold_partition(pos)
    test = parts[fold]
    train_all = np.concatenate([p for i, p in enumerate(parts) if i != fold])
    inner_train, inner_val = inner_split(train_all, fold)
    return {'test': _sort_rows(test),
            'inner_train': _sort_rows(inner_train),
            'inner_val': _sort_rows(inner_val),
            'partition_sha': _sha(np.concatenate(parts)),
            'test_sha': _sha(_sort_rows(test)),
            'inner_train_sha': _sha(_sort_rows(inner_train)),
            'inner_val_sha': _sha(_sort_rows(inner_val))}


def pairs_of(triplets):
    if len(triplets) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(triplets[:, :2], axis=0)


def pair_mask(n_mi, n_dis, pairs):
    m = np.zeros((n_mi, n_dis), dtype=np.int8)
    if len(pairs):
        m[pairs[:, 0], pairs[:, 1]] = 1
    return m


def multihot(Y, pairs):
    C = Y.shape[2]
    out = np.zeros((len(pairs), C), dtype=np.int8)
    if len(pairs):
        out = (np.asarray(Y)[pairs[:, 0], pairs[:, 1], :] > 0).astype(np.int8)
    return out


def multihot_masked(Y, pairs, allowed_triplets):
    """Multi-hot restricted to channels present in allowed_triplets —
    held-out type bits stay 0 on straddled pairs."""
    allow = np.zeros(Y.shape[:2] + (Y.shape[2],), dtype=bool)
    if len(allowed_triplets):
        t = np.asarray(allowed_triplets)
        allow[t[:, 0], t[:, 1], t[:, 2]] = True
    out = np.zeros((len(pairs), Y.shape[2]), dtype=np.int8)
    if len(pairs):
        raw = (np.asarray(Y)[pairs[:, 0], pairs[:, 1], :] > 0)
        out = (raw & allow[pairs[:, 0], pairs[:, 1], :]).astype(np.int8)
    return out


def triplet_set(triplets):
    return set(map(tuple, np.asarray(triplets).tolist()))


def val_pairs_disjoint(inner_val, train_pairs):
    """Runner gate requires train/val pair disjointness: drop inner_val
    triplets on straddled pairs. Returns (val_pairs, kept, dropped)."""
    tr_pairs = set(map(tuple, np.asarray(train_pairs).tolist()))
    kept = np.asarray([t for t in inner_val
                       if (t[0], t[1]) not in tr_pairs], dtype=np.int64)
    dropped = len(inner_val) - len(kept)
    return pairs_of(kept), kept, dropped


def neg_supervision_index(mask, n_pos, seed):
    """One-time subsample of mask-zero cells for existence supervision
    (port of upstream single torch.randperm subsample; pinned rng)."""
    zeros = np.argwhere(~np.asarray(mask).astype(bool)).astype(np.int64)
    size = min(NEG_SUPERV_MULT * int(n_pos), len(zeros))
    sel = np.random.default_rng(int(seed) + NEG_SUPERV_SEED_OFF).choice(
        len(zeros), size=size, replace=False)
    idx = _sort_rows(zeros[sel])
    return idx, _sha(idx)


def unknown_universe(Y):
    return np.argwhere(~np.asarray(Y).astype(bool)).astype(np.int64)


def eval_unknown_sample(universe, n, seed):
    sel = np.random.default_rng(int(seed)).choice(
        len(universe), size=int(n), replace=False)
    return _sort_rows(universe[sel])


def pair_zero_index(Y):
    """Cells (i,j) with no positive channel anywhere — 'pair-zero'."""
    pz = ~np.asarray(Y).astype(bool).any(axis=2)
    return np.argwhere(pz).astype(np.int64)


def pair_unseen_test_pairs(Y, test_triplets):
    """Pairs whose EVERY positive channel lies in the test fold —
    pair existence fully held out."""
    test_pairs = triplet_set(test_triplets)  # (i,j,k)
    pos_pairs = pairs_of(positive_universe(Y))
    tp = set(map(tuple, np.asarray(pos_pairs).tolist()))
    out = []
    for (i, j) in sorted(tp):
        chans = {k for k in range(Y.shape[2]) if Y[i, j, k]}
        if chans and all((i, j, k) in test_pairs for k in chans):
            out.append((i, j))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def unknown_strata(unknown_triplets, mask, Y):
    """Per-negative info-condition strata (F03). Returns int8 codes:
    0=triplet_zero_on_train_pair, 1=triplet_zero_on_heldout_pair,
    2=pair_zero."""
    u = np.asarray(unknown_triplets)
    yb = np.asarray(Y).astype(bool)
    pair_pos = yb.any(axis=2)
    train = np.asarray(mask).astype(bool)[u[:, 0], u[:, 1]]
    gpos = pair_pos[u[:, 0], u[:, 1]]
    codes = np.full(len(u), 2, dtype=np.int8)
    codes[gpos & ~train] = 1
    codes[train] = 0
    return codes


def positive_strata(test_triplets, mask):
    t = np.asarray(test_triplets)
    return np.asarray(mask).astype(bool)[t[:, 0], t[:, 1]].astype(np.int8)
