# -*- coding: utf-8 -*-
"""Train-only MISIM-style functional similarity — REV2 descriptor keys.

Independent implementation of the inspected default formula (author
archive script 3.1, read — never executed). For miRNAs a,b with train-only
disease-KEY sets Ka,Kb the score is the cosine of two augmented vectors:

    A[k]      = DV[k]           for k in Ka
    A[k1,k2]  = T[k1,k2]        for all UNORDERED key pairs in Ka x Kb
    B analogously; sim = (sum_{k in Ka∩Kb} DV^2 + sum_pairs T^2)
        / sqrt( (sum_Ka DV^2 + sum_pairs T^2)(sum_Kb DV^2 + sum T^2) )

diagonal fixed 1.0 (author convention), zero denominator -> 0.

REV2: sets are over semantic.desc_keys, not row indices — rows sharing a
descriptor key (aliases) contribute once, matching the author formula's
name-keyed dicts. ('U',i) keys remain distinct per row (each uncovered
row is its own entity — it has dv=1 and no shared ancestors).

PURITY CONTRACT: the only label-derived input is `train_mask`
(n_mirna x n_disease bool) from contracts.train_associations — held-out
labels are structurally unreachable.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class FunctionalResult:
    """sim:          (n_mirna, n_mirna) float64, diag = 1.0.
    nonempty:     (n_mirna,) bool — >=1 train association.
    provenance:   mask sha256 + semantic provenance + formula note.
    """
    sim: np.ndarray
    nonempty: np.ndarray
    provenance: dict


def functional_similarity(train_mask, sem):
    """Compute the fold-only functional matrix.

    train_mask: (n_mirna, n_disease) bool — train associations ONLY.
    sem:        SemanticResult aligned to mask columns (desc_keys used
                for canonical dedup).
    """
    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != sem.T.shape[0]:
        raise ValueError('train_mask must be (n_mirna, n_disease) and '
                         'align with the semantic disease order')
    T, dv, keys = sem.T, np.asarray(sem.dv), sem.desc_keys
    rep = sem.key_rep
    n_m = mask.shape[0]
    # per-miRNA KEY sets (alias rows collapse to one key)
    ksets = [set() for _ in range(n_m)]
    for i in range(n_m):
        for j in np.flatnonzero(mask[i]):
            ksets[i].add(keys[j])
    nonempty = np.array([len(s) > 0 for s in ksets])
    # per-key dv (rows sharing a key carry identical dv by contract)
    key_dv = {k: dv[rep[k]] for k in rep}
    sim = np.zeros((n_m, n_m))
    for a in range(n_m):
        Ka = ksets[a]
        if not Ka:
            continue
        ra = sorted(rep[k] for k in Ka)
        ownA = float((dv[ra] ** 2).sum())
        for b in range(a + 1, n_m):
            Kb = ksets[b]
            if not Kb:
                continue
            rb = sorted(rep[k] for k in Kb)
            # unordered unique key pairs: canonical key pair dedup —
            # mirrors the author's '__'.join(sorted([d1,d2])) collapse.
            # Iteration sorted by key repr for hash-seed determinism.
            pair_mass = 0.0
            seen = set()
            for k1 in sorted(Ka, key=repr):
                r1 = rep[k1]
                for k2 in sorted(Kb, key=repr):
                    r2 = rep[k2]
                    canon = (r1, r2) if r1 <= r2 else (r2, r1)
                    if canon in seen:
                        continue
                    seen.add(canon)
                    pair_mass += T[r1, r2] ** 2
            shared = sorted((Ka & Kb), key=repr)
            cross = float(sum(key_dv[k] ** 2 for k in shared))
            nA = ownA + pair_mass
            nB = float((dv[rb] ** 2).sum()) + pair_mass
            denom = (nA * nB) ** 0.5
            sim[a, b] = sim[b, a] = (cross + pair_mass) / denom \
                if denom else 0.0
    np.fill_diagonal(sim, 1.0)
    prov = {'train_mask_sha256': hashlib.sha256(
                np.ascontiguousarray(mask).tobytes()).hexdigest(),
            'semantic': dict(sem.provenance),
            'formula': 'MISIM-default augmented one-hot cosine '
                       '(theta_dag=0.5, theta_pair=1.0); descriptor-key '
                       'dedup; train-mask only'}
    return FunctionalResult(sim=sim, nonempty=nonempty, provenance=prov)
