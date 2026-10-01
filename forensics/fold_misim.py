"""fold_misim.py — reconstruct MISIM-style miRNA functional similarity
from a TRAIN-only association matrix, using DSSM as the disease semantic
similarity. MISFUGE/Wang formula:

  S(mi, mj) = [ sum_{d in DTi} max_{d' in DTj} sem(d, d')
              + sum_{d in DTj} max_{d' in DTi} sem(d, d') ] / (|DTi| + |DTj|)

Validation: compute on FULL labels and correlate vs the artifact's
mi_fun_sim_3.2_3.csv (== MISIM 2.0). If formula is right, corr ~1.0.
Then use per-fold on train-only matrix = truly leakage-free counterpart.
"""
import numpy as np


def functional_sim(trainA, DSSM):
    """trainA [M, D] binary (train only). DSSM [D, D] semantic sim.
    Returns [M, M] functional similarity."""
    M, D = trainA.shape
    sets = [np.where(trainA[m] > 0)[0] for m in range(M)]
    S = np.zeros((M, M), dtype=np.float32)
    # precompute row-max matrix per disease-set lazily
    for i in range(M):
        si = sets[i]
        if len(si) == 0:
            continue
        for j in range(i, M):
            sj = sets[j]
            if len(sj) == 0:
                continue
            sub = DSSM[np.ix_(si, sj)]
            v = (sub.max(1).sum() + sub.max(0).sum()) / (len(si) + len(sj))
            S[i, j] = S[j, i] = v
    np.fill_diagonal(S, 1.0)
    return S
