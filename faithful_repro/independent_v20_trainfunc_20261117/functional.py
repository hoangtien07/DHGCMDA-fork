"""Fold-local disease-set-based miRNA similarity, with declared empty-set rule."""
import numpy as np

def functional_similarity(association, disease_similarity):
    a=(np.asarray(association)!=0).astype(np.float32)
    d=np.asarray(disease_similarity,dtype=np.float32)
    if a.ndim!=2 or d.shape!=(a.shape[1],a.shape[1]) or not np.isfinite(d).all():
        raise ValueError('Invalid disease similarity or association shape')
    if not np.allclose(d,d.T,atol=1e-6) or np.min(d)<0:
        raise ValueError('Disease similarity must be symmetric and nonnegative')
    d=d.copy();np.fill_diagonal(d,1.)
    best=np.zeros((len(a),a.shape[1]),dtype=np.float32)
    for i in range(len(a)):
        known=np.flatnonzero(a[i])
        if len(known):best[i]=d[known].max(axis=0)
    matches=best @ a.T
    degree=a.sum(axis=1);denom=degree[:,None]+degree[None,:]
    sim=np.divide(matches+matches.T,denom,out=np.zeros_like(matches),where=denom>0)
    np.fill_diagonal(sim,1.)
    return sim
