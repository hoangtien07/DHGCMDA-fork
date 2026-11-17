"""Compute release-style effective-number class weights from training pairs."""
import numpy as np

def train_weights(matrix, train_pos, beta=0.99999):
    matrix=np.asarray(matrix); pos=np.asarray(train_pos,dtype=np.int64)
    labels=matrix[pos[:,0],pos[:,1]].astype(np.int64)
    if np.any((labels<1)|(labels>4)) or len(set(map(tuple,pos))) != len(pos):
        raise ValueError('Invalid or duplicate training labels')
    counts=np.bincount(labels,minlength=5)[1:5]
    if (counts==0).any(): raise ValueError('At least one training class is missing')
    effective=(1-np.power(beta,counts))/(1-beta)
    raw=1/effective
    return counts,raw/raw.sum()*4
