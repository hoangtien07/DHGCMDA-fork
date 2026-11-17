"""Construct fold features using training associations only."""
import numpy as np

def train_matrix(full, train_pos, test_pos):
    full = np.asarray(full)
    train = np.asarray(train_pos, dtype=np.int64)
    test = np.asarray(test_pos, dtype=np.int64)
    if set(map(tuple, train)) & set(map(tuple, test)):
        raise ValueError('Training and test positives overlap')
    result = np.zeros_like(full)
    result[train[:, 0], train[:, 1]] = full[train[:, 0], train[:, 1]]
    if np.count_nonzero(result) != len(train):
        raise ValueError('Invalid or duplicate training positives')
    return result

def binary_gip(matrix):
    a = (np.asarray(matrix) != 0).astype(np.float32)
    total = float((a*a).sum())
    if total == 0:
        raise ValueError('GIP requires at least one training edge')
    squared = (a*a).sum(axis=1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * (a @ a.T), 0)
    return np.exp(-a.shape[0] * distances / total).astype(np.float32)
