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
