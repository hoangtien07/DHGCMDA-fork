import unittest
import numpy as np
from mask import train_matrix

class MaskTests(unittest.TestCase):
    def test_only_train_edges_and_input_unmodified(self):
        full = np.array([[1, 4, 0], [2, 0, 3]], dtype=np.float32)
        original = full.copy()
        train = np.array([[0, 0], [1, 2]])
        result = train_matrix(full, train, np.array([[0, 1], [1, 0]]))
        np.testing.assert_array_equal(result, [[1, 0, 0], [0, 0, 3]])
        np.testing.assert_array_equal(full, original)
        changed = full.copy(); changed[0, 1] = 2; changed[1, 0] = 4
        np.testing.assert_array_equal(result, train_matrix(changed, train, np.array([[0, 1], [1, 0]])))

    def test_overlapping_train_test_rejected(self):
        with self.assertRaises(ValueError):
            train_matrix(np.ones((2, 2)), np.array([[0, 1]]), np.array([[0, 1]]))

if __name__ == '__main__': unittest.main()
