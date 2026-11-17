import unittest
import numpy as np
from mask import binary_gip

class GipTests(unittest.TestCase):
    def test_binary_values_and_manual_result(self):
        x = np.array([[1, 0], [4, 0], [0, 2]], dtype=np.float32)
        expected = np.array([[1, 1, np.exp(-2)], [1, 1, np.exp(-2)], [np.exp(-2), np.exp(-2), 1]])
        np.testing.assert_allclose(binary_gip(x), expected, rtol=1e-6)

    def test_all_zero_fails(self):
        with self.assertRaises(ValueError): binary_gip(np.zeros((2, 2)))

if __name__ == '__main__': unittest.main()
