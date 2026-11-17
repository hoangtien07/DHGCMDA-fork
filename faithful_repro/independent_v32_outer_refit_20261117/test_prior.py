import unittest
import numpy as np
from prior import fit_counts,predict_counts
class PriorTests(unittest.TestCase):
 def test_counts_fallback_and_product(self):
  p=np.array([[0,0],[0,1],[1,0]]);y=np.array([[1,0],[0,1],[1,1]])
  c=fit_counts(p,y,3,3);self.assertTrue(np.array_equal(c['global'],[2,2]));self.assertTrue(np.array_equal(c['mi'][0],[1,1]));self.assertTrue(np.array_equal(c['dis'][0],[2,1]))
  q=np.array([[2,2],[0,0]]);scores=predict_counts(c,q)
  np.testing.assert_allclose(scores['disease'][0], [2,2],atol=1e-11);np.testing.assert_allclose(scores['mirna'][0],[2,2],atol=1e-11)
  np.testing.assert_allclose(scores['product_laplace1'][1],np.log1p([1,1])+np.log1p([2,1])-np.log1p([2,2]))
 def test_invalid_binary_labels_rejected(self):
  with self.assertRaises(ValueError):fit_counts(np.array([[0,0]]),np.array([[0,2]]),1,1)
if __name__=='__main__':unittest.main()
