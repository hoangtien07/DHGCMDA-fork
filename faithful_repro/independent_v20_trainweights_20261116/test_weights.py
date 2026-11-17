import unittest
import numpy as np
from weights import train_weights

class WeightTests(unittest.TestCase):
    def test_fold_only_and_formula(self):
        m=np.array([[1,2,0,4],[3,1,1,4]],dtype=int)
        ix=np.array([[0,0],[0,1],[0,3],[1,0],[1,1]])
        counts,w=train_weights(m,ix)
        np.testing.assert_array_equal(counts,[2,1,1,1])
        raw=1/(1-np.power(0.99999,np.array([2,1,1,1])))
        np.testing.assert_allclose(w,raw/raw.sum()*4,rtol=1e-12)
        m[1,2]=4; m[1,3]=2
        np.testing.assert_allclose(train_weights(m,ix)[1],w,rtol=0,atol=0)
    def test_absent_class_rejected(self):
        with self.assertRaises(ValueError): train_weights(np.ones((1,1)),np.array([[0,0]]))

if __name__=='__main__': unittest.main()
