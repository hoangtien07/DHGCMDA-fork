import unittest
import numpy as np
from functional import functional_similarity

class FunctionalTests(unittest.TestCase):
    def test_hand_calculation_and_empty(self):
        a=np.array([[1,1,0],[0,1,0],[0,0,1],[0,0,0]])
        d=np.array([[1,.4,.2],[.4,1,.3],[.2,.3,1]])
        expected=np.array([[1,(.4+1+1)/3,(.2+.3+.3)/3,0],
                           [( .4+1+1)/3,1,.3,0],
                           [(.2+.3+.3)/3,.3,1,0],[0,0,0,1]])
        np.testing.assert_allclose(functional_similarity(a,d),expected,atol=1e-6)
    def test_types_as_binary_and_no_mutation(self):
        a=np.array([[1,0],[0,3]]); b=a.copy(); d=np.eye(2)
        np.testing.assert_allclose(functional_similarity(a,d),functional_similarity((a>0),d))
        np.testing.assert_array_equal(a,b)
    def test_bad_disease_matrix(self):
        with self.assertRaises(ValueError): functional_similarity(np.ones((2,2)),np.ones((3,3)))
    def test_matches_independent_disease_set_loop(self):
        rng=np.random.RandomState(17)
        a=rng.randint(0,2,(12,9)); d=rng.rand(9,9);d=(d+d.T)/2;np.fill_diagonal(d,1)
        ref=np.zeros((len(a),len(a)))
        for i in range(len(a)):
            for j in range(len(a)):
                x=np.flatnonzero(a[i]);y=np.flatnonzero(a[j])
                if len(x) and len(y):
                    s=d[np.ix_(x,y)];ref[i,j]=(s.max(0).sum()+s.max(1).sum())/(len(x)+len(y))
                if i==j:ref[i,j]=1
        np.testing.assert_allclose(functional_similarity(a,d),ref,atol=1e-6)

if __name__=='__main__': unittest.main()
