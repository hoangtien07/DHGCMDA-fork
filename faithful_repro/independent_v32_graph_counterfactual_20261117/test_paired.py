import unittest
import torch
from paired import run_pair

class PairedTests(unittest.TestCase):
    def test_common_rng_repeat_and_original_unchanged(self):
        torch.manual_seed(123)
        m=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.5),torch.nn.Linear(3,1));m.train()
        original={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        x=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
        def objective(model,scale):
            loss=(model(x)*scale).square().mean();return loss,{'scalar':float(loss.detach())}
        out=run_pair(m,1.,2.,objective)
        self.assertTrue(out['repeat_identical'])
        self.assertAlmostEqual(out['old']['loss']*4,out['new']['loss'],places=5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(original[k],v) for k,v in m.state_dict().items()))

if __name__=='__main__':unittest.main()
