import unittest
from contract import make_cfg, validate_refit_keys
class ContractTests(unittest.TestCase):
 def test_policy_diff_and_no_mutation(self):
  parent={'stage':'U1','seed':1234,'inter_seed':1234,'max_epochs':650,'dynamic_enabled':True,'lr':.0001}
  a=make_cfg(parent,42,'dynamic');b=make_cfg(parent,42,'static')
  self.assertEqual(parent['stage'],'U1');self.assertEqual(a['max_epochs'],650);self.assertTrue(a['refit_noval']);self.assertEqual(a['seed'],a['inter_seed']);self.assertEqual({k for k in a if a[k]!=b[k]},{'dynamic_enabled'})
 def test_forbid_heldout_fields(self):
  validate_refit_keys({'views':{},'train_mask':1,'train_pairs':1,'y_train':1})
  for key in ('val_pairs','test_pairs','outer_mask','y_test'):
   with self.assertRaises(ValueError):validate_refit_keys({'views':{},'train_mask':1,'train_pairs':1,'y_train':1,key:1})
if __name__=='__main__':unittest.main()
