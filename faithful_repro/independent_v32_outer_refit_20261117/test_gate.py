import unittest
from gate import required_keys,check_keys
class GateTests(unittest.TestCase):
 def test_all_30_only(self):
  keys=required_keys();self.assertEqual(len(keys),30);check_keys(keys)
  with self.assertRaises(ValueError):check_keys(keys[:-1])
  with self.assertRaises(ValueError):check_keys(keys+[keys[0]])
  with self.assertRaises(ValueError):check_keys(keys[:-1]+['wrong'])
if __name__=='__main__':unittest.main()
