import ast,json,tempfile,types,unittest
from pathlib import Path
P=Path(__file__).resolve().parent

def status_expr():
 tree=ast.parse((P/'finalizer.py').read_text(encoding='utf-8-sig'))
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in node.targets) and isinstance(node.value,ast.Dict):
   for k,v in zip(node.value.keys,node.value.values):
    if isinstance(k,ast.Constant) and k.value=='status':return compile(ast.fix_missing_locations(ast.Expression(v)),str(P/'finalizer.py'),'eval')
 raise AssertionError('status expression missing')

class FinalizerStatusTest(unittest.TestCase):
 def evaluate(self,watcher_status,core=0,overview=0):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d)
   if watcher_status is not None:(b/'watcher_done.json').write_text(json.dumps({'status':watcher_status}),encoding='utf-8')
   return eval(status_expr(),{'B':b,'json':json,'r1':types.SimpleNamespace(returncode=core),'r2':types.SimpleNamespace(returncode=overview)})
 def test_failed_watcher_never_reports_pass(self):self.assertEqual(self.evaluate('PARTIAL_OR_ERROR'),'PARTIAL')
 def test_missing_watcher_never_reports_pass(self):self.assertEqual(self.evaluate(None),'PARTIAL')
 def test_pass_requires_watcher_and_both_checks(self):
  self.assertEqual(self.evaluate('PASS'),'PASS');self.assertEqual(self.evaluate('PASS',core=1),'PARTIAL');self.assertEqual(self.evaluate('PASS',overview=1),'PARTIAL')
if __name__=='__main__':unittest.main()
