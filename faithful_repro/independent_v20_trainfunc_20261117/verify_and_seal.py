"""Verify saved run contract and seal the evidence for later reproduction."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
run = root / 'runs/full'
contract = json.loads((run/'contract.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'PLAN.md') == contract['protocol_sha256']
assert sha(root/'run_masked.py') == contract['script_sha256']
assert sha(root/'source_manifest.json') == contract['source_manifest_sha256']
log = (run/'console.log').read_text(encoding='utf-8')
assert len(re.findall(r'^Epoch 650,', log, flags=re.M)) == 5
assert len(re.findall(r'^MASK AUDIT [1-5] train edges 1072 test type exposures 0$',log,flags=re.M)) == 5
assert len(re.findall(r'^TRAIN WEIGHT AUDIT [1-5] counts ',log,flags=re.M)) == 5
assert len(re.findall(r'^FUNCTION AUDIT [1-5] sha256 ',log,flags=re.M)) == 5
assert json.loads((run/'metric_audit.json').read_text())['status'] == 'PASS'
assert json.loads((root/'comparison.json').read_text())['status'] == 'PASS'
assert json.loads((root/'feature_audit.json').read_text())['status'] == 'PASS'
paths = sorted(root.glob('*.py')) + [root/'PLAN.md',root/'REPORT.md',root/'source_manifest.json',root/'feature_audit.json',root/'comparison.json']
paths += [run/p for p in ('contract.json','completion.json','console.log','metric_audit.json',
                           'fold_1.npz','fold_2.npz','fold_3.npz','fold_4.npz','fold_5.npz')]
paths += [run/f'functional_fold_{i}.npy' for i in range(1,6)]
assert all(p.is_file() for p in paths)
manifest={p.relative_to(root).as_posix():{'size':p.stat().st_size,'sha256':sha(p)} for p in paths}
(root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f'PASS: 5 final epochs, 5 mask audits, unchanged contract, {len(paths)} files sealed')
