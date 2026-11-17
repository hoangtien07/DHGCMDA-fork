"""Check the released GIP matrix against the pinned full binary association CSV."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'independent_v20_release_20261116'
spec = '06f9ba3eed92e334d9a2b8003fc06d83596e5614:v2.0_495m383D/all_mirna_disease_pairs.csv'
blob = subprocess.check_output(['git', 'cat-file', 'blob', spec])
git_hash = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
assert git_hash == '3b1a253895ce800bddaabea96017510218a543e7'
rows = np.loadtxt(io.BytesIO(blob), delimiter=',', dtype=int)
A = np.zeros((495, 383), dtype=float)
A[rows[:, 0]-1, rows[:, 1]-1] = rows[:, 2]
assert int(A.sum()) == 5430
s = (A*A).sum(1); delta = s[:, None] + s[None, :] - 2*A@A.T
gip = np.exp(-495*delta/(A*A).sum())
released = np.loadtxt(BASE / 'pinned_release/v2.0_495m383D/M_GSM.txt')
diff = float(np.max(np.abs(gip - released)))
assert diff < 1e-12, diff
data = {'status':'PASS','binary_csv_git_blob':git_hash,'binary_pairs':int(A.sum()),
        'gip_max_abs_difference':diff,'folds':[]}
for i in range(1, 6):
    f=np.load(BASE / 'runs/full' / f'fold_{i}.npz')
    test=f['test_pos']; overlap=int(A[test[:,0],test[:,1]].sum())
    assert overlap==len(test)
    data['folds'].append({'fold':i,'test_positive_pairs_in_full_binary_input':overlap})
(HERE/'gip_audit.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
