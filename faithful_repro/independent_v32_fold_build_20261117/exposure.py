"""Index-only overlap of candidate outer-test cells with historical U1 selection sets."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
p=FR/'colab_package_f0_selection_20261115/inputs/pair_split_candidate.csv'
digest=hashlib.sha256(p.read_bytes()).hexdigest()
assert digest=='aefc9b05f971210db01281f4362c9ac4132b4de361c7315f5ca6c4d2b8081040'
with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==8735
result=[]
for target in range(5):
    test=[r for r in rows if r[f'seg_fold{target}']=='test'];assert len(test)==1747
    fold0=collections.Counter(r['seg_fold0'] for r in test)
    fold1=collections.Counter(r['seg_fold1'] for r in test)
    both=collections.Counter(r['seg_fold0']+'|'+r['seg_fold1'] for r in test)
    result.append({'target_outer_fold':target,'n_pairs':len(test),'membership_in_fold0_selection':dict(fold0),
                   'membership_in_fold1_selection':dict(fold1),'joint_membership':dict(both)})
out={'status':'PASS','split_sha256':digest,'folds':result,
     'interpretation':'Pair membership only: prior inclusion in training/validation does not establish individual manual inspection or disqualify ordinary CV; it limits independence of subsequent research selection after observed outcomes. No test labels or predictions read.'}
(HERE/'exposure.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
