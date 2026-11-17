import hashlib,json
from pathlib import Path
import numpy as np
from prior import fit_counts
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=json.loads((P/'bundle_audit.json').read_text());assert a['status']=='PASS';out=P/'priors';out.mkdir(exist_ok=False);rows=[]
for i in range(5):
 b=P/'bundles'/f'fold{i}_refit/candidate_fold_data.npz';assert sha(b)==a['folds'][i]['bundle_sha256'];z=np.load(b,allow_pickle=False);counts=fit_counts(z['train_pairs'],z['y_train'],z['train_mask'].shape[0],z['train_mask'].shape[1]);p=out/f'fold{i}_counts.npz';np.savez_compressed(p,**counts)
 with np.load(p,allow_pickle=False) as q:
  assert all(np.array_equal(q[k],v) for k,v in counts.items())
 rows.append({'fold':i,'bundle_sha256':sha(b),'counts_sha256':sha(p),'n_train':len(z['train_pairs'])})
with (P/'prior_manifest.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'prior_sha256':sha(P/'prior.py'),'builder_sha256':sha(Path(__file__)),'folds':rows},f,indent=2);f.write('\n')
print('PASS: five train-only prior count arrays sealed')
