import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';sys.path.insert(0,str(F/'independent_v32_metric_baseline_20261117'));from analyze import metrics
from prior import fit_counts,predict_counts
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();baseline=json.loads((D/'validation_baselines.json').read_text())['folds'];rows=[]
for i in range(5):
 path=(F/'paper_guided'/f'candidate_fold{i}_selection_rev1/candidate_fold_data.npz') if i<2 else D/'bundles'/f'fold{i}_selection/candidate_fold_data.npz';z=np.load(path,allow_pickle=False);assert sha(path)==baseline[i]['bundle_sha256'];scores=predict_counts(fit_counts(z['train_pairs'],z['y_train'],411,271),z['val_pairs'])
 for m,s in scores.items():
  got=metrics(z['y_val'],s)
  for k in ('macro_f1','micro_f1','any_hit'):assert abs(got[k]-baseline[i]['methods'][m][k])<1e-12,(i,m,k,got[k],baseline[i]['methods'][m][k])
 rows.append({'fold':i,'bundle_sha256':sha(path),'methods':sorted(scores)})
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'prior_sha256':sha(P/'prior.py'),'script_sha256':sha(Path(__file__)),'scope':'reproduce preexisting validation baseline, no outer test','folds':rows}
with (P/'prior_validation_parity.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print('PASS 5 validation folds, 4 baseline rules, 3 metrics each, train counts only')
