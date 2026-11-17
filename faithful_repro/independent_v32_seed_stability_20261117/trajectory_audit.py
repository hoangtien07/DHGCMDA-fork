"""Inspect saved selection trajectories only, without checkpoint/test access."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
score=json.loads((FR/'independent_v32_fold_build_20261117/u1_partial_metrics.json').read_text(encoding='utf-8'))
assert score['completed_folds']==5
rows=[]
for i in range(5):
    d=(FR/'paper_guided'/('u1_horizon650_fold0_rev1' if i==0 else 'u1_fold1_650_selection_rev1')) if i<2 else FR/'independent_v32_fold_build_20261117/u1_selection'/f'fold{i}'
    p=d/'out/manifest.json';m=json.loads(p.read_text(encoding='utf-8'));assert m['final_epoch']==650 and m['aborted'] is None
    vals=[x for x in m['trajectory'] if x.get('val_macro_f1') is not None]
    points={str(k):next(x for x in vals if x['epoch']==k)['val_macro_f1'] for k in (200,400,650)}
    best=m['best'];assert abs(best['val_macro_f1']-score['folds'][i]['model_best']['macro_f1'])<1e-12
    after=[x['val_macro_f1'] for x in vals if x['epoch']>=550]
    rows.append({'fold':i,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n_validation_looks':len(vals),
        'macro_f1_at_epochs':points,'best_epoch':best['epoch'],'best_macro_f1':best['val_macro_f1'],
        'gain_200_to_650':points['650']-points['200'],'best_minus_650':best['val_macro_f1']-points['650'],
        'validation_f1_min_max_550_to_650':[min(after),max(after)]})
out={'status':'PASS','scope':'retrospective same observed validation; no test scores','folds':rows,
     'mean_gain_200_to_650':float(np.mean([x['gain_200_to_650'] for x in rows])),
     'mean_best_minus_650':float(np.mean([x['best_minus_650'] for x in rows])),
     'limits':'Longer training and checkpoint maxima were selected after data exposure; no convergence or generalization claim.'}
(HERE/'trajectory_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
