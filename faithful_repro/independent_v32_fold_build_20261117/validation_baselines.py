"""Fixed count baselines on 5 selection folds; no model or outer-test data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
sys.path.insert(0,str(FR/'independent_v32_metric_baseline_20261117'))
from analyze import metrics,prior
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((FR/'independent_v32_metric_baseline_20261117/results.json').read_text(encoding='utf-8'))
res=[]
for fold in range(5):
    p=(FR/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz') if fold<2 else (HERE/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz')
    z=np.load(p,allow_pickle=False);tp=z['train_pairs'];ty=z['y_train'];vp=z['val_pairs'];vy=z['y_val']
    assert not set(map(tuple,tp)) & set(map(tuple,vp))
    c_m=np.zeros((411,5));c_d=np.zeros((271,5));np.add.at(c_m,tp[:,0],ty);np.add.at(c_d,tp[:,1],ty)
    glob=ty.sum(0);assert (glob>0).all()
    d,missing_d=prior(tp,ty,vp,271)
    m=c_m[vp[:,0]].copy();missing_m=m.sum(1)==0;m[missing_m]=glob;m+=(glob/glob.sum())*1e-12
    scores={'global':np.broadcast_to(glob,(len(vp),5)).copy(),'disease':d,'mirna':m,
            'product_laplace1':np.log1p(c_m[vp[:,0]])+np.log1p(c_d[vp[:,1]])-np.log1p(glob)}
    outputs={name:metrics(vy,score) for name,score in scores.items()}
    if fold<2:
        prev=before['folds'][fold]['disease_prior']
        for k in ('macro_f1','micro_f1','any_hit'):assert abs(outputs['disease'][k]-prev[k])<1e-12,(fold,k)
    row={'fold':fold,'bundle_sha256':sha(p),'train_pairs':len(tp),'validation_pairs':len(vp),
         'unseen_disease_validation_pairs':missing_d,'unseen_mirna_validation_pairs':int(missing_m.sum()),'methods':outputs}
    res.append(row);print(json.dumps({'fold':fold,'macro_f1':{k:v['macro_f1'] for k,v in outputs.items()},
                                     'micro_f1':{k:v['micro_f1'] for k,v in outputs.items()},
                                     'unseen_mirna':int(missing_m.sum())}),flush=True)
means={name:{metric:float(np.mean([r['methods'][name][metric] for r in res])) for metric in ('macro_f1','micro_f1','any_hit')}
       for name in ('global','disease','mirna','product_laplace1')}
out={'scope':'inner-validation only; 5 grouped folds; NOT outer-test or paper score','folds':res,'mean_unweighted_over_folds':means,
     'limitations':'Global/disease/miRNA/product rules are descriptive; recipe and checkpoints have previous development exposure; not a blinded confirmatory benchmark.'}
(HERE/'validation_baselines.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
