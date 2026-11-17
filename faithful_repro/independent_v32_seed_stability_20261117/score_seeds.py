"""Verify saved U1 validation exports for fixed fold0 seed sensitivity."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
sys.path.insert(0,str(FR/'independent_v32_metric_baseline_20261117'))
from analyze import metrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bundle=FR/'paper_guided/candidate_fold0_selection_rev1/candidate_fold_data.npz'
assert sha(bundle)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
b=np.load(bundle,allow_pickle=False);rows=[]
reference=json.loads((FR/'paper_guided/u1_horizon650_fold0_rev1/out/manifest.json').read_text(encoding='utf-8'))
for seed in (1234,42,2025):
    root=FR/'paper_guided/u1_horizon650_fold0_rev1' if seed==1234 else HERE/'runs'/f'seed{seed}'
    if seed!=1234 and not (root/'completion.json').exists():continue
    man=root/'out/manifest.json';m=json.loads(man.read_text(encoding='utf-8'))
    if seed!=1234:
        done=json.loads((root/'completion.json').read_text(encoding='utf-8'))
        assert done['seed']==seed and done['final_epoch']==650 and done['aborted'] is None and sha(man)==done['manifest_sha256']
    assert m['final_epoch']==650 and m['cfg']['seed']==m['cfg']['inter_seed']==seed
    assert {k:v for k,v in m['cfg'].items() if k not in ('seed','inter_seed')}=={k:v for k,v in reference['cfg'].items() if k not in ('seed','inter_seed')}
    assert m['data_hashes']==reference['data_hashes'] and m['view_hashes']==reference['view_hashes']
    p=root/'out/predictions_val.npz';meta=json.loads((root/'out/predictions_val.meta.json').read_text(encoding='utf-8'))
    assert sha(p)==meta['npz_sha256'];z=np.load(p,allow_pickle=False)
    assert np.array_equal(z['pairs'],b['val_pairs']) and np.array_equal(z['y'],b['y_val'])
    q=metrics(z['y'],z['logits']);best=m['best']
    for k in ('macro_f1','micro_f1','any_hit'):assert abs(q[k]-best['val_'+k])<1e-12,(seed,k)
    tr=[r for r in m['trajectory'] if r.get('val_macro_f1') is not None]
    values={str(t):next(r['val_macro_f1'] for r in tr if r['epoch']==t) for t in (200,400,650)}
    assert len(tr)==65
    rows.append({'seed':seed,'environment':'Linux/Colab' if seed==1234 else 'Windows CPU',
                 'best_epoch':best['epoch'],'best_metrics':q,'validation_macro_at_epochs':values,
                 'prediction_sha256':sha(p),'manifest_sha256':sha(man)})
out={'status':'PASS','scope':'one candidate fold0; validation selected checkpoint; no outer-test',
     'completed_seeds':len(rows),'seeds':rows,'limits':'Different training seed and negative-bank seed together; seed1234 platform differs; exploratory and no CI/generalization claim.'}
if len(rows)==3:
    a=[r['best_metrics']['macro_f1'] for r in rows];out['macro_best_mean']=float(np.mean(a));out['macro_best_range']=[min(a),max(a)]
(HERE/'seed_metrics.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for r in rows:print(json.dumps({'seed':r['seed'],'best_epoch':r['best_epoch'],'best_macro':r['best_metrics']['macro_f1'],'epochs':r['validation_macro_at_epochs']}))
