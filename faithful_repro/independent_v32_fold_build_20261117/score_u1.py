"""Audit completed U1 inner-validation exports and match fixed count baselines."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
sys.path.insert(0,str(FR/'independent_v32_metric_baseline_20261117'))
from analyze import metrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baselines=json.loads((HERE/'validation_baselines.json').read_text(encoding='utf-8'))['folds']
out=[]
for fold in range(5):
    run=(FR/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else (HERE/'u1_selection'/f'fold{fold}')
    bundle=(FR/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz') if fold<2 else (HERE/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz')
    if fold>=2 and not (run/'completion.json').exists():continue
    if fold>=2:
        comp=json.loads((run/'completion.json').read_text(encoding='utf-8'))
        assert comp['aborted'] is None and comp['final_epoch']==650 and sha(run/'out/manifest.json')==comp['manifest_sha256']
    assert sha(bundle)==baselines[fold]['bundle_sha256']
    f=np.load(bundle,allow_pickle=False);ex=run/'out/predictions_val.npz';v=np.load(ex,allow_pickle=False)
    em=json.loads((run/'out/predictions_val.meta.json').read_text(encoding='utf-8'))
    man=json.loads((run/'out/manifest.json').read_text(encoding='utf-8'))
    assert sha(ex)==em['npz_sha256'] and np.array_equal(v['pairs'],f['val_pairs']) and np.array_equal(v['y'],f['y_val'])
    result=metrics(v['y'],v['logits']);best=man['best'];assert man['final_epoch']==650 and man['aborted'] is None
    for k in ('macro_f1','micro_f1','any_hit'):assert abs(result[k]-best['val_'+k])<1e-12,(fold,k)
    final=next(r for r in reversed(man['trajectory']) if r.get('epoch')==650)
    row={'fold':fold,'n_validation':len(v['pairs']),'best_epoch':best['epoch'],'model_best':result,
         'model_final':{k:final['val_'+k] for k in ('macro_f1','micro_f1','any_hit')},
         'baselines':{k:{m:baselines[fold]['methods'][k][m] for m in ('macro_f1','micro_f1','any_hit')}
                      for k in ('global','disease','mirna','product_laplace1')},
         'prediction_sha256':sha(ex),'bundle_sha256':sha(bundle)}
    out.append(row);print(json.dumps({'fold':fold,'best_epoch':best['epoch'],'best_macro':result['macro_f1'],
               'final_macro':row['model_final']['macro_f1'],'product_macro':row['baselines']['product_laplace1']['macro_f1']}))
obj={'scope':'completed folds only, exploratory inner-validation, best chosen on same validation, no outer-test',
     'completed_folds':len(out),'folds':out,
     'limits':'650 horizon chosen after earlier development; fold0/1 Linux and fold2-4 Windows; not independent test or paper reproduction'}
if len(out)==5:
    obj['mean_over_folds']={x:float(np.mean([r['model_best'][x] for r in out])) for x in ('macro_f1','micro_f1','any_hit')}
(HERE/'u1_partial_metrics.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
