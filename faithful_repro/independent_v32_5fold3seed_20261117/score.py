import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';sys.path.insert(0,str(F/'independent_v32_metric_baseline_20261117'));from analyze import metrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();baselines=json.loads((D/'validation_baselines.json').read_text(encoding='utf-8'))['folds'];rows=[]
for fold in range(5):
 bundle=(F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz') if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz';assert sha(bundle)==baselines[fold]['bundle_sha256'];b=np.load(bundle,allow_pickle=False)
 for seed in (1234,42,2025):
  if seed==1234:run=(F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}'
  elif fold==0:run=F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}'
  else:run=P/'runs'/f'fold{fold}_seed{seed}'
  if seed!=1234 or fold>=2:
   done=json.loads((run/'completion.json').read_text(encoding='utf-8'));assert done['aborted'] is None and done['final_epoch']==650 and done['manifest_sha256']==sha(run/'out/manifest.json')
  if seed!=1234 and fold>0:
   launch=json.loads((run/'launch.json').read_text(encoding='utf-8'));assert launch['protocol_sha256']==sha(P/'PROTOCOL.md') and launch['runner_sha256']==sha(P/'train.py') and launch['bundle_sha256']==sha(bundle) and launch['seed']==seed and launch['fold']==fold
  man=json.loads((run/'out/manifest.json').read_text(encoding='utf-8'));assert man['cfg']['seed']==man['cfg']['inter_seed']==seed and man['final_epoch']==650 and man['aborted'] is None
  pred=run/'out/predictions_val.npz';meta=json.loads((run/'out/predictions_val.meta.json').read_text(encoding='utf-8'));assert sha(pred)==meta['npz_sha256'];v=np.load(pred,allow_pickle=False);assert np.array_equal(v['pairs'],b['val_pairs']) and np.array_equal(v['y'],b['y_val'])
  q=metrics(v['y'],v['logits']);best=man['best'];final=next(x for x in man['trajectory'] if x['epoch']==650)
  for k in ('macro_f1','micro_f1','any_hit'):assert abs(q[k]-best['val_'+k])<1e-12
  rows.append({'fold':fold,'seed':seed,'best_epoch':best['epoch'],'best':q,'final':{k:final['val_'+k] for k in ('macro_f1','micro_f1','any_hit')},'product':{k:baselines[fold]['methods']['product_laplace1'][k] for k in ('macro_f1','micro_f1','any_hit')},'disease':{k:baselines[fold]['methods']['disease'][k] for k in ('macro_f1','micro_f1','any_hit')},'prediction_sha256':sha(pred),'manifest_sha256':sha(run/'out/manifest.json'),'bundle_sha256':sha(bundle)})
assert len(rows)==15
summary={str(seed):{'best_macro_mean_over_folds':float(np.mean([x['best']['macro_f1'] for x in rows if x['seed']==seed])),'best_macro_sd_over_folds':float(np.std([x['best']['macro_f1'] for x in rows if x['seed']==seed],ddof=1)),'final_macro_mean_over_folds':float(np.mean([x['final']['macro_f1'] for x in rows if x['seed']==seed]))} for seed in (1234,42,2025)}
out={'status':'PASS','scope':'5 folds x 3 seeds, inner validation selected checkpoints; not outer test; prior exposure','protocol_sha256':sha(P/'PROTOCOL.md'),'scorer_sha256':sha(Path(__file__)),'rows':rows,'summary':summary}
with (P/'results.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(summary),flush=True)
