import hashlib,json,pathlib,sys,numpy as np
P=pathlib.Path('faithful_repro/independent_v32_full_plan_audit_20261117');F=pathlib.Path('faithful_repro');D=F/'independent_v32_fold_build_20261117';sys.path.insert(0,str(F/'independent_v32_metric_baseline_20261117'));from analyze import metrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((D/'u1_partial_metrics.json').read_text());base=json.loads((D/'validation_baselines.json').read_text())['folds'];assert old['completed_folds']==5
out=[]
for i in range(5):
 run=(F/'paper_guided'/('u1_horizon650_fold0_rev1' if i==0 else 'u1_fold1_650_selection_rev1')) if i<2 else D/'u1_selection'/f'fold{i}'
 bundle=(F/'paper_guided'/f'candidate_fold{i}_selection_rev1/candidate_fold_data.npz') if i<2 else D/'bundles'/f'fold{i}_selection/candidate_fold_data.npz'
 pred=run/'out/predictions_val.npz';meta=json.loads((run/'out/predictions_val.meta.json').read_text());assert sha(pred)==meta['npz_sha256'] and sha(bundle)==base[i]['bundle_sha256'];b=np.load(bundle,allow_pickle=False);v=np.load(pred,allow_pickle=False);assert np.array_equal(v['pairs'],b['val_pairs']) and np.array_equal(v['y'],b['y_val']);q=metrics(v['y'],v['logits']);prev=old['folds'][i]
 for k in ('macro_f1','micro_f1','any_hit'):assert abs(q[k]-prev['model_best'][k])<1e-12
 out.append({'fold':i,'n':len(v['pairs']),'model':{k:q[k] for k in ('macro_f1','micro_f1','any_hit')},'product':{k:base[i]['methods']['product_laplace1'][k] for k in ('macro_f1','micro_f1','any_hit')},'disease':{k:base[i]['methods']['disease'][k] for k in ('macro_f1','micro_f1','any_hit')},'prediction_sha256':sha(pred),'bundle_sha256':sha(bundle)})
obj={'status':'PASS','scope':'previously selected inner validation, not test','folds':out,'means':{m:{k:float(np.mean([x[m][k] for x in out])) for k in ('macro_f1','micro_f1','any_hit')} for m in ('model','product','disease')}}
with (P/'fivefold_recheck.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
print(json.dumps(obj['means']))
