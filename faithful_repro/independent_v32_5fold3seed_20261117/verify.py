import hashlib,json,math,statistics
from pathlib import Path
P=Path('faithful_repro/independent_v32_5fold3seed_20261117');F=P.parent;D=F/'independent_v32_fold_build_20261117';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();x=json.loads((P/'results.json').read_text());assert x['status']=='PASS' and len(x['rows'])==15 and x['protocol_sha256']==sha(P/'PROTOCOL.md') and x['scorer_sha256']==sha(P/'score.py');assert {(r['fold'],r['seed']) for r in x['rows']}=={(f,s) for f in range(5) for s in (1234,42,2025)}
for r in x['rows']:
 f,s=r['fold'],r['seed'];bundle=(F/'paper_guided'/f'candidate_fold{f}_selection_rev1/candidate_fold_data.npz') if f<2 else D/'bundles'/f'fold{f}_selection/candidate_fold_data.npz'
 if s==1234:run=(F/'paper_guided'/('u1_horizon650_fold0_rev1' if f==0 else 'u1_fold1_650_selection_rev1')) if f<2 else D/'u1_selection'/f'fold{f}'
 elif f==0:run=F/'independent_v32_seed_stability_20261117/runs'/f'seed{s}'
 else:run=P/'runs'/f'fold{f}_seed{s}'
 assert r['bundle_sha256']==sha(bundle) and r['prediction_sha256']==sha(run/'out/predictions_val.npz') and r['manifest_sha256']==sha(run/'out/manifest.json')
 q=r['best'];classes=q['per_class'];tp=sum(v['tp'] for v in classes.values());fp=sum(v['fp'] for v in classes.values());fn=sum(v['fn'] for v in classes.values());assert math.isclose(q['macro_f1'],statistics.mean(v['f1'] for v in classes.values()),abs_tol=1e-12) and math.isclose(q['micro_f1'],2*tp/(2*tp+fp+fn),abs_tol=1e-12)
 if f and s!=1234:
  l=json.loads((run/'launch.json').read_text());d=json.loads((run/'completion.json').read_text());assert l['protocol_sha256']==sha(P/'PROTOCOL.md') and l['runner_sha256']==sha(P/'train.py') and d['aborted'] is None and d['final_epoch']==650
for s in (1234,42,2025):
 a=[r for r in x['rows'] if r['seed']==s];z=x['summary'][str(s)];assert math.isclose(z['best_macro_mean_over_folds'],statistics.mean(r['best']['macro_f1'] for r in a),abs_tol=1e-12) and math.isclose(z['best_macro_sd_over_folds'],statistics.stdev(r['best']['macro_f1'] for r in a),abs_tol=1e-12)
print('PASS 15 fold/seed cells; data, prediction, manifest and per-class metric hashes/calculations; 8 new runs complete')
