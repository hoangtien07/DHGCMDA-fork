import hashlib,json,pathlib,subprocess,sys
import numpy as np
P=pathlib.Path('faithful_repro/independent_v32_full_plan_audit_20261117');root=pathlib.Path.cwd();sys.path.insert(0,str(root))
from Calculate_Metrics import OptimizedMetricCalculator
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit='06f9ba3eed92e334d9a2b8003fc06d83596e5614'
raw=subprocess.run(['git','show',commit+':Calculate_Metrics.py'],check=True,capture_output=True).stdout
local=pathlib.Path('Calculate_Metrics.py').read_bytes();print('metric raw source sha equal',hashlib.sha256(raw).hexdigest()==hashlib.sha256(local).hexdigest())
snapshot=pathlib.Path('v3.2_spld_paper/Y_multilabel.npy');ref=pathlib.Path('faithful_repro/data/v3.2_ref_MDAv3.2-3.npz');Y=np.load(snapshot,allow_pickle=False);Z=np.load(ref,allow_pickle=False)
assert Y.shape==(411,271,5) and np.array_equal(Y,Z['Y']);per=np.count_nonzero(Y,axis=(0,1)).tolist();pairs=int(np.count_nonzero(Y.sum(2)));multi=int(np.count_nonzero(Y.sum(2)>1));assert int(Y.sum())==11748 and pairs==8735 and multi==2110
calc=OptimizedMetricCalculator();truth=[[1,0,0,0,0]];q6=calc.compute_top1_metrics(truth,[[1,.9,.1,0,0,0]]);q5=calc.compute_top1_metrics(truth,[[.9,0,.1,0,0]])
assert q6['valid_samples']==0 and q6['top1_f1']==0 and q5['valid_samples']==1 and q5['correct_predictions']==0
x={'status':'PASS','author_commit':commit,'local_metric_sha256':sha(pathlib.Path('Calculate_Metrics.py')),'upstream_metric_sha256':hashlib.sha256(raw).hexdigest(),'local_equals_upstream_bytes':local==raw,'local_equals_upstream_normalized_newlines':local.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'),'snapshot_sha256':sha(snapshot),'ref_sha256':sha(ref),'shape':list(Y.shape),'typed_count':int(Y.sum()),'unique_pairs':pairs,'multitype_pairs':multi,'slice_counts':per,'metric_6_scores':{k:float(q6[k]) for k in ('valid_samples','correct_predictions','top1_f1')},'metric_5_direct_scores':{k:float(q5[k]) for k in ('valid_samples','correct_predictions','top1_f1')}}
with (P/'source_audit_rev2.json').open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(x,ensure_ascii=False),flush=True)
