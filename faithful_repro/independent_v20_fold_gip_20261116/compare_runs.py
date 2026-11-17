"""Compare completed paired folds; no model selection or inferential claims."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
base = HERE.parent / 'independent_v20_release_20261116' / 'runs' / 'full'
masked = HERE / 'runs' / 'full'
a = json.loads((base/'metric_audit.json').read_text(encoding='utf-8'))
b = json.loads((masked/'metric_audit.json').read_text(encoding='utf-8'))
assert a['status'] == b['status'] == 'PASS'
rows=[]
for fa,fb in zip(a['folds'],b['folds']):
    i=fa['fold']; assert i==fb['fold']
    x=np.load(base/f'fold_{i}.npz'); y=np.load(masked/f'fold_{i}.npz')
    for name in ('train_pos','train_neg','test_pos','test_neg','true_scalar_pos','y_multihot'):
        assert np.array_equal(x[name],y[name]),(i,name)
    for group, metric in [('released','f1'),('multihot','macro_f1'),('multihot','micro_f1')]:
        rows.append({'fold':i,'metric':f'{group}.{metric}','baseline':fa[group][metric],
                     'masked_gip':fb[group][metric], 'difference':fb[group][metric]-fa[group][metric]})
result={'status':'PASS','paired_fold_count':5,'paired_metrics':rows,
        'mean_differences':{key:float(np.mean([r['difference'] for r in rows if r['metric']==key]))
                            for key in ('released.f1','multihot.macro_f1','multihot.micro_f1')},
        'interpretation':'Joint intervention changes association features/graph and input GIP source. Static features and global class weights remain; no causal attribution to individual change or statistical generalization from one seed.'}
(HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
