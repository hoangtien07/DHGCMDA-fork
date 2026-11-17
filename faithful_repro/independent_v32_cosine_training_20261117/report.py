import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path.insert(0,str(SRC));import metrics
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
raw={1234:FR/'paper_guided/u1_horizon650_fold0_rev1/out',42:FR/'independent_v32_seed_stability_20261117/runs/seed42/out',2025:FR/'independent_v32_seed_stability_20261117/runs/seed2025/out'}
launch=read(P/'launch.json');assert launch['script_sha256']==sha(P/'train.py') and launch['protocol_sha256']==sha(P/'PROTOCOL.md')
for path,h in launch['pins'].items():assert sha(path)==h,path
rows=[]
for seed in (1234,42,2025):
 done=read(P/f'seed{seed}/completion.json');assert done['status']=='PASS' and done['epoch']==650
 pair={}
 for name,d in [('raw',raw[seed]),('cosine',P/f'seed{seed}/out')]:
  m=read(d/'manifest.json');assert m['aborted'] is None and m['final_epoch']==650 and m['cfg']['seed']==seed
  tr=m['trajectory'];assert len(tr)==650
  pred=d/'predictions_val.npz';meta=read(d/'predictions_val.meta.json');assert sha(pred)==meta['npz_sha256']
  with np.load(pred,allow_pickle=False) as z:
   met=metrics.type_metrics(z['y'],z['logits']);pairhash=hashlib.sha256(z['pairs'].tobytes()+z['y'].tobytes()).hexdigest()
  assert abs(met['primary']['macro_f1']-m['best']['val_macro_f1'])<1e-7
  final=tr[-1];loss=[x['train_loss'] for x in tr];spikes=[x['epoch'] for i,x in enumerate(tr) if i and x['train_loss']>2*tr[i-1]['train_loss']]
  pair[name]={'manifest_sha256':sha(d/'manifest.json'),'prediction_sha256':sha(pred),'pair_truth_sha256':pairhash,'best_epoch':m['best']['epoch'],'best_macro_f1':met['primary']['macro_f1'],'best_micro_f1':met['micro_f1'],'best_any_hit':met['any_hit_rate'],'per_class':met['per_class'],'final_macro_f1':final['val_macro_f1'],'final_micro_f1':final['val_micro_f1'],'final_any_hit':final['val_any_hit'],'max_loss':max(loss),'max_grad_norm':max(x['grad_norm'] for x in tr),'spike_count':len(spikes),'spike_epochs':spikes}
 assert pair['raw']['pair_truth_sha256']==pair['cosine']['pair_truth_sha256'];rows.append({'seed':seed,**pair})
result={'status':'PASS','scope':'exploratory same fold validation, three seeds; historical raw controls','report_script_sha256':sha(__file__),'protocol_sha256':sha(P/'PROTOCOL.md'),'rows':rows}
with (P/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(result,indent=2))
