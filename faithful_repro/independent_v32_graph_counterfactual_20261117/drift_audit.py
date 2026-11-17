import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;FR=P.parent
pa=FR/'independent_v32_seed_stability_20261117/runs/seed42/out/manifest.json';pb=P/'replay_retry2/out/manifest.json'
a=json.loads(pa.read_text());b=json.loads(pb.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(a[k]==b[k] for k in ('data_hashes','view_hashes','source_hashes','imported_pure_hashes'))
cd={k:[a['cfg'].get(k),b['cfg'].get(k)] for k in set(a['cfg'])|set(b['cfg']) if a['cfg'].get(k)!=b['cfg'].get(k)}
assert cd=={'max_epochs':[650,230]}
first={}
for k in ('train_loss','grad_norm','update_norm'):
 x,y=next((x,y) for x,y in zip(a['trajectory'],b['trajectory']) if x[k]!=y[k])
 first[k]={'epoch':x['epoch'],'historical':x[k],'replay':y[k]}
graph=[]
for x,y in zip(a['dynamic_history'],b['dynamic_history']):
 d=[k for k in x if 'sha' in k and x.get(k)!=y.get(k)]
 if d:graph={'epoch':x['epoch'],'fields':d};break
out={'status':'PASS','original_sha256':sha(pa),'replay_sha256':sha(pb),'cfg_diff':cd,'first_exact_differences':first,'first_dynamic_hash_difference':graph,'interpretation':'Descriptive comparison only; exact floats, no causal attribution.'}
with (P/'drift_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
