import hashlib,json,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
P=Path(__file__).resolve().parent;E=P/'evaluation';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((E/'results.json').read_text());seal=json.loads((P/'seal.json').read_text());v=json.loads((E/'verification.json').read_text());assert v['status']=='PASS' and v['results_sha256']==sha(E/'results.json') and v['seal_sha256']==sha(P/'seal.json') and r['seal_sha256']==sha(P/'seal.json') and seal['score_sha256']==sha(P/'score.py') and seal['verify_eval_sha256']==sha(P/'verify_eval.py')
for x in r['rows']:
 f,s,a=x['fold'],x['seed'],x['arm'];p=E/f'fold{f}_seed{s}_{a}.npz';z=np.load(p,allow_pickle=False);assert sha(p)==x['prediction_sha256'];y=z['y'];pred=np.eye(5,dtype=int)[z['logits'].argmax(1)];q=x['metrics'];assert abs(f1_score(y,pred,average='macro',zero_division=0)-q['macro_f1'])<1e-12 and abs(f1_score(y,pred,average='micro',zero_division=0)-q['micro_f1'])<1e-12
for f in range(5):
 vals=[x for x in r['rows'] if x['fold']==f];assert len(vals)==6
 hashes={hashlib.sha256(np.load(E/f"fold{f}_seed{x['seed']}_{x['arm']}.npz",allow_pickle=False)['pairs'].tobytes()).hexdigest() for x in vals};assert len(hashes)==1
means={a:{k:statistics.mean(x['metrics'][k] for x in r['rows'] if x['arm']==a) for k in ('macro_f1','micro_f1','any_hit')} for a in ('dynamic','static')};bases={b:{k:statistics.mean(x['metrics'][b][k] for x in r['baselines']) for k in ('macro_f1','micro_f1','any_hit')} for b in ('product_laplace1','disease')};by={(x['fold'],x['seed'],x['arm']):x['metrics']['macro_f1'] for x in r['rows']};wins=sum(by[(f,s,'static')]>by[(f,s,'dynamic')] for f in range(5) for s in (1234,42,2025))
out={'status':'PASS','scope':'read-only recomputation from held-out exports; historical exposure; 15 correlated cells','results_sha256':sha(E/'results.json'),'seal_sha256':sha(P/'seal.json'),'means':means,'baselines':bases,'static_wins_of_15':wins,'script_sha256':sha(Path(__file__))}
with (P/'independent_review.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
