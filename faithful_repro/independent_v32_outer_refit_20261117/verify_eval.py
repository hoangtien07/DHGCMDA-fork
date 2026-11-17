import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
P=Path(__file__).resolve().parent;F=P.parent;IN=F/'colab_package_f0_selection_20261115/inputs';sys.path.insert(0,str(P));from prior import predict_counts
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();E=P/'evaluation'
if not (E/'results.json').is_file():raise RuntimeError('outer results absent; no test access')
r=json.loads((E/'results.json').read_text());seal=json.loads((P/'seal.json').read_text());assert seal['verify_eval_sha256']==sha(Path(__file__));assert r['status']=='PASS' and r['seal_sha256']==sha(P/'seal.json') and r['scorer_sha256']==sha(P/'score.py') and r['protocol_sha256']==sha(P/'PROTOCOL.md') and len(r['rows'])==30 and len(r['baselines'])==5
with (IN/'pair_split_candidate.csv').open(encoding='utf-8-sig',newline='') as f:split=list(csv.DictReader(f))
with np.load(IN/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',allow_pickle=True) as z:Y=z['Y']
types=('genetics','epigenetics','circulation','target','tissue');fold_cache={}
for fold in range(5):
 pairs=np.array(sorted({(int(a['mir_idx']),int(a['dis_idx'])) for a in split if a[f'seg_fold{fold}']=='test'}),dtype=np.int64);y=Y[pairs[:,0],pairs[:,1]].astype(int);fold_cache[fold]=(pairs,y)
def check(y,logits,q):
 pred=logits.argmax(1);one=np.eye(5,dtype=int)[pred];tp=(one*y).sum(0);fp=(one*(1-y)).sum(0);fn=((1-one)*y).sum(0);f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5),where=(2*tp+fp+fn)>0)
 assert abs(float(f.mean())-q['macro_f1'])<1e-12 and abs(float(f1_score(y,one,average='macro',zero_division=0))-q['macro_f1'])<1e-12
 assert abs(float(f1_score(y,one,average='micro',zero_division=0))-q['micro_f1'])<1e-12 and abs(float(y[np.arange(len(y)),pred].mean())-q['any_hit'])<1e-12
 for i,k in enumerate(types):
  x=q['per_class'][k];assert x['tp']==int(tp[i]) and x['fp']==int(fp[i]) and x['fn']==int(fn[i]) and abs(x['f1']-f[i])<1e-12
for x in r['rows']:
 f,s,a=x['fold'],x['seed'],x['arm'];p=E/f'fold{f}_seed{s}_{a}.npz';assert x['prediction_sha256']==sha(p);z=np.load(p,allow_pickle=False);pairs,y=fold_cache[f];assert np.array_equal(z['pairs'],pairs) and np.array_equal(z['y'],y);check(y,z['logits'],x['metrics'])
for x in r['baselines']:
 f=x['fold'];p=P/'priors'/f'fold{f}_counts.npz';assert x['train_counts_sha256']==sha(p);z=np.load(p,allow_pickle=False);counts={k:z[k] for k in z.files};pairs,y=fold_cache[f]
 for name,score in predict_counts(counts,pairs).items():check(y,score,x['metrics'][name])
assert {(x['fold'],x['seed'],x['arm']) for x in r['rows']}=={(f,s,a) for f in range(5) for s in (1234,42,2025) for a in ('dynamic','static')}
assert {x['fold'] for x in r['baselines']}==set(range(5))
verified={'status':'PASS','results_sha256':sha(E/'results.json'),'verifier_sha256':sha(Path(__file__)),'seal_sha256':sha(P/'seal.json'),'prediction_count':30,'baseline_count':20}
with (E/'verification.json').open('x',encoding='utf-8') as f:json.dump(verified,f,indent=2);f.write('\n')
print('PASS 30 outer prediction hashes, test pairs/truth, sklearn and per-class metrics; 20 train-only baselines')
