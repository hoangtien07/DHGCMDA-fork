import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';sys.path.insert(0,str(F/'independent_v32_metric_baseline_20261117'));from analyze import metrics
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
base=read(A/'results.json');assert base['status']=='PASS' and len(base['rows'])==15
raw=[];pool={}
for fold in range(5):
 bundle=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz'
 with np.load(bundle,allow_pickle=False) as z:tp=z['train_pairs'];ty=z['y_train'];vp=z['val_pairs'];vy=z['y_val']
 cm=np.zeros((411,5));cd=np.zeros((271,5));np.add.at(cm,tp[:,0],ty);np.add.at(cd,tp[:,1],ty);g=ty.sum(0);assert (g>0).all()
 md=np.bincount(tp[:,0],minlength=411)[vp[:,0]];dd=np.bincount(tp[:,1],minlength=271)[vp[:,1]]
 disease=cd[vp[:,1]].copy();disease[disease.sum(1)==0]=g;disease+=(g/g.sum())*1e-12
 product=np.log1p(cm[vp[:,0]])+np.log1p(cd[vp[:,1]])-np.log1p(g)
 groups={}
 for name,v in [('mi_degree',md),('disease_degree',dd)]:
  for lab,msk in [('0',v==0),('1-2',(v>=1)&(v<=2)),('3-9',(v>=3)&(v<=9)),('10+',v>=10)]:groups[name+':'+lab]=msk
 groups['labels:single']=vy.sum(1)==1;groups['labels:multiple']=vy.sum(1)>1
 for k in range(5):groups['true_class:'+str(k)]=vy[:,k]>0
 for seed in (1234,42,2025):
  row=next(r for r in base['rows'] if r['fold']==fold and r['seed']==seed);assert sha(bundle)==row['bundle_sha256']
  run=((F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}') if seed==1234 else (F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}' if fold==0 else A/'runs'/f'fold{fold}_seed{seed}')
  path=run/'out/predictions_val.npz';assert sha(path)==row['prediction_sha256']
  with np.load(path,allow_pickle=False) as z:assert np.array_equal(vp,z['pairs']) and np.array_equal(vy,z['y']);scores=z['logits']
  for name,sc in [('model',scores),('product',product),('disease',disease)]:
   q=metrics(vy,sc);ref=row['best' if name=='model' else name]
   for k in ('macro_f1','micro_f1','any_hit'):assert abs(q[k]-ref[k])<1e-12,(fold,seed,name,k)
  pred={'model':scores.argmax(1),'product':product.argmax(1),'disease':disease.argmax(1)}
  hits={k:vy[np.arange(len(vy)),v]>0 for k,v in pred.items()}
  for label,mask in groups.items():
   n=int(mask.sum());entry={'fold':fold,'seed':seed,'group':label,'n':n}
   if n:
    entry['support']=vy[mask].sum(0).astype(int).tolist()
    for name in pred:entry[name+'_hit']=float(hits[name][mask].mean())
    for name in ('product','disease'):
     entry['model_only_vs_'+name]=int((hits['model']&~hits[name]&mask).sum());entry[name+'_only']=int((hits[name]&~hits['model']&mask).sum())
    key=(seed,label);p=pool.setdefault(key,{'n':0,'model_hit_sum':0,'product_hit_sum':0,'disease_hit_sum':0,'model_only_vs_product':0,'product_only':0,'model_only_vs_disease':0,'disease_only':0})
    p['n']+=n
    for name in pred:p[name+'_hit_sum']+=int(hits[name][mask].sum())
    for name in ('product','disease'):p['model_only_vs_'+name]+=entry['model_only_vs_'+name];p[name+'_only']+=entry[name+'_only']
   raw.append(entry)
summary=[]
for (seed,label),v in sorted(pool.items()):
 n=v['n'];summary.append({'seed':seed,'group':label,**v,**{name+'_any_hit':v[name+'_hit_sum']/n for name in ('model','product','disease')}})
out={'status':'PASS','scope':'post-hoc inner-validation only; shared folds and seed, no independent test','protocol_sha256':sha(P/'PROTOCOL_STRATA.md'),'script_sha256':sha(__file__),'baseline_results_sha256':sha(A/'results.json'),'rows':raw,'pooled_by_seed':summary}
with (P/'strata.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for e in summary:
 if e['seed']==1234 and e['group'] in ('mi_degree:0','mi_degree:1-2','mi_degree:3-9','mi_degree:10+','labels:single','labels:multiple'):
  print(e['group'],'n=',e['n'],'hit model/product/disease=',*[round(e[n+'_any_hit'],3) for n in ('model','product','disease')])
