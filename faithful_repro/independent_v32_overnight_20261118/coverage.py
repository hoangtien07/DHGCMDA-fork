import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';L=F/'audit_v32_identity/mirna_mapping_ledger_v32.csv';M=F/'audit_v32_identity/strict_build_manifest.json';REF=F/'data/v3.2_ref_MDAv3.2-3.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'));m=read(M);assert sha(L)==m['ledger_sha256'];base=read(A/'results.json')
with L.open(encoding='utf-8-sig',newline='') as f:ledger=list(csv.DictReader(f))
with np.load(REF,allow_pickle=False) as z:names=z['mir_names']
assert len(ledger)==411 and [r['name'] for r in ledger]==names.tolist();rules=np.asarray([r['rule'] for r in ledger]);n=collections.Counter(rules);assert n=={'gff3-parent-exact':338,'gff3-parent-locus-variant':33,'name-not-in-mirbase-gff3':40}
result=[]
for seed in (1234,42,2025):
 accum={k:{'n':0,'mirnas':set(),'model':0,'product':0,'disease':0} for k in n}
 for fold in range(5):
  bundle=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz';row=next(r for r in base['rows'] if r['fold']==fold and r['seed']==seed);assert sha(bundle)==row['bundle_sha256']
  with np.load(bundle,allow_pickle=False) as z:tp=z['train_pairs'];ty=z['y_train'];vp=z['val_pairs'];vy=z['y_val']
  run=((F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}') if seed==1234 else (F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}' if fold==0 else A/'runs'/f'fold{fold}_seed{seed}')
  path=run/'out/predictions_val.npz';assert sha(path)==row['prediction_sha256']
  with np.load(path,allow_pickle=False) as z:assert np.array_equal(vp,z['pairs']) and np.array_equal(vy,z['y']);model=z['logits']
  cm=np.zeros((411,5));cd=np.zeros((271,5));np.add.at(cm,tp[:,0],ty);np.add.at(cd,tp[:,1],ty);g=ty.sum(0);disease=cd[vp[:,1]].copy();disease[disease.sum(1)==0]=g;disease+=(g/g.sum())*1e-12;product=np.log1p(cm[vp[:,0]])+np.log1p(cd[vp[:,1]])-np.log1p(g)
  hits={k:vy[np.arange(len(vy)),v.argmax(1)]>0 for k,v in [('model',model),('product',product),('disease',disease)]}
  for rule,q in accum.items():
   mask=rules[vp[:,0]]==rule;q['n']+=int(mask.sum());q['mirnas'].update(vp[mask,0].tolist())
   for k in hits:q[k]+=int(hits[k][mask].sum())
 for rule,q in accum.items():result.append({'seed':seed,'rule':rule,'unique_mirnas':len(q['mirnas']),'n_val_pair_occurrences':q['n'],'model_hit':q['model']/q['n'] if q['n'] else None,'product_hit':q['product']/q['n'] if q['n'] else None,'disease_hit':q['disease']/q['n'] if q['n'] else None})
out={'status':'PASS','scope':'post-hoc inner validation; mapping uncertainty is not model error or author coverage','protocol_sha256':sha(P/'PROTOCOL_COVERAGE.md'),'script_sha256':sha(__file__),'ledger_sha256':sha(L),'rows':result}
with (P/'coverage.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
for r in result:
 if r['seed']==1234:print(r['rule'],'n=',r['n_val_pair_occurrences'],'mirnas=',r['unique_mirnas'],'model/product/disease=',*[round(r[k],3) for k in ('model_hit','product_hit','disease_hit')])
