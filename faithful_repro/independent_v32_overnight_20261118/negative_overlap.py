import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)];import c2_runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
base=read(A/'results.json');assert base['status']=='PASS';rows=[]
for fold in range(5):
 bundle=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz'
 with np.load(bundle,allow_pickle=False) as z:tm=z['train_mask'].astype(bool);vp=z['val_pairs'];vy=z['y_val']
 assert not tm[vp[:,0],vp[:,1]].any() and (vy.sum(1)>0).all()
 for seed in (1234,42,2025):
  row=next(r for r in base['rows'] if r['fold']==fold and r['seed']==seed);assert sha(bundle)==row['bundle_sha256']
  run=((F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}') if seed==1234 else (F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}' if fold==0 else A/'runs'/f'fold{fold}_seed{seed}')
  man=read(run/'out/manifest.json');assert sha(run/'out/manifest.json')==row['manifest_sha256'];assert man['cfg']['seed']==man['cfg']['inter_seed']==seed
  for k,h in man['source_hashes'].items():assert sha(SRC/k)==h
  bank,bsha,_=c2_runner.build_negative_bank(man['cfg'],{'train_mask':tm});assert bsha==man['neg_bank']['sha256'] and not (bank&tm).any();on=bank[vp[:,0],vp[:,1]]
  n=int(on.sum());rows.append({'fold':fold,'seed':seed,'negative_bank_size':int(bank.sum()),'val_positive_count':len(vp),'val_positive_in_bank':n,'fraction_val_positive':n/len(vp),'fraction_bank_known_val_positive':n/int(bank.sum()),'by_type_overlap':vy[on].sum(0).astype(int).tolist(),'by_type_total':vy.sum(0).astype(int).tolist(),'bank_sha256':bsha})
out={'status':'PASS','scope':'known inner-validation positives accidentally used as train-zero negatives; no outer-test read','protocol_sha256':sha(P/'PROTOCOL_NEGATIVES.md'),'script_sha256':sha(__file__),'rows':rows}
with (P/'negative_overlap.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
for row in rows:print(row['fold'],row['seed'],row['val_positive_in_bank'],'/',row['val_positive_count'],'of bank',row['negative_bank_size'])
