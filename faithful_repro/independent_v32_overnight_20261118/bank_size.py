import copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)];import c2_runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
old=read(P/'negative_overlap.json');assert old['status']=='PASS';base=read(A/'results.json');rows=[]
for ref in old['rows']:
 fold,seed=ref['fold'],ref['seed'];bundle=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz'
 with np.load(bundle,allow_pickle=False) as z:tm=z['train_mask'].astype(bool);vp=z['val_pairs'];vy=z['y_val']
 row=next(r for r in base['rows'] if r['fold']==fold and r['seed']==seed);assert sha(bundle)==row['bundle_sha256'];run=((F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}') if seed==1234 else (F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}' if fold==0 else A/'runs'/f'fold{fold}_seed{seed}')
 man=read(run/'out/manifest.json');assert sha(run/'out/manifest.json')==row['manifest_sha256'];cfg=man['cfg'];bank10,h10,_=c2_runner.build_negative_bank(cfg,{'train_mask':tm});assert h10==ref['bank_sha256'];cfg1=dict(cfg);cfg1['inter_cap_mult']=1;bank1,h1,_=c2_runner.build_negative_bank(cfg1,{'train_mask':tm});assert not (bank1&~bank10).any();on=bank1[vp[:,0],vp[:,1]];rows.append({'fold':fold,'seed':seed,'n_val':len(vp),'n_bank10':int(bank10.sum()),'n_bank1':int(bank1.sum()),'val_in_bank10':ref['val_positive_in_bank'],'val_in_bank1':int(on.sum()),'bank1_sha256':h1,'subset':True})
out={'status':'PASS','scope':'predeclared negative bank count sensitivity, no test labels','protocol_sha256':sha(P/'PROTOCOL_BANK_SIZE.md'),'script_sha256':sha(__file__),'rows':rows}
with (P/'bank_size.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
for r in rows:print(r['fold'],r['seed'],'val collisions cap10/cap1:',r['val_in_bank10'],r['val_in_bank1'],'/',r['n_val'])
