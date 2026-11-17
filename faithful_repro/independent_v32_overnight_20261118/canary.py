import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;PK=F/'colab_package_f0_selection_20261115';SRC=PK/'src';D=F/'independent_v32_fold_build_20261117';base=json.loads((F/'independent_v32_5fold3seed_20261117/results.json').read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins=json.loads((PK/'package_manifest.json').read_text(encoding='utf-8'))['files'];fn=SRC/'build_candidate_fold_features_rev3.py';rel='src/'+fn.name
assert sha(fn)==pins[rel];sys.path.insert(0,str(SRC));spec=importlib.util.spec_from_file_location('builder_canary',fn);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
strict=PK/'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz';assert sha(strict)==pins['inputs/'+strict.name]
with np.load(strict,allow_pickle=True) as z:Y=z['Y'];seq=z['sim_m1'];gene=z['sim_d1']
rows=[]
for fold in range(5):
 path=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz';entry=next(r for r in base['rows'] if r['fold']==fold and r['seed']==1234);assert sha(path)==entry['bundle_sha256']
 with np.load(path,allow_pickle=False) as z:expected={k:z[k] for k in z.files}
 tp=expected['train_pairs'];vp=expected['val_pairs'];mask=expected['train_mask'];sem=expected['view_dis_v2'][:,:271];func=expected['view_mi_v2'][:,:411];calls=[]
 def func_fn(m):
  assert np.array_equal(m,mask);calls.append('train mask verified');return func
 def build(y):return builder.normalize_dtypes(builder.build_flat(y,tp,vp,seq,gene,func_fn,lambda:sem))
 a=build(Y);Y2=Y.copy();Y2[vp[:,0],vp[:,1],:]=np.roll(Y[vp[:,0],vp[:,1],:],1,axis=1);b=build(Y2);changed=int(np.any(a['y_val']!=b['y_val'],axis=1).sum());assert changed>0
 for k in a:
  assert k in expected and np.array_equal(a[k],expected[k]),(fold,k,'baseline mismatch')
  if k!='y_val':assert np.array_equal(a[k],b[k]),(fold,k,'label contamination')
 assert len(calls)==2;rows.append({'fold':fold,'bundle_sha256':sha(path),'val_pairs':len(vp),'changed_val_labels':changed,'unchanged_output_keys':[k for k in a if k!='y_val'],'verified_train_mask_calls':len(calls)})
out={'status':'PASS','scope':'pure build_flat with train-derived functional/semantic inputs held fixed; NOT a full static-feature lineage audit','protocol_sha256':sha(P/'PROTOCOL_CANARY.md'),'script_sha256':sha(__file__),'builder_sha256':sha(fn),'strict_sha256':sha(strict),'rows':rows}
with (P/'canary.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
for r in rows:print('fold',r['fold'],'changed validation labels',r['changed_val_labels'],'/',r['val_pairs'],'train/view unchanged PASS')
