import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;ref=F/'data/v3.2_ref_MDAv3.2-3.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(ref,allow_pickle=False) as z:y=z['Y'].astype(bool)
assert y.shape==(411,271,5);pairs=y.any(2);trip=np.argwhere(y);assert len(trip)==11748 and pairs.sum()==8735
unknown=~y;unknown_on_known_pair=int((unknown & pairs[:,:,None]).sum());unknown_on_no_pair=int((unknown & ~pairs[:,:,None]).sum());assert unknown_on_known_pair+unknown_on_no_pair==int(unknown.sum())
rows=[]
for seed in range(5):
 order=np.random.default_rng(seed).permutation(len(trip));folds=np.array_split(order,5);assert len(np.unique(np.concatenate(folds)))==len(trip)
 for fold,idx in enumerate(folds):
  train=np.ones(len(trip),dtype=bool);train[idx]=False;train_pairs=np.zeros(y.shape[:2],dtype=bool);train_pairs[trip[train,0],trip[train,1]]=True;t=trip[idx];seen=train_pairs[t[:,0],t[:,1]];assert not set(idx)&set(np.flatnonzero(train));rows.append({'seed':seed,'fold':fold,'test_positive_triplets':len(idx),'test_triplets_pair_seen_in_train':int(seen.sum()),'fraction':float(seen.mean()),'distinct_test_pairs':int(len(np.unique(t[:,:2],axis=0)))})
 fractions=[r['fraction'] for r in rows];out={'status':'PASS','scope':'simulated CVtriplet pair overlap, not author folds','protocol_sha256':sha(P/'PROTOCOL_TRIPLET.md'),'script_sha256':sha(__file__),'ref_sha256':sha(ref),'possible_triplets':int(y.size),'known_positive_triplets':len(trip),'unknown_triplets':int(unknown.sum()),'unknown_triplets_known_pair_other_type':unknown_on_known_pair,'unknown_triplets_no_known_pair':unknown_on_no_pair,'overlap_fraction_mean_25_simulated_folds':float(np.mean(fractions)),'overlap_fraction_min':float(np.min(fractions)),'overlap_fraction_max':float(np.max(fractions)),'rows':rows}
with (P/'triplet_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print('positive triplets pair seen in train: mean/min/max',*[round(v,4) for v in (out['overlap_fraction_mean_25_simulated_folds'],out['overlap_fraction_min'],out['overlap_fraction_max'])]);print('unknown triplets known pair other type',unknown_on_known_pair,'/',int(unknown.sum()))
