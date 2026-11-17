import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;R=F.parent;S=R/'v3.2_spld_paper';ref=F/'data/v3.2_ref_MDAv3.2-3.npz';PK=F/'colab_package_f0_selection_20261115';strict=PK/'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz';split=PK/'inputs/pair_split_candidate.csv';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(ref,allow_pickle=False) as z:a={k:z[k] for k in z.files}
with np.load(strict,allow_pickle=True) as z:b={k:z[k] for k in z.files}
y=np.load(S/'Y_multilabel.npy',allow_pickle=False);assert y.shape==(411,271,5) and np.isin(y,[0,1]).all() and np.array_equal(y,a['Y']) and np.array_equal(y,b['Y'])
for name,key in [('miRNA_names.txt','mir_names'),('disease_names.txt','dis_names')]:
 names=np.array((S/name).read_text(encoding='utf-8-sig').splitlines());assert np.array_equal(names,a[key]) and np.array_equal(names,b[key]);assert len(set(names.tolist()))==len(names)
for name,key in [('M_MISIM.npy','sim_m2'),('D_MESH.npy','sim_d2')]:assert np.array_equal(np.load(S/name,allow_pickle=False),a[key])
assert y.sum()==11748 and (y.sum(2)>0).sum()==8735 and (y.sum(2)>1).sum()==2110
with split.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==8735;keys=[(int(r['mir_idx']),int(r['dis_idx'])) for r in rows];assert len(set(keys))==len(keys) and all(y[i,j].any() for i,j in keys)
test_hits=collections.Counter();folds=[]
for fold in range(5):
 groups=collections.defaultdict(set);sets={x:set() for x in ('inner_train','inner_val','test')}
 for row,key in zip(rows,keys):
  seg=row[f'seg_fold{fold}'];assert seg in sets;sets[seg].add(key);groups[row['group_key']].add(seg)
  if seg=='test':test_hits[key]+=1
 assert all(len(v)==1 for v in groups.values()) and all(not sets[x]&sets[z] for x,z in [('inner_train','inner_val'),('inner_train','test'),('inner_val','test')]);folds.append({'fold':fold,'train':len(sets['inner_train']),'val':len(sets['inner_val']),'test':len(sets['test']),'group_crossings':0})
assert len(test_hits)==8735 and set(test_hits.values())=={1}
pairdeg=(y.sum(2)>0).sum(1);typdeg=y.sum((1,2));single=[str(a['mir_names'][i]) for i in np.flatnonzero(typdeg==1)]
out={'status':'PASS','scope':'local SPLD/ref/strict identity and pair-group split; author v3.2 identity unverified','protocol_sha256':sha(P/'PROTOCOL_DATA.md'),'script_sha256':sha(__file__),'ref_sha256':sha(ref),'strict_sha256':sha(strict),'split_sha256':sha(split),'snapshot_sha256':sha(S/'Y_multilabel.npy'),'shape':list(y.shape),'typed_counts':y.sum((0,1)).astype(int).tolist(),'n_pairs':int((y.sum(2)>0).sum()),'n_multitype':int((y.sum(2)>1).sum()),'n_pairs_by_type_count':{str(k):int((y.sum(2)==k).sum()) for k in range(1,6)},'single_typed_link_mirnas':single,'min_pair_degree':int(pairdeg.min()),'folds':folds}
with (P/'data_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
