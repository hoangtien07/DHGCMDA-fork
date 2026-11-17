"""Read-only v3.2 identity and actual U1 fold0 input audit."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
FR=ROOT/'faithful_repro'; SNAP=ROOT/'v3.2_spld_paper'
ref=FR/'data/v3.2_ref_MDAv3.2-3.npz'
strict=FR/'data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
bundle=FR/'paper_guided/candidate_fold0_selection_rev1/candidate_fold_data.npz'
meta=bundle.with_suffix('.meta.json')
expected=FR/'paper_guided/u1_horizon650_fold0_rev1/expected.json'
driver=FR/'paper_guided/u1_horizon650_fold0_rev1/driver.py'
ledger=FR/'audit_v32_identity/misim_overlap/source_archive/eval_vs_curated_pairs.csv'
candidate=FR/'colab_package_f0_selection_20261115/inputs/pair_split_candidate.csv'
candidate_manifest=FR/'colab_package_f0_selection_20261115/inputs/candidate_manifest.json'
b1=FR/'paper_guided/candidate_fold1_selection_rev1/candidate_fold_data.npz'
m1=b1.with_suffix('.meta.json');e1=FR/'paper_guided/u1_fold1_650_selection_rev1/expected.json'
inputs=[ref,strict,bundle,meta,expected,driver,ledger,candidate,candidate_manifest,b1,m1,e1,SNAP/'Y_multilabel.npy',SNAP/'triplets.csv',
        SNAP/'miRNA_names.txt',SNAP/'disease_names.txt',SNAP/'M_MISIM.npy',SNAP/'D_MESH.npy']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
asha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
assert sha(bundle)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
R=np.load(ref,allow_pickle=False); S=np.load(strict,allow_pickle=True); F=np.load(bundle,allow_pickle=False)
Y=np.load(SNAP/'Y_multilabel.npy',allow_pickle=False)
names_m=[x.strip() for x in (SNAP/'miRNA_names.txt').read_text(encoding='utf-8-sig').splitlines()]
names_d=[x.strip() for x in (SNAP/'disease_names.txt').read_text(encoding='utf-8-sig').splitlines()]
types=[str(t) for t in R['type_names']];assert types==['genetics','epigenetics','circulation','target','tissue']
assert Y.shape==(411,271,5) and np.isin(Y,[0,1]).all()
assert np.array_equal(Y,R['Y']) and np.array_equal(Y,S['Y'])
assert names_m==list(R['mir_names'])==list(S['mir_names'])
assert names_d==list(R['dis_names'])==list(S['dis_names'])
assert list(S['type_names'])==types
triples=[]
with (SNAP/'triplets.csv').open(encoding='utf-8-sig',newline='') as h:
    for r in csv.DictReader(h):triples.append((int(r['mi']),int(r['di']),int(r['type'])))
tripset=set(triples); yset=set(map(tuple,np.argwhere(Y>0)))
assert len(triples)==len(tripset)==len(yset)==11748 and tripset==yset
pos=Y.sum(axis=2)>0;assert int(pos.sum())==8735
assert np.array_equal(R['sim_m2'],np.load(SNAP/'M_MISIM.npy'))
assert np.array_equal(R['sim_d2'],np.load(SNAP/'D_MESH.npy'))
assert np.array_equal(R['sim_m2'],S['sim_m2']) and np.array_equal(R['sim_d2'],S['sim_d2'])
M=json.loads(meta.read_text(encoding='utf-8')); E=json.loads(expected.read_text(encoding='utf-8'))
for k,h in M['export_hashes'].items():assert asha(F[k])==h,(k,'bundle meta mismatch')
for k,h in E['data_hashes'].items():assert asha(F[k])==h,(k,'U1 data expected mismatch')
for k,h in E['view_hashes'].items():assert asha(F['view_'+k])==h,(k,'U1 view expected mismatch')
train=F['train_pairs'];val=F['val_pairs'];train_set=set(map(tuple,train));val_set=set(map(tuple,val))
assert len(train_set)==len(train)==6293 and len(val_set)==len(val)==695 and not train_set&val_set
assert all(pos[i,j] for i,j in train_set|val_set)
all_pos=set(map(tuple,np.argwhere(pos))); test_set=all_pos-train_set-val_set
mask=np.zeros(pos.shape,dtype=np.int8);mask[train[:,0],train[:,1]]=1
assert np.array_equal(F['train_mask'],mask)
assert np.array_equal(F['y_train'],Y[train[:,0],train[:,1]])
assert np.array_equal(F['y_val'],Y[val[:,0],val[:,1]])
for name,n,expected_mask in [('view_mi_v1',411,mask),('view_mi_v2',411,mask),
                             ('view_dis_v1',271,mask.T),('view_dis_v2',271,mask.T)]:
    assert np.array_equal(F[name][:,n:],expected_mask),name
assert np.array_equal(F['view_mi_v1'][:,:411],S['sim_m1'])
assert np.array_equal(F['view_dis_v1'][:,:271],S['sim_d1'])
fmi=F['view_mi_v2'][:,:411];sem=F['view_dis_v2'][:,:271]
assert sha(candidate)==M['pins']['pair_split_candidate.csv'] and sha(candidate_manifest)==M['pins']['candidate_manifest.json']
with candidate.open(encoding='utf-8-sig',newline='') as h:splitrows=list(csv.DictReader(h))
splitmap={(int(r['mir_idx']),int(r['dis_idx'])):r for r in splitrows}
assert len(splitmap)==len(splitrows)==len(all_pos) and set(splitmap)==all_pos
assert {p for p,r in splitmap.items() if r['seg_fold0']=='inner_train'}==train_set
assert {p for p,r in splitmap.items() if r['seg_fold0']=='inner_val'}==val_set
assert {p for p,r in splitmap.items() if r['seg_fold0']=='test'}==test_set
fold_group_counts=[]
for f in range(5):
    groups=collections.defaultdict(set);counts=collections.Counter();support=collections.defaultdict(lambda:np.zeros(5,dtype=np.int64))
    for r in splitrows:
        seg=r[f'seg_fold{f}'];groups[r['group_key']].add(seg);counts[seg]+=1
        support[seg]+=Y[int(r['mir_idx']),int(r['dis_idx'])].astype(np.int64)
    assert all(len(segs)==1 for segs in groups.values())
    groupcounts=collections.Counter(next(iter(segs)) for segs in groups.values())
    fold_group_counts.append({'fold':f,'pair_counts':dict(counts),'group_counts':dict(groupcounts),
                              'typed_support':{seg:dict(zip(types,v.tolist())) for seg,v in support.items()},
                              'cross_segment_groups':0})
assert all(sum(r[f'seg_fold{f}']=='test' for f in range(5))==1 for r in splitrows)
F1=np.load(b1,allow_pickle=False);M1=json.loads(m1.read_text(encoding='utf-8'));E1=json.loads(e1.read_text(encoding='utf-8'))
assert sha(b1)=='30b0ea3c216dd3a3f1a853d72b86888d08c414040bafc0c49ffeb7b98ab852ea'
for k,h in M1['export_hashes'].items():assert asha(F1[k])==h,(k,'fold1 meta')
for k,h in E1['data_hashes'].items():assert asha(F1[k])==h,(k,'fold1 expected data')
for k,h in E1['view_hashes'].items():assert asha(F1['view_'+k])==h,(k,'fold1 expected view')
t1=F1['train_pairs'];v1=F1['val_pairs'];ts1=set(map(tuple,t1));vs1=set(map(tuple,v1))
assert {p for p,r in splitmap.items() if r['seg_fold1']=='inner_train'}==ts1
assert {p for p,r in splitmap.items() if r['seg_fold1']=='inner_val'}==vs1
held1=all_pos-ts1-vs1;mask1=np.zeros(pos.shape,dtype=np.int8);mask1[t1[:,0],t1[:,1]]=1
assert np.array_equal(mask1,F1['train_mask']) and np.array_equal(F1['y_train'],Y[t1[:,0],t1[:,1]])
assert np.array_equal(F1['y_val'],Y[v1[:,0],v1[:,1]])
for name,n,expected_mask in [('view_mi_v1',411,mask1),('view_mi_v2',411,mask1),
                             ('view_dis_v1',271,mask1.T),('view_dis_v2',271,mask1.T)]:
    assert np.array_equal(F1[name][:,n:],expected_mask)
assert np.array_equal(F1['view_mi_v1'][:,:411],S['sim_m1']) and np.array_equal(F1['view_dis_v1'][:,:271],S['sim_d1'])
with ledger.open(encoding='utf-8-sig',newline='') as h:
    rows=list(csv.DictReader(h))
L={(int(r['mi_idx']),int(r['di_idx'])):r for r in rows}
assert len(rows)==len(L)==len(all_pos) and set(L)==all_pos
assert all(r['mir_name']==names_m[i] and r['dis_name']==names_d[j] for (i,j),r in L.items())
segment=lambda ij:'train' if ij in train_set else 'val' if ij in val_set else 'outer_test_complement'
seg_counts=collections.Counter();tier_counts=collections.defaultdict(collections.Counter)
with (OUT/'pair_ledger.csv').open('w',encoding='utf-8',newline='') as h:
    w=csv.writer(h);w.writerow(['mi_idx','di_idx','mir_name','disease_name','fold0_u1_segment','types','misim_pair_match_tier'])
    for i,j in sorted(all_pos):
        seg=segment((i,j));tier=L[(i,j)]['match_tier'];seg_counts[seg]+=1;tier_counts[seg][tier]+=1
        w.writerow([i,j,names_m[i],names_d[j],seg,'|'.join(types[k] for k in np.flatnonzero(Y[i,j])),tier])
with (OUT/'triplet_ledger.csv').open('w',encoding='utf-8',newline='') as h:
    w=csv.writer(h);w.writerow(['mi_idx','di_idx','type_idx','mir_name','disease_name','type_name','fold0_u1_segment','misim_pair_match_tier'])
    for i,j,k in sorted(tripset):w.writerow([i,j,k,names_m[i],names_d[j],types[k],segment((i,j)),L[(i,j)]['match_tier']])
cnt=Y.sum(axis=(0,1));degrees=Y.sum(axis=2).sum(axis=1);pair_degrees=pos.sum(axis=1)
def isolated_rows(a):
    x=np.array(a,copy=True);np.fill_diagonal(x,0);return int(np.count_nonzero(~np.any(x!=0,axis=1)))
out={'status':'PASS','input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
 'source_limits':'SPLD lineage snapshot identity to DHGCMDA author tensor not established; no author v3.2 tensor inspected',
 'dataset':{'shape':list(Y.shape),'typed_triplets':int(Y.sum()),'unique_pairs':int(pos.sum()),'per_type':dict(zip(types,cnt.astype(int).tolist())),
            'multi_type_pairs':int((Y.sum(2)>1).sum()),'mirna_unique_pair_degree_one':[names_m[i] for i in np.where(pair_degrees==1)[0]],
            'mirna_typed_degree_one':[names_m[i] for i in np.where(degrees==1)[0]],
            'triplet_csv_equal_tensor':True,'ref_strict_names_order_and_tensor_equal_snapshot':True},
 'u1_fold0_selection':{'bundle_sha256':sha(bundle),'hashes_match_driver_meta_expected':True,'segment_counts':dict(seg_counts),
   'train_val_test_disjoint':True,'train_mask_nonzero':int(mask.sum()),'heldout_positive_entries_in_mask':int(sum(mask[i,j] for i,j in val_set|test_set)),
   'all_four_association_blocks_match_train_mask':True,'seq_gene_match_strict':True,
   'recomputed_functional_equals_stored_misim':bool(np.array_equal(fmi,R['sim_m2'])),
   'func_misim_mean_abs_diff':float(np.abs(fmi-R['sim_m2']).mean()),
   'semantic_equals_stored_mesh':bool(np.array_equal(sem,R['sim_d2'])),
   'zero_offdiagonal_rows':{'S_seq':isolated_rows(S['sim_m1']),'S_gene':isolated_rows(S['sim_d1']),
                            'S_func_train':isolated_rows(fmi),'S_sem':isolated_rows(sem)},
   'misim_source_pair_tiers':{s:dict(c) for s,c in tier_counts.items()},
   'candidate_grouped_split':fold_group_counts},
 'u1_fold1_selection':{'bundle_sha256':sha(b1),'hashes_match_driver_meta_expected':True,
   'train_pairs':len(ts1),'val_pairs':len(vs1),'outer_test_pairs':len(held1),
   'heldout_positive_entries_in_mask':int(sum(mask1[i,j] for i,j in vs1|held1)),
   'all_four_association_blocks_match_train_mask':True,'seq_gene_match_strict':True,
   'functional_equals_stored_misim':bool(np.array_equal(F1['view_mi_v2'][:,:411],R['sim_m2']))},
 'limits':'Outer test segment identified by complement for provenance only, no predictions or scores read. MISIM presence is pair-level; not proof of a type label or causal score effect. Candidate U1 inner selection, not author fold.'}
(OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
