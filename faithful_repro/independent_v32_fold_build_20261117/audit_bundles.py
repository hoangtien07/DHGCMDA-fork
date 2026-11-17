"""Independently verify fold 2–4 inputs without loading any test labels or predictions."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PKG=ROOT/'faithful_repro/colab_package_f0_selection_20261115';IN=PKG/'inputs';SRC=PKG/'src'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
asha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
pins=json.loads((PKG/'package_manifest.json').read_text(encoding='utf-8'))['files']
for rel in ['src/semantic.py','src/functional.py','src/contracts.py','inputs/HMDDv3.xlsx','inputs/meshToID.csv',
            'inputs/mtrees2019.bin','inputs/pair_split_candidate.csv','inputs/candidate_manifest.json',
            'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz']:
    assert sha(PKG/rel)==pins[rel],rel
sys.path.insert(0,str(SRC));import semantic,functional
strict=np.load(IN/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',allow_pickle=True)
with (IN/'pair_split_candidate.csv').open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
assert len(rows)==8735
ms=semantic.load_mesh_source(str(IN/'HMDDv3.xlsx'),str(IN/'meshToID.csv'),str(IN/'mtrees2019.bin'))
sem=semantic.semantic_similarity([str(x) for x in strict['dis_names']],ms,multi_descriptor='expand')
results=[]
for i in (2,3,4):
    root=HERE/'bundles'/f'fold{i}_selection';data=root/'candidate_fold_data.npz';z=np.load(data,allow_pickle=False)
    meta=json.loads((root/'candidate_fold_data.meta.json').read_text(encoding='utf-8'))
    recon=json.loads((root/'recon_meta.json').read_text(encoding='utf-8'))
    assert meta['fold']==i and meta['phase']=='selection' and recon['npz_sha256']==sha(data)
    for k,digest in meta['export_hashes'].items():assert asha(z[k])==digest,(i,k)
    split={seg:{(int(r['mir_idx']),int(r['dis_idx'])) for r in rows if r[f'seg_fold{i}']==seg}
           for seg in ('inner_train','inner_val','test')}
    tp=z['train_pairs'];vp=z['val_pairs'];train=set(map(tuple,tp));val=set(map(tuple,vp))
    assert train==split['inner_train'] and val==split['inner_val'] and not train&val
    assert not (train|val)&split['test'] and len(train)+len(val)+len(split['test'])==len(rows)
    mask=np.zeros((411,271),dtype=np.int8);mask[tp[:,0],tp[:,1]]=1
    assert np.array_equal(mask,z['train_mask'])
    assert all(mask[x,y]==0 for x,y in val|split['test'])
    assert np.array_equal(z['y_train'],strict['Y'][tp[:,0],tp[:,1]])
    assert np.array_equal(z['y_val'],strict['Y'][vp[:,0],vp[:,1]])
    for name,n,a in [('view_mi_v1',411,mask),('view_mi_v2',411,mask),('view_dis_v1',271,mask.T),('view_dis_v2',271,mask.T)]:
        assert np.array_equal(z[name][:,n:],a) and np.isfinite(z[name]).all()
    assert np.array_equal(z['view_mi_v1'][:,:411],strict['sim_m1'])
    assert np.array_equal(z['view_dis_v1'][:,:271],strict['sim_d1'])
    fun=functional.functional_similarity(mask.astype(bool),sem)
    assert np.array_equal(z['view_mi_v2'][:,:411],fun.sim.astype(np.float32))
    assert np.array_equal(z['view_dis_v2'][:,:271],sem.S.astype(np.float32))
    groups=collections.defaultdict(set)
    for r in rows:groups[r['group_key']].add(r[f'seg_fold{i}'])
    assert all(len(x)==1 for x in groups.values())
    rec={'fold':i,'status':'PASS','bundle_sha256':sha(data),'train_pairs':len(train),'val_pairs':len(val),
      'outer_test_pairs':len(split['test']),'heldout_entries_in_mask':0,'group_crossings':0,
      'train_type_support':z['y_train'].sum(0).astype(int).tolist(),
      'val_type_support':z['y_val'].sum(0).astype(int).tolist(),
      'functional_nonempty_mirnas':int(fun.nonempty.sum()),'semantic_covered_diseases':int(sem.coverage.sum()),
      'functional_and_semantic_rebuild_elementwise_equal':True,
      'static_seq_gene_equal_strict':True,'meta_array_hashes_verified':True,
      'test_labels_read_for_scoring':False}
    results.append(rec);print(json.dumps(rec),flush=True)
out={'status':'PASS','folds':results,'limitations':'Selection inputs only; no training, checkpoint, predictions or test score; disease, sequence and gene provenance differs from unrecovered paper recipe.'}
(HERE/'audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
