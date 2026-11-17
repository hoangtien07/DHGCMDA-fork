import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();base=json.loads((A/'results.json').read_text());rows=[]
for fold in range(5):
 p=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1/candidate_fold_data.npz' if fold<2 else D/'bundles'/f'fold{fold}_selection/candidate_fold_data.npz';assert sha(p)==next(r for r in base['rows'] if r['fold']==fold)['bundle_sha256']
 with np.load(p,allow_pickle=False) as z:tp=z['train_pairs'];vp=z['val_pairs']
 assert not set(map(tuple,tp))&set(map(tuple,vp));tm=np.unique(tp[:,0]);td=np.unique(tp[:,1]);vm=np.unique(vp[:,0]);vd=np.unique(vp[:,1]);cm=~np.isin(vp[:,0],tm);cd=~np.isin(vp[:,1],td)
 row={'fold':fold,'train_pairs':len(tp),'validation_pairs':len(vp),'val_pair_overlap_train':0,'train_mirnas':len(tm),'train_diseases':len(td),'val_mirnas':len(vm),'val_diseases':len(vd),'val_mirna_unseen':int(cm.sum()),'val_disease_unseen':int(cd.sum()),'val_both_unseen':int((cm&cd).sum()),'val_both_seen':int((~cm&~cd).sum()),'bundle_sha256':sha(p)};rows.append(row);print(fold,'val both seen',row['val_both_seen'],'/',len(vp),'unseen mirna',row['val_mirna_unseen'],'unseen disease',row['val_disease_unseen'])
out={'status':'PASS','scope':'inner-selection split entity overlap; pair-disjoint, not necessarily entity-disjoint','protocol_sha256':sha(P/'PROTOCOL_SPLIT.md'),'script_sha256':sha(__file__),'rows':rows}
with (P/'split_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
