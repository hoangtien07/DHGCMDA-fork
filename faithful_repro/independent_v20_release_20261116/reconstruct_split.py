"""Reconstruct release split from source without running a model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'pinned_release/v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv'


def main():
    A=np.zeros((495,383),dtype=np.float32)
    multi=np.zeros((495,383,4),dtype=np.uint8)
    with DATA.open(encoding='utf-8-sig',newline='') as h:
        for row in csv.reader(h):
            i,j,k=map(int,row[:3]);i-=1;j-=1;A[i,j]=k;multi[i,j,k-1]=1
    assert int(multi.sum())==1679 and int((A>0).sum())==1498
    groups={'zero':[],'type1':[],'type2':[],'type3':[],'type4':[],'nonzero':[]}
    for i,j in zip(*np.where(A!=0)):
        groups['nonzero'].append([i,j]);groups['type'+str(int(A[i,j]))].append([i,j])
    for i,j in zip(*np.where(A==0)):groups['zero'].append([i,j])
    rng=np.random.RandomState(0)
    for key in groups:
        rng.shuffle(groups[key]);groups[key]=np.asarray(groups[key],dtype=np.int64)
    npos=len(groups['nonzero']); nneg=len(groups['zero'])
    posstep=npos//10; negstep=nneg//10
    cvpos=groups['nonzero'][:9*posstep]; cvneg=groups['zero'][:9*negstep]
    pstep=len(cvpos)//5;nstep=len(cvneg)//5
    folds=[]
    for i in range(5):
        p=cvpos[i*pstep:(i+1)*pstep];n=cvneg[i*nstep:(i+1)*nstep]
        archive=np.load(HERE/'runs/smoke'/f'fold_{i+1}.npz')
        assert np.array_equal(p,archive['test_pos']) and np.array_equal(n,archive['test_neg'])
        folds.append({'fold':i+1,'positive':len(p),'negative':len(n),
             'multitype':int((multi[p[:,0],p[:,1]].sum(1)>1).sum()),
             'type_exposed_in_association_features':int(np.array_equal(A[p[:,0],p[:,1]],archive['true_scalar_pos']))*len(p)})
    indpos=groups['nonzero'][9*posstep:10*posstep]
    indneg=groups['zero'][9*negstep:10*negstep]
    used={(int(i),int(j)) for row in [*[cvpos[k*pstep:(k+1)*pstep] for k in range(5)],indpos] for i,j in row}
    unassigned=[(int(i),int(j)) for i,j in groups['nonzero'] if (int(i),int(j)) not in used]
    assert len(used)+len(unassigned)==npos
    out={'source_csv_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
      'scope':'Exact release split matched against smoke captures; no training or scoring.',
      'split_seed':0,'full_positive_pairs':npos,'full_negative_pairs':nneg,
      'folds':folds,'cv_positive_test_union':5*pstep,'cv_negative_test_union':5*nstep,
      'independent_positive':len(indpos),'independent_negative':len(indneg),
      'positive_excluded_from_both_cv_and_independent':len(unassigned),
      'independent_and_leftover_interpretation':'Original code ignores these remainder pairs in CV; independent test is defined but not scored by main_optimized.',
      'unassigned_positive_indices_0based':unassigned,
      'full_positive_typed_labels':int(multi.sum()),'full_multitype_pairs':int((multi.sum(2)>1).sum())}
    (HERE/'split_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
