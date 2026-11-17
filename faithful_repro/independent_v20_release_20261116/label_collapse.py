"""Quantify release loader's last-write label collapse from its pinned CSV."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'pinned_release/v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv'


def main():
    y=np.zeros((495,383,4),dtype=np.uint8);scalar=np.zeros((495,383),dtype=np.uint8)
    seq=[]
    with DATA.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.reader(f):
            i,j,k=map(int,row[:3]);i-=1;j-=1
            y[i,j,k-1]=1;scalar[i,j]=k;seq.append(k)
    kcount=y.sum(2);present=kcount>0
    original=y.sum((0,1)).astype(int)
    retained=np.bincount(scalar[present],minlength=5)[1:]
    lost=original-retained
    assert (original==[443,199,356,681]).all() and (retained==[367,157,293,681]).all()
    assert int(lost.sum())==181 and int((kcount>1).sum())==161
    fold=[]
    for i in range(1,6):
        a=np.load(HERE/'runs/smoke'/f'fold_{i}.npz');p=a['test_pos']
        true=y[p[:,0],p[:,1]];s=scalar[p[:,0],p[:,1]]
        assert np.array_equal(true,a['y_multihot']) and np.array_equal(s,a['true_scalar_pos'])
        fold.append({'fold':i,'n_test':len(p),'true_labels':int(true.sum()),
                     'discarded_type_labels':int(true.sum()-len(p)),'multitype_pairs':int((true.sum(1)>1).sum())})
    out={'csv_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
      'channel_order':['circulation','epigenetics','target','genetics'],
      'unique_typed_links_before_collapse':int(y.sum()),'positive_pairs_after_collapse':int(present.sum()),
      'pairs_with_multiple_types':int((kcount>1).sum()),
      'type_count_before_collapse':original.tolist(),'last_write_count':retained.tolist(),
      'typed_links_not_represented_as_scalar_label':lost.tolist(),
      'csv_types_monotonic_nondecreasing':bool(np.all(np.diff(seq)>=0)),
      'fold_test_label_loss':fold,
      'interpretation':'Deterministic loss of labels in release scalar matrix. Does not show this was the authors paper-experiment implementation.'}
    (HERE/'label_collapse_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
