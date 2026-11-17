"""Demonstrate test-label exposure in already captured release input features.

This deliberately uses leaked test labels and is NOT a model evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent


def main():
    folds=[]
    for i in range(1,6):
        p=HERE/'runs/smoke'/f'fold_{i}.npz';d=np.load(p)
        y=d['y_multihot'];truth=d['true_scalar_pos'].astype(int)
        exposed=d['feature_scalar_pos'].astype(int)
        assert np.array_equal(exposed,truth) and np.isin(exposed,[1,2,3,4]).all()
        pred=exposed-1;one=np.eye(4,dtype=np.uint8)[pred]
        recalls=[float((pred[truth-1==k]==k).mean()) for k in np.unique(truth-1)]
        p1=float((pred==truth-1).mean());r1=float(np.mean(recalls))
        f=2*p1*r1/(p1+r1)
        assert p1==r1==f==1.0
        folds.append({'fold':i,'n':len(truth),'source_npz_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'exposed_labels_equal_truth':int((exposed==truth).sum()),
          'released_precision':p1,'released_recall':r1,'released_f1':f,
          'multihot_macro_f1':float(f1_score(y,one,average='macro',zero_division=0)),
          'multihot_micro_f1':float(f1_score(y,one,average='micro',zero_division=0)),
          'multihot_any_hit':float(y[np.arange(len(y)),pred].mean())})
    result={'scope':'Intentional test-label reading from input feature, not a fitted model or legitimate generalization metric',
       'n_exposed':sum(x['exposed_labels_equal_truth'] for x in folds),
       'n_test':sum(x['n'] for x in folds),'folds':folds,
       'mean_released_f1':float(np.mean([x['released_f1'] for x in folds])),
       'mean_multihot_macro_f1':float(np.mean([x['multihot_macro_f1'] for x in folds])),
       'mean_multihot_micro_f1':float(np.mean([x['multihot_micro_f1'] for x in folds]))}
    (HERE/'feature_oracle.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
