"""Score each completed full-run fold without touching any in-progress fold."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/full'


def main():
    log=(OUT/'console.log').read_text(encoding='utf-8')
    completed={int(i):float(f) for i,f in re.findall(r'Fold (\d+) completed - AUC: [0-9.]+, Top-1 F1: ([0-9.]+)',log)}
    results=[]
    for i in sorted(completed):
        path=OUT/f'fold_{i}.npz';b=path.read_bytes();d=np.load(path)
        y=d['y_multihot'];truth=d['true_scalar_pos'].astype(int)-1;scores=d['score_pos'][:,1:5]
        assert np.count_nonzero(d['feature_scalar_pos'])==0
        pred=scores.argmax(1);one=np.eye(4,dtype=int)[pred]
        p=float(np.mean(pred==truth));r=float(np.mean([np.mean(pred[truth==k]==k) for k in np.unique(truth)]))
        f=2*p*r/(p+r) if p+r else 0.
        assert abs(f-completed[i])<0.000051
        results.append({'fold':i,'n_positive':len(truth),'released_precision':p,'released_recall':r,'released_f1':f,
          'multihot_macro_f1':float(f1_score(y,one,average='macro',zero_division=0)),
          'multihot_micro_f1':float(f1_score(y,one,average='micro',zero_division=0)),
          'any_hit':float(y[np.arange(len(y)),pred].mean()),
          'n_test_types_in_feature':int((d['feature_scalar_pos']==d['true_scalar_pos']).sum()),
          'score_distinct_rows':len(np.unique(scores,axis=0)),'npz_sha256':hashlib.sha256(b).hexdigest()})
    out={'scope':'completed folds only; train-only association, GIP, functional similarity and class weights; disease priors unverified',
         'completed_folds':len(results),'folds':results}
    (OUT/'partial_metric_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
