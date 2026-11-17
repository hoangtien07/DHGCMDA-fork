"""Recompute stored functional inputs and quantify coverage on test pairs."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from functional import functional_similarity
from mask import train_matrix

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'independent_v20_release_20261116/pinned_release/v2.0_495m383D'
run=HERE/'runs/full';log=(run/'console.log').read_text(encoding='utf-8')
csv=np.loadtxt(UP/'multi_all_mirna_disease_pairs_without_negative.csv',delimiter=',',dtype=int)
full=np.zeros((495,383),dtype=np.uint8)
for i,j,t in csv:full[i-1,j-1]=t
dd=np.loadtxt(UP/'D_SSM1.txt',dtype=np.float32);release=np.loadtxt(UP/'M_FSM.txt',dtype=np.float32)
results=[]
for i in range(1,6):
    fold=np.load(run/f'fold_{i}.npz'); feature=run/f'functional_fold_{i}.npy';actual=np.load(feature)
    masked=train_matrix(full,fold['train_pos'],fold['test_pos']); calculated=functional_similarity(masked,dd)
    assert np.allclose(actual,calculated,atol=1e-7) and np.isfinite(actual).all()
    digest=hashlib.sha256(feature.read_bytes()).hexdigest()
    assert re.search(rf'^FUNCTION AUDIT {i} sha256 {digest} ',log,re.M)
    test=fold['test_pos'];cold=(masked!=0).sum(1)[test[:,0]]==0
    score=fold['score_pos'][:,1:5];pred=score.argmax(1);truth=fold['true_scalar_pos'].astype(int)-1
    results.append({'fold':i,'feature_sha256':digest,'n_zero_degree_mirnas':int(((masked!=0).sum(1)==0).sum()),
      'n_test_pairs_with_zero_degree_mirna':int(cold.sum()),'n_test_pairs':len(test),
      'scalar_top1_accuracy_cold':float((pred[cold]==truth[cold]).mean()) if cold.any() else None,
      'scalar_top1_accuracy_other':float((pred[~cold]==truth[~cold]).mean()) if (~cold).any() else None,
      'mean_abs_diff_from_release_feature':float(np.abs(actual-release).mean())})
out={'status':'PASS','folds':results,'interpretation':'Stratified scalar accuracy is descriptive; not a randomized estimate of cold-start effect.'}
(HERE/'feature_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
