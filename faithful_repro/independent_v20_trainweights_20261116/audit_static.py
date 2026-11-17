"""Inspect static source matrices and their association-profile correlations."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'independent_v20_release_20261116'
DATA=BASE/'pinned_release/v2.0_495m383D'
commit='06f9ba3eed92e334d9a2b8003fc06d83596e5614'
blob=subprocess.check_output(['git','cat-file','blob',commit+':v2.0_495m383D/all_mirna_disease_pairs.csv'])
assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()=='3b1a253895ce800bddaabea96017510218a543e7'
rows=np.loadtxt(io.BytesIO(blob),delimiter=',',dtype=int)
binary=np.zeros((495,383),dtype=np.uint8); binary[rows[:,0]-1,rows[:,1]-1]=rows[:,2]
typed=np.loadtxt(DATA/'multi_all_mirna_disease_pairs_without_negative.csv',delimiter=',',dtype=int)
association=np.zeros_like(binary);association[typed[:,0]-1,typed[:,1]-1]=1
matrices={name:np.loadtxt(DATA/name) for name in ('M_FSM.txt','M_GSM.txt','D_SSM1.txt','D_SSM2.txt')}
result={'source_commit':commit,'static_matrices':{},'association_correlation':{}}
for name,x in matrices.items():
    result['static_matrices'][name]={'shape':list(x.shape),'sha256':hashlib.sha256((DATA/name).read_bytes()).hexdigest(),
      'asymmetry_max':float(np.max(np.abs(x-x.T))),'all_finite':bool(np.isfinite(x).all())}
for name,a in [('binary_5430',binary),('typed_1498',association)]:
    aa=a.astype(np.float64); shared=aa@aa.T; sizes=aa.sum(1);union=sizes[:,None]+sizes[None,:]-shared
    iu=np.triu_indices(len(a),1); j=np.divide(shared,union,out=np.zeros_like(shared),where=union!=0)[iu]
    x=matrices['M_FSM.txt'][iu]; same=shared[iu]>0
    result['association_correlation'][name]={'positive_pairs':int(a.sum()),'pearson':float(pearsonr(x,j).statistic),
      'spearman':float(spearmanr(x,j).statistic),'similarity_mean_if_shared_disease':float(x[same].mean()),
      'similarity_mean_if_no_shared_disease':float(x[~same].mean()),'n_mirna_pairs':len(x)}
result['limitation']='Correlation is descriptive and does not prove static feature was built using held-out associations.'
(HERE/'static_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
