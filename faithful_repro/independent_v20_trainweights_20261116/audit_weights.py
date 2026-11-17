"""Independently reconstruct per-fold weights and compare to recorded loss buffers."""
import ast
import json
from pathlib import Path
import re
import numpy as np
from weights import train_weights

root=Path(__file__).resolve().parent
release=root.parent/'independent_v20_release_20261116/pinned_release/v2.0_495m383D'
rows=np.loadtxt(release/'multi_all_mirna_disease_pairs_without_negative.csv',delimiter=',',dtype=int)
full=np.zeros((495,383),dtype=np.uint8)
for i,j,t in rows: full[i-1,j-1]=t
out=[]
log=(root/'runs/full/console.log').read_text(encoding='utf-8')
for k in range(1,6):
    f=np.load(root/'runs/full'/f'fold_{k}.npz')
    counts,w=train_weights(full,f['train_pos'])
    line=re.search(rf'^TRAIN WEIGHT AUDIT {k} counts (\[[^\]]+\]) weights (\[[^\]]+\])$',log,re.M)
    assert line
    recorded_counts=ast.literal_eval(line.group(1)); recorded_weights=ast.literal_eval(line.group(2))
    assert np.array_equal(counts,recorded_counts) and np.allclose(w,recorded_weights,atol=1e-12,rtol=0)
    out.append({'fold':k,'train_counts':counts.tolist(),'train_weights':w.tolist(),'total':int(counts.sum())})
(root/'weights_audit.json').write_text(json.dumps({'status':'PASS','folds':out},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','folds':out},indent=2))
