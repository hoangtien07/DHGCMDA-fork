"""Retrospective loss jumps conditioned on previous-epoch refresh telemetry."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
paths=[(1234,FR/'paper_guided/u1_horizon650_fold0_rev1/out/manifest.json'),
       (42,HERE/'runs/seed42/out/manifest.json'),(2025,HERE/'runs/seed2025/out/manifest.json')]
rows=[]
for seed,p in paths:
    m=json.loads(p.read_text(encoding='utf-8'));t=m['trajectory'];assert len(t)==650 and m['aborted'] is None
    group={True:[],False:[]}
    for prev,cur in zip(t,t[1:]):
        assert cur['epoch']==prev['epoch']+1 and prev['train_loss']>0
        group[bool(prev['dynamic_refreshed'])].append(float(cur['train_loss']/prev['train_loss']))
    counts={}
    for flag,vals in group.items():
        a=np.array(vals);assert np.isfinite(a).all()
        counts['after_refresh' if flag else 'other']={'n':len(a),'ratio_gt_2':int((a>2).sum()),
          'ratio_gt_5':int((a>5).sum()),'median_ratio':float(np.median(a)),
          'max_ratio':float(a.max())}
    rows.append({'seed':seed,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'groups':counts})
out={'status':'PASS','scope':'retrospective per-step adjacency, threshold specified after observing spikes',
     'runs':rows,'limits':'Temporal association only; comparison is observational and does not establish that refresh causes spikes.'}
(HERE/'transition_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
