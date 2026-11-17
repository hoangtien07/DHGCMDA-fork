"""Summarize recorded loss/gradient spikes without inferring their cause."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
paths=[(1234,FR/'paper_guided/u1_horizon650_fold0_rev1/out/manifest.json'),
       (42,HERE/'runs/seed42/out/manifest.json'),(2025,HERE/'runs/seed2025/out/manifest.json')]
rows=[]
for seed,p in paths:
    if not p.is_file():continue
    m=json.loads(p.read_text(encoding='utf-8'))
    if m.get('final_epoch')!=650 or m.get('aborted') is not None:continue
    tr=m['trajectory'];assert len(tr)==650
    vals=np.array([r['train_loss'] for r in tr],dtype=float);grads=np.array([r['grad_norm'] for r in tr],dtype=float)
    assert np.isfinite(vals).all() and np.isfinite(grads).all()
    k=int(np.argmax(vals));r=tr[k];prev=tr[k-1] if k else None
    recon=float(r['L_recon']*r['unit_recon_factor']*r['lam_recon_applied'])
    rows.append({'seed':seed,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'max_loss_epoch':int(r['epoch']),'max_loss':float(vals[k]),'median_loss':float(np.median(vals)),
        'recontribution':recon,'recontribution_fraction':recon/float(vals[k]),
        'mirna_recon_sum':float(r['recon_sum_mi']),'disease_recon_sum':float(r['recon_sum_dis']),
        'grad_norm_at_max_loss':float(grads[k]),'max_grad_norm':float(grads.max()),
        'graph_epoch_at_max_loss':int(r['graph_epoch']),
        'refresh_logged_at_max_loss':bool(r['dynamic_refreshed']),
        'refresh_logged_previous_epoch':bool(prev['dynamic_refreshed']) if prev else None,
        'previous_loss':float(prev['train_loss']) if prev else None})
out={'status':'PASS','runs':rows,'limits':'Observed telemetry and timing only; refresh/loss association is not proof of cause; seeds and platforms differ across runs.'}
(HERE/'loss_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
