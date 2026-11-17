"""Verify completion, telemetry reconstruction and seal evidence."""
import hashlib
import json
import math
from pathlib import Path
P=Path(__file__).resolve().parent;FR=P.parents[0]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
metrics=json.loads((P/'seed_metrics.json').read_text(encoding='utf-8'))
loss=json.loads((P/'loss_audit.json').read_text(encoding='utf-8'))
trans=json.loads((P/'transition_audit.json').read_text(encoding='utf-8'))
traj=json.loads((P/'trajectory_audit.json').read_text(encoding='utf-8'))
assert metrics['status']==loss['status']==trans['status']==traj['status']=='PASS'
assert metrics['completed_seeds']==len(loss['runs'])==len(trans['runs'])==3 and len(traj['folds'])==5
assert all(sum(v['n'] for v in r['groups'].values())==649 for r in trans['runs'])
files=[P/x for x in ('PROTOCOL.md','REPORT.md','run_seed.py','score_seeds.py','trajectory_audit.py','loss_audit.py','transition_audit.py','verify.py',
                       'seed_metrics.json','trajectory_audit.json','loss_audit.json','transition_audit.json')]
for seed in (42,2025):
    run=P/'runs'/f'seed{seed}';launch=json.loads((run/'launch.json').read_text(encoding='utf-8'));done=json.loads((run/'completion.json').read_text(encoding='utf-8'))
    assert launch['protocol_sha256']==sha(P/'PROTOCOL.md') and launch['runner_sha256']==sha(P/'run_seed.py')
    assert done['aborted'] is None and done['final_epoch']==650 and done['manifest_sha256']==sha(run/'out/manifest.json')
    m=json.loads((run/'out/manifest.json').read_text(encoding='utf-8'))
    for r in m['trajectory']:
        a=float(r['L_recon'])*float(r['unit_recon_factor'])
        b=(float(r['recon_sum_mi'])+float(r['recon_sum_dis']))/float(r['unit_mass'])
        assert math.isclose(a,b,rel_tol=2e-6,abs_tol=1e-6),(seed,r['epoch'],a,b)
    files.extend(run/x for x in ('launch.json','expected.json','completion.json','progress.log','out/manifest.json',
                                 'out/predictions_val.npz','out/predictions_val.meta.json'))
manifest={str(x.relative_to(P)).replace('\\','/'):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in files}
(P/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f'PASS: 3 seed metrics, 5 fold trajectories, 1300 reconstruction reductions, {len(files)} files sealed')
