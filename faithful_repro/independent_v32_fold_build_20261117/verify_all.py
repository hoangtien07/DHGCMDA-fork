"""Check all completed outputs and seal compact evidence with hashes."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent;FR=P.parents[0]
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
audit=json.loads((P/'audit.json').read_text(encoding='utf-8'))
baseline=json.loads((P/'validation_baselines.json').read_text(encoding='utf-8'))
scored=json.loads((P/'u1_partial_metrics.json').read_text(encoding='utf-8'))
exposure=json.loads((P/'exposure.json').read_text(encoding='utf-8'))
assert audit['status']==exposure['status']=='PASS' and len(audit['folds'])==3
assert len(baseline['folds'])==scored['completed_folds']==len(scored['folds'])==5
assert all(r['heldout_entries_in_mask']==r['group_crossings']==0 for r in audit['folds'])
assert all(r['functional_and_semantic_rebuild_elementwise_equal'] for r in audit['folds'])
files=[P/x for x in ('PLAN.md','BASELINE_PROTOCOL.md','U1_PROTOCOL.md','BASELINE_REPORT.md','U1_REPORT.md',
       'build.py','audit_bundles.py','validation_baselines.py','exposure.py','run_u1.py','score_u1.py','verify_all.py',
       'audit.json','validation_baselines.json','exposure.json','u1_partial_metrics.json')]
for r in audit['folds']:
    i=r['fold'];b=P/'bundles'/f'fold{i}_selection';run=P/'u1_selection'/f'fold{i}'
    assert sha(b/'candidate_fold_data.npz')==r['bundle_sha256']==scored['folds'][i]['bundle_sha256']
    launch=json.loads((run/'launch.json').read_text(encoding='utf-8'))
    comp=json.loads((run/'completion.json').read_text(encoding='utf-8'))
    assert launch['protocol_sha256']==sha(P/'U1_PROTOCOL.md') and launch['runner_sha256']==sha(P/'run_u1.py')
    assert launch['bundle_sha256']==r['bundle_sha256'] and comp['aborted'] is None and comp['final_epoch']==650
    assert comp['manifest_sha256']==sha(run/'out/manifest.json')
    assert sha(run/'out/predictions_val.npz')==scored['folds'][i]['prediction_sha256']
    files += [b/x for x in ('candidate_fold_data.npz','candidate_fold_data.meta.json','recon_meta.json','recon_audits.json')]
    files += [run/x for x in ('launch.json','expected.json','completion.json','progress.log','out/manifest.json',
                              'out/predictions_val.npz','out/predictions_val.meta.json')]
    files += [P/f'fold{i}_builder.stdout.log',P/f'fold{i}_builder.stderr.log']
assert len(scored['mean_over_folds'])==3
manifest={str(p.relative_to(P)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
(P/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f'PASS: 3 audited input bundles and runs, 5 validation results, {len(files)} evidence files sealed')
