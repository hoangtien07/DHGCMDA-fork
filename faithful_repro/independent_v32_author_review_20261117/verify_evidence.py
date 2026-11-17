import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
read=lambda x:json.loads((p/x).read_text(encoding='utf-8-sig'))
m=read('peer_review_manifest.json')
for f in m['files']:
 assert sha(p/f['file'])==f['sha256']
 assert sha((p/f['file']).with_suffix('.txt'))==f['text_sha256']
for name in ['parameter_audit','objective_trace']:
 j=read(name+'.json');assert j['status']=='PASS' and sha(p/(name+'.py'))==j['script_sha256']
j=read('cosine_counterfactual.json');assert j['status']=='PASS' and sha(p/'cosine_counterfactual.py')==j['script_sha256'] and sha(p/'PROTOCOL_COSINE.md')==j['protocol_sha256'];assert len(j['cases'])==4
folders={42:'independent_v32_incidence_capture_20261117',2025:'independent_v32_full_plan_seed2025_20261117',1234:'independent_v32_full_plan_seed1234_20261117'}
for c in j['cases']:
 d=p.parent/folders[c['seed']]/'run';event=json.loads((d/f"event_{c['epoch']:04d}.json").read_text()); assert sha(d/event['checkpoint'])==c['checkpoint_sha256']==event['sha256']
 assert c['raw']['rng_end_sha256']==c['cosine']['rng_end_sha256'] and c['cosine_repeat_grad_relative_difference']<1e-4
 assert c['cosine']['grad_norm']<c['raw']['grad_norm']
print('PASS: 7 PDF/text hashes, audit script hashes, cosine protocol/script, 4 checkpoint hashes and recorded controls')
