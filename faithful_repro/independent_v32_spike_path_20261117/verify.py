import hashlib,json,pathlib,math
p=pathlib.Path('faithful_repro/independent_v32_spike_path_20261117');c=p.parent/'independent_v32_refresh_capture_20261117/run';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
t=json.loads((p/'trace.json').read_text(encoding='utf-8'));n=json.loads((p/'nodes.json').read_text(encoding='utf-8'))
assert t['status']==n['status']=='PASS' and t['protocol_sha256']==sha(p/'PROTOCOL.md') and t['script_sha256']==sha(p/'trace.py')
assert n['protocol_sha256']==sha(p/'PROTOCOL_NODES.md') and n['script_sha256']==sha(p/'nodes.py') and n['trace_sha256']==sha(p/'trace.json')
for a,b in zip(t['runs'],n['runs']):
 assert a['epoch']==b['epoch'] and a['checkpoint_sha256']==b['checkpoint_sha256']==sha(c/('spike_%04d.pt'%a['epoch']))
 pair=json.loads((c/('paired_%04d.json'%a['epoch'])).read_text(encoding='utf-8'))['paired']
 for side in ('old','new'):
  assert math.isclose(a[side]['residual_sum'],pair[side]['details']['recon_sum_mi'],rel_tol=1e-5)
  d=b[side]['reconstruction_decomposition'];assert math.isclose(d['gram_sq']+d['minus_two_cross']+d['target_sq'],d['residual_sum'],rel_tol=1e-10)
  assert math.isclose(d['residual_sum'],a[side]['residual_sum'],rel_tol=1e-5)
 assert a['old']['rng_end_sha256']==a['new']['rng_end_sha256'] and math.isclose(a['repeat_new_residual'],a['new']['residual_sum'],rel_tol=1e-5)
print('PASS: checkpoint and script hashes, paired reference, RNG, repeat, residual decomposition for 2 events')
