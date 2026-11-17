import hashlib,json,math,pathlib
P=pathlib.Path('faithful_repro/independent_v32_decoder_gate_20261117');CAP=P.parent/'independent_v32_refresh_capture_20261117/run';T=P.parent/'independent_v32_spike_path_20261117/trace.json';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
g=json.loads((P/'counterfactual.json').read_text());r=json.loads((P/'rows.json').read_text());assert g['status']==r['status']=='PASS'
for a,script,protocol in [(g,'counterfactual.py','PROTOCOL.md'),(r,'rows.py','PROTOCOL_ROWS.md')]:
 assert a['script_sha256']==sha(P/script) and a['protocol_sha256']==sha(P/protocol) and a['reference_trace_sha256']==sha(T)
 for e in a['runs']:
  epoch=e['epoch'];assert e['checkpoint_sha256']==sha(CAP/('spike_%04d.pt'%epoch));p=json.loads((CAP/('paired_%04d.json'%epoch)).read_text())['paired']
  for mode,side in ([('oldmag_oldgate','old'),('newmag_newgate','new')] if script=='counterfactual.py' else [('old','old'),('new','new')]):assert math.isclose(e['results'][mode]['residual_sum'],p[side]['details']['recon_sum_mi'],rel_tol=1e-5)
  if script=='rows.py':
   assert len(set(e['diagnostics']['selected_indices']))==13
   for v in e['results'].values():assert math.isclose(v['touch_residual']+v['untouched_residual'],v['residual_sum'],rel_tol=1e-10)
print('PASS: checkpoint/protocol/script/reference hashes; original residuals; 13-node selection and partition')
