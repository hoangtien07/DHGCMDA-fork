import hashlib,json,pathlib,math
P=pathlib.Path('faithful_repro/independent_v32_hypergraph_path_20261117');F=P.parent;CAP=F/'independent_v32_refresh_capture_20261117/run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((P/'hypergraph.json').read_text());b=json.loads((P/'row_intervention.json').read_text());assert a['status']==b['status']=='PASS'
assert a['protocol_sha256']==sha(P/'PROTOCOL.md') and a['script_sha256']==sha(P/'hypergraph.py')
assert a['selected_reference_sha256']==sha(F/'independent_v32_decoder_gate_20261117/rows.json') and a['trace_sha256']==sha(F/'independent_v32_spike_path_20261117/trace.json')
assert b['protocol_sha256']==sha(P/'PROTOCOL_ROWS.md') and b['script_sha256']==sha(P/'row_intervention.py') and b['hypergraph_sha256']==sha(P/'hypergraph.json')
for x,y in zip(a['runs'],b['runs']):
 assert x['epoch']==y['epoch'] and x['checkpoint_sha256']==y['checkpoint_sha256']==sha(CAP/('spike_%04d.pt'%x['epoch']))
 for side in ('old','new'):
  assert math.isclose(x[side]['residual_sum'],y['results'][side]['residual_sum'],rel_tol=1e-5)
  assert all(q['relative_decomposition_error']<1e-5 for q in x[side]['convs'].values())
 for side in ('old_plus_rows','new_minus_rows'):assert y['results'][side]['asymmetry_norm']>0
print('PASS: checkpoints, protocols, hashes, conv reconstruction, counterfactual corners and asymmetry')
