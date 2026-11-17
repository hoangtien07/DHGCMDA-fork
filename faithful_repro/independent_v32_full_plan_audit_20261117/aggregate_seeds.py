import hashlib,json,pathlib
P=pathlib.Path('faithful_repro/independent_v32_full_plan_audit_20261117');F=P.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();runs=[];base=None
for seed,folder in [(42,'independent_v32_incidence_capture_20261117'),(2025,'independent_v32_full_plan_seed2025_20261117'),(1234,'independent_v32_full_plan_seed1234_20261117')]:
 root=F/folder;R=root/'run';done=json.loads((R/'completion.json').read_text());m=json.loads((R/'out/manifest.json').read_text());assert done['status']=='PASS' and done['manifest_sha256']==sha(R/'out/manifest.json') and m['cfg']['seed']==m['cfg']['inter_seed']==seed
 cfg={k:v for k,v in m['cfg'].items() if k not in ('seed','inter_seed')};props={'cfg':cfg,'source_hashes':m['source_hashes'],'imported_pure_hashes':m['imported_pure_hashes'],'data_hashes':m['data_hashes'],'view_hashes':m['view_hashes']}
 if base is None:base=props
 else:assert props==base
 events=[]
 for e in done['events']:
  if not e['spike']:continue
  ep=e['epoch'];p=json.loads((R/f'paired_{ep:04}.json').read_text());inc=json.loads((R/f'incidence_{ep:04}.json').read_text());assert p['status']==inc['status']=='PASS' and p['checkpoint_sha256']==inc['checkpoint_sha256']==e['sha256']==sha(R/e['checkpoint']);assert p['paired']['repeat_within_tolerance'] and p['paired']['final_rng_equal']
  a=p['paired']['old'];b=p['paired']['new'];i=inc['modalities']['mi'];events.append({'epoch':ep,'previous_loss':e['previous_loss'],'observed_loss':e['loss'],'old_graph_loss':a['loss'],'new_graph_loss':b['loss'],'old_recontribution':a['details']['recontribution'],'new_recontribution':b['details']['recontribution'],'old_grad_norm':a['grad_norm'],'new_grad_norm':b['grad_norm'],'incidence_top13_share':i['top13_degree_sum_share'],'boundary_ties':i['exact_k_boundary_ties']})
 runs.append({'seed':seed,'manifest_sha256':sha(R/'out/manifest.json'),'events':events})
out={'status':'PASS','scope':'three instrumented training runs on same fold; seed and inter_seed change together; not independent splits','runs':runs}
with (P/'three_seed_diagnostics.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,ensure_ascii=False))
