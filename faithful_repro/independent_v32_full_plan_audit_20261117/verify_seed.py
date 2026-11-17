import hashlib,json,math,pathlib,sys
seed=int(sys.argv[1]);P=pathlib.Path(f'faithful_repro/independent_v32_full_plan_seed{seed}_20261117');R=P/'run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads((R/'completion.json').read_text());l=json.loads((R/'launch.json').read_text());m=json.loads((R/'out/manifest.json').read_text());assert d['status']=='PASS' and d['final_epoch']==650 and d['capture_count']==129 and d['manifest_sha256']==sha(R/'out/manifest.json');assert l['protocol_sha256']==sha(P/'PROTOCOL.md') and l['script_sha256']==sha(P/'capture.py');assert m['cfg']['seed']==m['cfg']['inter_seed']==seed
t=m['trajectory'];assert len(t)==650;spikes=[(b['epoch'],bool(a['dynamic_refreshed'])) for a,b in zip(t,t[1:]) if b['train_loss']>2*a['train_loss']];assert [i[0] for i in spikes]==[e['epoch'] for e in d['events'] if e['spike']]
for e in d['events']:
 ep=e['epoch'];ef=R/('event_%04d.json'%ep);cp=R/e['checkpoint'];assert sha(cp)==e['sha256'];inc=json.loads((R/('incidence_%04d.json'%ep)).read_text());paired=json.loads((R/('paired_%04d.json'%ep)).read_text());assert inc['checkpoint_sha256']==paired['checkpoint_sha256']==e['sha256'] and inc['event_sha256']==paired['event_sha256']==sha(ef);assert inc['script_sha256']==sha(P/'incidence.py') and paired['evaluator_sha256']==sha(P/'evaluate.py');assert paired['paired']['repeat_within_tolerance'] and paired['paired']['final_rng_equal'] and math.isclose(paired['paired']['new']['loss'],e['loss'],rel_tol=1e-5)
 hist=next(h for h in m['dynamic_history'] if h['epoch']==ep-1)
 for mod in ('mi','dis'):
  z=inc['modalities'][mod];assert z['producer_sha256']==hist['producer_gram_'+mod+'_sha256'] and z['normalized_sha256']==hist['normalized_'+mod+'_sha256'] and z['incidence_sha256']==hist['incidence_'+mod+'_sha256'] and z['G_sha256']==hist['G_'+mod+'_v1_sha256']
out={'status':'PASS','seed':seed,'manifest_sha256':sha(R/'out/manifest.json'),'spikes':spikes,'event_count':len(d['events'])}
with (P/'verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
