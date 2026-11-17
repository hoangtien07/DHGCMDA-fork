import hashlib,json,pathlib,math
P=pathlib.Path('faithful_repro/independent_v32_refresh_series_20261117');R=P/'run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((R/'out/manifest.json').read_text());d=json.loads((R/'completion.json').read_text());s=json.loads((P/'series.json').read_text());l=json.loads((R/'launch.json').read_text())
assert d['status']==s['status']=='PASS' and d['final_epoch']==650 and d['capture_count']==129 and len(s['records'])==130 and s['manifest_sha256']==d['manifest_sha256']==sha(R/'out/manifest.json')
assert l['script_sha256']==s['capture_sha256']==sha(P/'capture.py') and l['protocol_sha256']==s['protocol_sha256']==sha(P/'PROTOCOL.md') and s['script_sha256']==sha(P/'analyze.py')
for r in s['records']:assert r['file_sha256']==sha(R/('refresh_%04d.pt'%r['refresh_epoch']))
t=m['trajectory'];sp=[b['epoch'] for a,b in zip(t,t[1:]) if b['train_loss']>2*a['train_loss']];assert sp==[e['epoch'] for e in d['events'] if e['spike']]==[r['refresh_epoch']+1 for r in s['records'] if r.get('spike')]
for e in d['events']:
 ep=e['epoch'];ef=R/('event_%04d.json'%ep);inc=json.loads((R/('incidence_%04d.json'%ep)).read_text());p=json.loads((R/('paired_%04d.json'%ep)).read_text());assert sha(R/e['checkpoint'])==inc['checkpoint_sha256']==p['checkpoint_sha256']==e['sha256'] and inc['event_sha256']==p['event_sha256']==sha(ef);assert inc['script_sha256']==sha(P/'incidence.py') and p['evaluator_sha256']==sha(P/'evaluate.py') and p['paired']['repeat_within_tolerance'] and p['paired']['final_rng_equal'] and math.isclose(p['paired']['new']['loss'],e['loss'],rel_tol=1e-5)
 h=next(x for x in m['dynamic_history'] if x['epoch']==ep-1)
 for mod in ('mi','dis'):
  z=inc['modalities'][mod];assert z['producer_sha256']==h['producer_gram_'+mod+'_sha256'] and z['normalized_sha256']==h['normalized_'+mod+'_sha256'] and z['incidence_sha256']==h['incidence_'+mod+'_sha256'] and z['G_sha256']==h['G_'+mod+'_v1_sha256']
print('PASS: 130 refresh files, 650 epochs, event coverage, checkpoints, incidence and paired gates; spike epochs',sp)
