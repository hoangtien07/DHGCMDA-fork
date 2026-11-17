import hashlib,json,math,pathlib
P=pathlib.Path('faithful_repro/independent_v32_refresh_capture_20261117');R=P/'run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
completion=json.loads((R/'completion.json').read_text(encoding='utf-8'));launch=json.loads((R/'launch.json').read_text(encoding='utf-8'));m=json.loads((R/'out/manifest.json').read_text(encoding='utf-8'))
assert completion['status']=='PASS' and completion['final_epoch']==650 and completion['capture_count']==129 and completion['manifest_sha256']==sha(R/'out/manifest.json')
assert launch['protocol_sha256']==sha(P/'PROTOCOL.md') and launch['script_sha256']==sha(P/'capture.py')
tr=m['trajectory'];assert len(tr)==650
allspikes=[(b['epoch'],b['train_loss']/a['train_loss'],bool(a['dynamic_refreshed'])) for a,b in zip(tr,tr[1:]) if b['train_loss']>2*a['train_loss']]
events=[e for e in completion['events'] if e['spike']];assert [x[0] for x in allspikes]==[e['epoch'] for e in events]
for e in completion['events']:
 assert sha(R/e['checkpoint'])==e['sha256']
 pair=json.loads((R/('paired_%04d.json'%e['epoch'])).read_text(encoding='utf-8'));assert pair['checkpoint_sha256']==e['sha256'] and pair['event_sha256']==sha(R/('event_%04d.json'%e['epoch']))
 assert pair['evaluator_sha256']==sha(P/'evaluate.py') and pair['paired']['repeat_within_tolerance'] and pair['paired']['final_rng_equal']
 assert math.isclose(pair['paired']['new']['loss'],e['loss'],rel_tol=1e-5)
for epoch in (226,591):
 for prefix,script,protocol in [('factors','factors.py' if epoch==226 else 'factors_0591.py','PROTOCOL_FACTORS.md' if epoch==226 else 'PROTOCOL_0591.md'),('g_parts','g_parts.py' if epoch==226 else 'g_parts_0591.py','PROTOCOL_G_PARTS.md' if epoch==226 else 'PROTOCOL_0591.md')]:
  j=json.loads((P/('%s_%04d.json'%(prefix,epoch))).read_text(encoding='utf-8'));assert j['script_sha256']==sha(P/script) and j['protocol_sha256']==sha(P/protocol) and j['snapshot_sha256']==sha(R/('spike_%04d.pt'%epoch))
  assert all(x['repeat_within_tolerance'] and x['final_rng_equal'] for x in j['rows'].values())
print('PASS 650 epochs, hashes, event coverage, replay and factor repeat gates; all >2x rises:',allspikes)
