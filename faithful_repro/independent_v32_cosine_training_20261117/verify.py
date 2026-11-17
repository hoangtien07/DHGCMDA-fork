import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;f=p.parent
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
read=lambda x:json.loads(Path(x).read_text(encoding='utf-8-sig'))
r=read(p/'results.json');assert r['status']=='PASS' and r['report_script_sha256']==sha(p/'report.py') and r['protocol_sha256']==sha(p/'PROTOCOL.md')
raw={1234:f/'paper_guided/u1_horizon650_fold0_rev1/out',42:f/'independent_v32_seed_stability_20261117/runs/seed42/out',2025:f/'independent_v32_seed_stability_20261117/runs/seed2025/out'}
for row in r['rows']:
 seed=row['seed'];a=read(raw[seed]/'manifest.json');b=read(p/f'seed{seed}/out/manifest.json');c=read(p/f'seed{seed}/completion.json')
 assert a['cfg']==b['cfg'] and a['data_hashes']==b['data_hashes'] and a['view_hashes']==b['view_hashes']
 assert a['source_hashes']==b['source_hashes'] and a['imported_pure_hashes']==b['imported_pure_hashes']
 assert c['manifest_sha256']==sha(p/f'seed{seed}/out/manifest.json')==row['cosine']['manifest_sha256']
 assert sha(raw[seed]/'manifest.json')==row['raw']['manifest_sha256']
 assert a['final_epoch']==b['final_epoch']==650 and a['aborted'] is None and b['aborted'] is None
 assert row['raw']['pair_truth_sha256']==row['cosine']['pair_truth_sha256']
 print(seed,'cfg, data, source, checkpoint run and prediction pairs PASS')
print('PASS: 3 paired controls; cosine intervention defined by separately hashed training wrapper')
