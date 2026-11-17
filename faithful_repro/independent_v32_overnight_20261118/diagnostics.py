import json,hashlib,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;F=P.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
raw={1234:F/'paper_guided/u1_horizon650_fold0_rev1/out',42:F/'independent_v32_seed_stability_20261117/runs/seed42/out',2025:F/'independent_v32_seed_stability_20261117/runs/seed2025/out'}
cos=F/'independent_v32_cosine_training_20261117';results=read(cos/'results.json');assert results['status']=='PASS'
rows=[]
for row in results['rows']:
 seed=row['seed']
 for arm,path in [('raw',raw[seed]),('cosine',cos/f'seed{seed}/out')]:
  m=read(path/'manifest.json');assert sha(path/'manifest.json')==row[arm]['manifest_sha256'];t=m['trajectory'];assert len(t)==650
  points=[];maxerr=0
  for x in t:
   c=m['cfg'];mass=x['unit_mass'];pieces={'type':x['L_type'],'intra':x['lam_intra_applied']*x['L_intra']*(x['intra_n_mi']+x['intra_n_dis'])/mass,'inter':x['lam_inter_applied']*(x['inter_info_sum']+c['inter_margin_coef']*x['inter_margin_term'])/mass,'recon':x['lam_recon_applied']*x['L_recon']*(x['recon_valid_mi']+x['recon_valid_dis'])/mass};err=abs(sum(pieces.values())-x['train_loss']);maxerr=max(maxerr,err)
   if x['epoch'] in (1,10,100,200,400,650):points.append({'epoch':x['epoch'],'loss':x['train_loss'],'components':pieces,'recon_fraction':pieces['recon']/x['train_loss'],'grad_norm':x['grad_norm'],'val_macro_f1':x.get('val_macro_f1')})
  assert maxerr<1e-4,(seed,arm,maxerr)
  rows.append({'seed':seed,'arm':arm,'manifest_sha256':sha(path/'manifest.json'),'max_loss_identity_error':maxerr,'points':points,'per_class':row[arm]['per_class'],'best_epoch':row[arm]['best_epoch']})
out={'status':'PASS','scope':'post-hoc descriptive trajectory, contributions are objective values, not component gradient attribution','plan_sha256':sha(P/'PLAN.md'),'script_sha256':sha(__file__),'rows':rows}
with (P/'diagnostics.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for r in rows:
 print(r['seed'],r['arm'],'recon fraction@1/100/650:',*[round(x['recon_fraction'],3) for x in r['points'] if x['epoch'] in (1,100,650)],'identity_err',r['max_loss_identity_error'])
