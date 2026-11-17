import hashlib,json,math
from pathlib import Path
P=Path('faithful_repro/independent_v32_author_review_20261117');root=P.parent/'independent_v32_outer_refit_20261117/runs';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=[]
for fold in range(5):
 run=root/f'fold{fold}_seed1234_dynamic/out';p=run/'trajectory_noval.json';tr=json.loads(p.read_text());assert len(tr)==650
 maxdiff=0.
 for x in tr:
  mass=float(x['unit_mass']);assert mass>0 and x['lam_recon_applied']==1.0
  a=float(x['L_recon'])*float(x['unit_recon_factor']);b=(float(x['recon_sum_mi'])+float(x['recon_sum_dis']))/mass
  diff=abs(a-b)/max(1,abs(b));maxdiff=max(maxdiff,diff)
  assert math.isclose(a,b,rel_tol=2e-6,abs_tol=1e-6)
 x=tr[0];out.append({'fold':fold,'mass':x['unit_mass'],'lambda3':x['lam_recon_applied'],'reciprocal_mass':1/x['unit_mass'],'valid_mi':x['recon_valid_mi'],'valid_dis':x['recon_valid_dis'],'first_epoch_masked_recon_sum':x['recon_sum_mi']+x['recon_sum_dis'],'first_epoch_effective_recontribution':x['L_recon']*x['unit_recon_factor'],'max_scaled_identity_error':maxdiff,'trajectory_sha256':sha(p)})
res={'status':'PASS','scope':'5 trajectories, 3250 reduction identities; conditional accounting not paper-author-code identity','rows':out,'script_sha256':sha(Path(__file__))}
with (P/'objective_trace.json').open('x',encoding='utf-8') as f:json.dump(res,f,indent=2);f.write('\n')
print(json.dumps(res))
