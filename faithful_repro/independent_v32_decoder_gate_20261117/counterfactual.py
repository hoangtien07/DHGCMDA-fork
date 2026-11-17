import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';CAP=FR/'independent_v32_refresh_capture_20261117/run';TRACE=FR/'independent_v32_spike_path_20261117/trace.json'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
runs=[]
for epoch in (226,591):
 ev=json.loads((CAP/('event_%04d.json'%epoch)).read_text());path=CAP/ev['checkpoint'];assert sha(path)==ev['sha256'];s=torch.load(path,map_location='cpu',weights_only=True);c=s['cfg'];I=s['inputs']
 model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));model.load_state_dict(s['model_state'],strict=True);model.train();head=model.recon.heads['mi'];target=s['targets']['mi_target'];valid=s['targets']['mi_valid'].bool(); data={}
 with torch.no_grad():
  for name,graph in [('old',s['old_graph']),('new',s['new_graph'])]:
   torch.set_rng_state(s['torch_rng'].clone());z,_,_,gates=model.mi_view(I['mi_v1'],graph[0]['mi_v1'],I['mi_v2'],graph[0]['mi_v2']);a=head.fc1(z);q=torch.relu(a);p=head.fc2(q);gram=p@p.T;err=(gram[valid]-target[valid]).double();data[name]={'z':z,'a':a,'q':q,'p':p,'sse':float(err.square().sum())}
  old,new=data['old'],data['new'];results={}
  for mag in ('old','new'):
   for gate in ('old','new'):
    q=data[mag]['a'].abs()*(data[gate]['a']>0);p=head.fc2(q);gram=p@p.T;v=(gram[valid]-target[valid]).double()
    results[mag+'mag_'+gate+'gate']={'residual_sum':float(v.square().sum()),'active_norm':float(q.double().norm()),'projection_norm':float(p.double().norm()),'projection_row_max':float(p.double().norm(dim=1).max())}
  ref=json.loads((CAP/('paired_%04d.json'%epoch)).read_text())['paired']
  for key,side in [('oldmag_oldgate','old'),('newmag_newgate','new')]:assert abs(results[key]['residual_sum']-ref[side]['details']['recon_sum_mi'])/max(1,ref[side]['details']['recon_sum_mi'])<1e-5
  oldmask=old['a']>0;newmask=new['a']>0;change=oldmask!=newmask
  aold=old['a'].double();anew=new['a'].double();po=old['p'].double();pn=new['p'].double();zo=old['z'].double();zn=new['z'].double()
  cosine=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()))
  extra={'changed_sign_fraction':float(change.double().mean()),'positive_fraction_old':float(oldmask.double().mean()),'positive_fraction_new':float(newmask.double().mean()),'activation_norm_old':float(old['q'].double().norm()),'activation_norm_new':float(new['q'].double().norm()),'preactivation_norm_old':float(aold.norm()),'preactivation_norm_new':float(anew.norm()),'fused_cosine':cosine(zo,zn),'preactivation_cosine':cosine(aold,anew),'projection_cosine':cosine(po,pn),'projection_diff_norm':float((pn-po).norm())}
 runs.append({'epoch':epoch,'checkpoint_sha256':sha(path),'results':results,'diagnostics':extra})
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'reference_trace_sha256':sha(TRACE),'runs':runs}
with (P/'counterfactual.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for r in runs:print(json.dumps(r,ensure_ascii=False),flush=True)
