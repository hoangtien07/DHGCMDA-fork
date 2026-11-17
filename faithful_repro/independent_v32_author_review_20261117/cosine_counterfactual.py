import copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model,u1_runner as u,losses
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();tsha=lambda t:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
cases=[(42,231,FR/'independent_v32_incidence_capture_20261117'),(2025,221,FR/'independent_v32_full_plan_seed2025_20261117'),(2025,481,FR/'independent_v32_full_plan_seed2025_20261117'),(1234,211,FR/'independent_v32_full_plan_seed1234_20261117')];out=[]
for seed,epoch,folder in cases:
 event=json.loads((folder/f'run/event_{epoch:04d}.json').read_text());path=folder/'run'/event['checkpoint'];assert sha(path)==event['sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];I=s['inputs'];assert c['seed']==seed
 model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=seed);model.load_state_dict(s['model_state']);model.train();graph=s['new_graph'];rng=s['torch_rng'];
 def once(mode):
  m=copy.deepcopy(model);m.train();m.zero_grad(set_to_none=True);torch.set_rng_state(rng.clone());pred=m(I,graph[0],graph[1],s['train_pairs']);typ=losses.type_loss(pred['logits'],s['train_y'].numpy(),s['weights']);intr=u.c3r._intra_terms(m,pred,c);inter=u._inter_terms_u1(pred,s['positive_mask'],s['negative_mask'],c);rec=u._recon_terms(m,pred,s['targets'],c)
  if mode=='cosine':
   sums=[]
   for mod,z in [('mi',pred['zm_pre']),('dis',pred['zd_pre'])]:
    proj=m.recon.heads[mod](z);unit=F.normalize(proj,p=2,dim=1,eps=1e-8);gram=unit@unit.T;valid=s['targets'][mod+'_valid'].bool();target=s['targets'][mod+'_target'];sums.append((gram[valid]-target[valid]).square().sum())
   rec['L_recon']=(sums[0]+sums[1])/(rec['recon_valid_mi']+rec['recon_valid_dis'])
  total=u.combine_objectives(typ,intr,inter,rec,s['unit_mass'],c);total.backward();grad=torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in m.parameters()]);assert torch.isfinite(total) and torch.isfinite(grad).all();v={'loss':float(total.detach()),'grad_norm':float(grad.double().norm()),'rng_end_sha256':tsha(torch.get_rng_state()),'recontribution':float((rec['lam']*(rec['recon_valid_mi']+rec['recon_valid_dis'])*rec['L_recon']/s['unit_mass']).detach())};return v,grad
 raw,gr=once('raw');cos,gc=once('cosine');repeat,rep=once('cosine');ref=json.loads((folder/f'run/paired_{epoch:04d}.json').read_text())['paired']['new'];assert abs(raw['loss']-ref['loss'])/max(1,abs(ref['loss']))<1e-5
 diff=float((gc.double()-rep.double()).norm())/max(1,float(gc.double().norm()));assert abs(cos['loss']-repeat['loss'])/max(1,abs(cos['loss']))<1e-5 and diff<1e-4 and raw['rng_end_sha256']==cos['rng_end_sha256']==repeat['rng_end_sha256']
 out.append({'seed':seed,'epoch':epoch,'checkpoint_sha256':sha(path),'raw':raw,'cosine':cos,'cosine_repeat_grad_relative_difference':diff})
 print(json.dumps(out[-1]),flush=True)
r={'status':'PASS','scope':'post-hoc frozen-state normalization only, not author-code replay or training','protocol_sha256':sha(P/'PROTOCOL_COSINE.md'),'script_sha256':sha(Path(__file__)),'cases':out}
with (P/'cosine_counterfactual.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
