import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';PREV=FR/'independent_v32_graph_counterfactual_20261117'
sys.path[:0]=[str(SRC/'stages'),str(SRC),str(PREV)]
import u1_runner as u,losses,c4_model
from paired import run_pair
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
for eventfile in sorted((P/'run').glob('event_*.json')):
 e=json.loads(eventfile.read_text(encoding='utf-8'));out=P/'run'/('paired_%04d.json'%e['epoch'])
 if out.exists():continue
 path=P/'run'/e['checkpoint'];assert sha(path)==e['sha256'];s=torch.load(path,map_location='cpu',weights_only=True);c=s['cfg'];inputs=s['inputs'];assert s['epoch']==e['epoch']
 model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in inputs.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=float(c['lam_recon'])!=0,recon_seed=int(c['seed']))
 model.load_state_dict(s['model_state'],strict=True);model.train();opt=u.s1r.build_optimizer(model,c);opt.load_state_dict(s['optimizer_state']);assert len(opt.state)>0
 torch.set_rng_state(s['torch_rng'])
 def objective(m,g):
  pred=m(inputs,g[0],g[1],s['train_pairs']);typ=losses.type_loss(pred['logits'],s['train_y'].numpy(),s['weights']);intr=u.c3r._intra_terms(m,pred,c);inter=u._inter_terms_u1(pred,s['positive_mask'],s['negative_mask'],c);rec=u._recon_terms(m,pred,s['targets'],c);total=u.combine_objectives(typ,intr,inter,rec,s['unit_mass'],c);fac=(rec['recon_valid_mi']+rec['recon_valid_dis'])/s['unit_mass']
  return total,{'type':float(typ.detach()),'inter':float(inter['L_inter'].detach()),'intra':float(intr['L_intra'].detach()),'recontribution':float(rec['lam']*fac*rec['L_recon'].detach()),'recon_sum_mi':float(rec['recon_sum_mi']),'recon_sum_dis':float(rec['recon_sum_dis'])}
 result=run_pair(model,s['old_graph'],s['new_graph'],objective)
 assert result['repeat_within_tolerance'] and result['final_rng_equal']
 assert abs(result['new']['loss']-e['loss'])/max(1.,abs(e['loss']))<=1e-5,(result['new']['loss'],e['loss'])
 record={'status':'PASS','epoch':e['epoch'],'spike':e['spike'],'checkpoint_sha256':sha(path),'event_sha256':sha(eventfile),'evaluator_sha256':sha(Path(__file__)),'optimizer_state_entries':len(opt.state),'paired':result}
 with out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'epoch':e['epoch'],'spike':e['spike'],'old_loss':result['old']['loss'],'new_loss':result['new']['loss'],'repeat_gradient_relative_difference':result['repeat_gradient_relative_difference'],'optimizer_state_entries':len(opt.state)}),flush=True)
