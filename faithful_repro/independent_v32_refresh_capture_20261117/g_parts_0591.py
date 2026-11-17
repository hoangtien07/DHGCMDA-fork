import hashlib,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; FR=HERE.parent
SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source'
sys.path[:0]=[str(SRC/'stages'),str(SRC),str(FR/'independent_v32_graph_counterfactual_20261117')]
import u1_runner as u, losses, c4_model
from paired import run_pair
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=json.loads((HERE/'run/paired_0591.json').read_text(encoding='utf-8'))
p=HERE/'run/spike_0591.pt';assert sha(p)==result['checkpoint_sha256']
torch.set_num_threads(2);torch.set_num_interop_threads(1)
s=torch.load(p,map_location='cpu',weights_only=True);c=s['cfg'];inputs=s['inputs']
model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in inputs.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=float(c['lam_recon'])!=0,recon_seed=int(c['seed']))
model.load_state_dict(s['model_state'],strict=True);model.train();torch.set_rng_state(s['torch_rng'])
def objective(m,graph):
 pred=m(inputs,graph[0],graph[1],s['train_pairs'])
 typ=losses.type_loss(pred['logits'],s['train_y'].numpy(),s['weights'])
 intr=u.c3r._intra_terms(m,pred,c);inter=u._inter_terms_u1(pred,s['positive_mask'],s['negative_mask'],c)
 rec=u._recon_terms(m,pred,s['targets'],c)
 total=u.combine_objectives(typ,intr,inter,rec,s['unit_mass'],c)
 fac=(rec['recon_valid_mi']+rec['recon_valid_dis'])/s['unit_mass']
 return total,{'type_loss':float(typ.detach()),'intra_mean':float(intr['L_intra'].detach()),'inter_mean':float(inter['L_inter'].detach()),'recontribution':float(rec['lam']*fac*rec['L_recon'].detach()),'recon_sum_mi':float(rec['recon_sum_mi']),'recon_sum_dis':float(rec['recon_sum_dis'])}
old=s['old_graph'];new=s['new_graph']; rows={}
for name,graph in [('mi_new',({k:(new[0][k] if k.startswith('mi_') else old[0][k]) for k in old[0]},old[1])),('dis_new',({k:(new[0][k] if k.startswith('dis_') else old[0][k]) for k in old[0]},old[1])),('both_new',(new[0],old[1])),('newG_newE',new)]:
 r=run_pair(model,old,graph,objective);assert r['final_rng_equal']
 assert abs(r['old']['loss']-result['paired']['old']['loss'])/max(1,abs(r['old']['loss']))<1e-5
 if name=='newG_newE':assert abs(r['new']['loss']-result['paired']['new']['loss'])/max(1,abs(r['new']['loss']))<1e-5
 rows[name]=r
out={'status':'PASS','scope':'post-hoc four combinations at one replay snapshot, mixed graphs may be artificial','snapshot_sha256':sha(p),'protocol_sha256':sha(HERE/'PROTOCOL_0591.md'),'script_sha256':sha(Path(__file__)),'rows':rows}
with (HERE/'g_parts_0591.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for k,v in rows.items():print(k, 'loss',v['new']['loss'],'grad',v['new']['grad_norm'],'parts',v['new']['details'],flush=True)

