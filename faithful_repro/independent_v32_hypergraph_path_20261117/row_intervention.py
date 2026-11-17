import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';CAP=FR/'independent_v32_refresh_capture_20261117/run';H=P/'hypergraph.json'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
ref=json.loads(H.read_text());runs=[]
for epoch in (226,591):
 item=next(r for r in ref['runs'] if r['epoch']==epoch);path=CAP/('spike_%04d.pt'%epoch);assert sha(path)==item['checkpoint_sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];I=s['inputs'];idx=item['selected_indices'];mask=torch.zeros(I['mi_v1'].shape[0],dtype=torch.bool);mask[idx]=True
 m=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));m.load_state_dict(s['model_state']);m.train();result={}
 with torch.no_grad():
  for mode in ('old','old_plus_rows','new_minus_rows','new'):
   old=s['old_graph'][0];new=s['new_graph'][0];G={k:v.clone() for k,v in (old if mode.startswith('old') else new).items()}
   if mode in ('old_plus_rows','new_minus_rows'):
    source=new if mode=='old_plus_rows' else old
    for key in ('mi_v1','mi_v2'):G[key][mask,:]=source[key][mask,:]
   z,_,_,gates=m.mi_view(I['mi_v1'],G['mi_v1'],I['mi_v2'],G['mi_v2']);p=m.recon.heads['mi'](z);gram=p@p.T;valid=s['targets']['mi_valid'].bool();err=(gram[valid]-s['targets']['mi_target'][valid]).double();result[mode]={'residual_sum':float(err.square().sum()),'projection_norm':float(p.double().norm()),'asymmetry_norm':float((G['mi_v1']-G['mi_v1'].T).double().norm()),'gates':gates.tolist()}
  for side in ('old','new'):assert abs(result[side]['residual_sum']-item[side]['residual_sum'])/item[side]['residual_sum']<1e-5
 runs.append({'epoch':epoch,'checkpoint_sha256':sha(path),'results':result})
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL_ROWS.md'),'script_sha256':sha(Path(__file__)),'hypergraph_sha256':sha(H),'runs':runs}
with (P/'row_intervention.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for r in runs:print(json.dumps(r),flush=True)
