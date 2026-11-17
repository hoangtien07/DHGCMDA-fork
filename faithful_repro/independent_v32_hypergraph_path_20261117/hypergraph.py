import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';CAP=FR/'independent_v32_refresh_capture_20261117/run';ROW=FR/'independent_v32_decoder_gate_20261117/rows.json';TRACE=FR/'independent_v32_spike_path_20261117/trace.json'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
rows=json.loads(ROW.read_text(encoding='utf-8'));traces=json.loads(TRACE.read_text(encoding='utf-8'));runs=[]
def vals(x):
 x=x.double();return {'mean':float(x.mean()),'median':float(x.median()),'min':float(x.min()),'max':float(x.max())}
def norm_rows(x):return x.double().norm(dim=1)
for epoch in (226,591):
 event=json.loads((CAP/('event_%04d.json'%epoch)).read_text());path=CAP/event['checkpoint'];assert sha(path)==event['sha256'];s=torch.load(path,map_location='cpu',weights_only=True);c=s['cfg'];I=s['inputs'];sel=next(x for x in rows['runs'] if x['epoch']==epoch)['diagnostics']['selected_indices'];assert len(sel)==len(set(sel))==13
 m=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));m.load_state_dict(s['model_state'],strict=True);m.train()
 mask=torch.zeros(I['mi_v1'].shape[0],dtype=torch.bool);mask[sel]=True;out={};tracked=next(q for q in traces['runs'] if q['epoch']==epoch)
 with torch.no_grad():
  for side in ('old','new'):
   graph=s[side+'_graph'];G=graph[0]['mi_v1'];rs=G.double().sum(1);rank=set(torch.argsort(rs,descending=True)[:13].tolist());S=set(sel)
   z,_,_,_=m.mi_view(I['mi_v1'],graph[0]['mi_v1'],I['mi_v2'],graph[0]['mi_v2']);p=m.recon.heads['mi'](z);gram=p@p.T;valid=s['targets']['mi_valid'].bool();res=(gram[valid]-s['targets']['mi_target'][valid]).double().square().sum();assert abs(float(res)-tracked[side]['residual_sum'])/tracked[side]['residual_sum']<1e-5
   convs={}
   for idx in (1,2):
    conv=getattr(m.mi_view,'conv%d'%idx);x=I['mi_v%d'%idx];Gi=graph[0]['mi_v%d'%idx];h=x@conv.weight+conv.bias;own=Gi[:,mask]@h[mask];rest=Gi[:,~mask]@h[~mask];y=conv(x,Gi);delta=(own+rest-y).double().norm()/max(1.,y.double().norm());assert delta<1e-5
    assert abs(float(y.double().norm())-tracked[side]['stages']['conv%d'%idx]['all']['norm'])/max(1,float(y.double().norm()))<1e-5
    convs['view%d'%idx]={'relative_decomposition_error':float(delta),'input_selected_norm':vals(norm_rows(h)[mask]),'input_other_norm':vals(norm_rows(h)[~mask]),'selected_output_norm':vals(norm_rows(y)[mask]),'other_output_norm':vals(norm_rows(y)[~mask]),'selected_source_contribution_norm':vals(norm_rows(own)[mask]),'other_source_contribution_norm':vals(norm_rows(rest)[mask]),'selected_source_alignment':float((own[mask].double()*rest[mask].double()).sum()/(own[mask].double().norm()*rest[mask].double().norm()))}
   out[side]={'selected_rowsum':vals(rs[mask]),'other_rowsum':vals(rs[~mask]),'top13_rowsum_overlap':len(S&rank),'selected_to_selected_rowsum':vals(G.double()[mask][:,mask].sum(1)),'selected_to_other_rowsum':vals(G.double()[mask][:,~mask].sum(1)),'projection_selected_norm':vals(norm_rows(p)[mask]),'projection_other_norm':vals(norm_rows(p)[~mask]),'residual_sum':float(res),'convs':convs}
 runs.append({'epoch':epoch,'checkpoint_sha256':sha(path),'selected_indices':sel,'old':out['old'],'new':out['new']})
r={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'selected_reference_sha256':sha(ROW),'trace_sha256':sha(TRACE),'runs':runs}
with (P/'hypergraph.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
for r in runs:
 print('epoch',r['epoch'])
 for k in ('old','new'):
  x=r[k];print(k,'overlap',x['top13_rowsum_overlap'],'rowsum selected',x['selected_rowsum'],'other',x['other_rowsum'],'projection',x['projection_selected_norm'],flush=True)
  for v,y in x['convs'].items():print(v,'output',y['selected_output_norm']['mean'],'contrib selected/other',y['selected_source_contribution_norm']['mean'],y['other_source_contribution_norm']['mean'],'cos',y['selected_source_alignment'],flush=True)
