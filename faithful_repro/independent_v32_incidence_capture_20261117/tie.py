import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import dynamic_graph as dg,c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
E=231;ev=json.loads((P/'run/event_0231.json').read_text());path=P/'run'/ev['checkpoint'];assert sha(path)==ev['sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];I=s['inputs'];norm=s['refresh_artifacts']['normalized_mi'];original=s['refresh_artifacts']['incidence_mi'];n=len(norm);k=int(c['knn_k']);H=torch.zeros_like(original);tied=[]
for center in range(n):
 others=[j for j in range(n) if j!=center];sc=norm[center].double();ordered=sorted(others,key=lambda j:(-float(sc[j]),-j));sel=ordered[:k];H[sel+[center],center]=1
 vals=sorted((float(sc[j]) for j in others),reverse=True);boundary=vals[k-1]==vals[k]
 if boundary:tied.append(center)
 else:assert torch.equal(H[:,center],original[:,center])
 assert float(H[:,center].sum())==k+1
assert len(tied)==29
newG=dg._hypergraph_laplacian(H).to(norm.dtype);baseline=s['new_graph'][0];m=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));m.load_state_dict(s['model_state']);m.train();res={}
with torch.no_grad():
 for name,G in [('ascending',baseline['mi_v1']),('descending',newG)]:
  z,_,_,_=m.mi_view(I['mi_v1'],G,I['mi_v2'],G);p=m.recon.heads['mi'](z);gram=p@p.T;valid=s['targets']['mi_valid'].bool();err=(gram[valid]-s['targets']['mi_target'][valid]).double();dv=(original if name=='ascending' else H).sum(1)
  res[name]={'residual_sum':float(err.square().sum()),'projection_norm':float(p.double().norm()),'max_degree':float(dv.max()),'degree_top13_share':float(torch.topk(dv,13).values.sum()/dv.sum())}
ref=json.loads((P/'run/paired_0231.json').read_text())['paired']['new']['details']['recon_sum_mi'];assert abs(res['ascending']['residual_sum']-ref)/ref<1e-5
out={'status':'PASS','checkpoint_sha256':sha(path),'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(P/'PROTOCOL_TIE.md'),'tied_centers':len(tied),'changed_incidence_columns':int((H!=original).any(dim=0).sum()),'changed_incidence_entries':int((H!=original).sum()),'G_relative_change':float((newG-baseline['mi_v1']).double().norm()/baseline['mi_v1'].double().norm()),'results':res}
with (P/'tie_0231.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(out),flush=True)
