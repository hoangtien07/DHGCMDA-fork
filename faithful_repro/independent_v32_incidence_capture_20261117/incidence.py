import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import dynamic_graph as dg,c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def tsha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
for ef in sorted((P/'run').glob('event_*.json')):
 e=json.loads(ef.read_text());dest=P/'run'/('incidence_%04d.json'%e['epoch'])
 if dest.exists():continue
 path=P/'run'/e['checkpoint'];assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];I=s['inputs'];out={}
 for modality in ('mi','dis'):
  gram=s['producer']['gram_'+modality];a=s['refresh_artifacts'];norm=a['normalized_'+modality];H=a['incidence_'+modality];G=s['new_graph'][0][modality+'_v1'];assert torch.equal(norm,dg._normalize_offdiag_minmax(gram))
  H2=dg._knn_incidence(norm,int(c['knn_k'])).to(H.dtype);assert torch.equal(H,H2);G2=dg._hypergraph_laplacian(H).to(G.dtype);assert torch.equal(G,G2)
  n=gram.shape[0];k=min(int(c['knn_k']),n-1);sc=norm.clone();sc.fill_diagonal_(float('-inf'));sorted_scores=torch.sort(sc,dim=1,descending=True).values;boundary=sorted_scores[:,k-1]-sorted_scores[:,k];dv=H.sum(1);top=sorted_scores[:,0];
  out[modality]={'n':n,'k':k,'producer_sha256':tsha(gram),'normalized_sha256':tsha(norm),'incidence_sha256':tsha(H),'G_sha256':tsha(G),'exact_k_boundary_ties':int((boundary==0).sum()),'near_k_boundary_1e-6':int((boundary.abs()<=1e-6).sum()),'k_margin_min':float(boundary.min()),'k_margin_median':float(boundary.median()),'degree_min':float(dv.min()),'degree_median':float(dv.median()),'degree_max':float(dv.max()),'top13_degree_sum_share':float(torch.topk(dv,13).values.sum()/dv.sum()),'unique_degrees':int(dv.unique().numel()),'H_column_sizes':dv.new_tensor(H.sum(0).unique()).tolist()}
 m=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));m.load_state_dict(s['model_state']);m.train();G=s['new_graph'][0]
 with torch.no_grad():z,_,_,_=m.mi_view(I['mi_v1'],G['mi_v1'],I['mi_v2'],G['mi_v2']);proj=m.recon.heads['mi'](z);chosen=torch.topk(proj.double().norm(dim=1),13).indices;dv=s['refresh_artifacts']['incidence_mi'].sum(1);cut=torch.topk(dv,13).values.min()
 out['mi']['selected13_degree_at_least_top13_cutoff']=int((dv[chosen]>=cut).sum());out['mi']['selected13_degree_min']=float(dv[chosen].min());out['mi']['selected13_degree_max']=float(dv[chosen].max())
 record={'status':'PASS','epoch':e['epoch'],'spike':e['spike'],'checkpoint_sha256':sha(path),'event_sha256':sha(ef),'script_sha256':sha(Path(__file__)),'modalities':out}
 with dest.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(record),flush=True)
