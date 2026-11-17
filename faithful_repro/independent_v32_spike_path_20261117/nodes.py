import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';CAP=FR/'independent_v32_refresh_capture_20261117/run'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(2);torch.set_num_interop_threads(1)
def cor(a,b):
 a=a.double();b=b.double();a=a-a.mean();b=b-b.mean();return float((a*b).sum()/(a.norm()*b.norm()))
def stat(x):
 x=x.double();q=torch.quantile(x,torch.tensor([0,.5,.9,.99,1.],dtype=torch.float64));return {'min':float(q[0]),'median':float(q[1]),'p90':float(q[2]),'p99':float(q[3]),'max':float(q[4])}
trace=json.loads((P/'trace.json').read_text(encoding='utf-8'));runs=[]
for epoch in (226,591):
 event=json.loads((CAP/('event_%04d.json'%epoch)).read_text());path=CAP/event['checkpoint'];assert sha(path)==event['sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];I=s['inputs']
 model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in I.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=True,recon_seed=int(c['seed']));model.load_state_dict(s['model_state']);model.train()
 result={}
 for label,graph in [('old',s['old_graph']),('new',s['new_graph'])]:
  G=graph[0]['mi_v1'];torch.set_rng_state(s['torch_rng'].clone())
  with torch.no_grad():
   z,z1,z2,gates=model.mi_view(I['mi_v1'],graph[0]['mi_v1'],I['mi_v2'],graph[0]['mi_v2']);p=model.recon.heads['mi'](z);gram=p@p.T
  norm=p.double().norm(dim=1); rows=G.double().sum(1); gn=G.double().norm(dim=1);mask=s['targets']['mi_valid'].bool();target=s['targets']['mi_target'].double();v=gram.double()[mask];t=target[mask];err=torch.zeros_like(gram,dtype=torch.float64);err[mask]=(v-t).square();r=err.sum(1);top=torch.argsort(r,descending=True)[:10]
  trace_run=next(q for q in trace['runs'] if q['epoch']==epoch)[label]
  assert abs(float(err.sum())-trace_run['residual_sum'])/max(1,trace_run['residual_sum'])<1e-5
  assert abs(float(norm.max())-trace_run['stages']['fc2']['rows']['max'])<1e-5
  result[label]={'graph_row_sum':stat(rows),'graph_row_l2':stat(gn),'projection_norm':stat(norm),'pearson_rowsum_projection_norm':cor(rows,norm),'pearson_rowl2_projection_norm':cor(gn,norm),'projection_norm_gt_2':int((norm>2).sum()),'reconstruction_decomposition':{'gram_sq':float((v*v).sum()),'minus_two_cross':float(-2*(v*t).sum()),'target_sq':float((t*t).sum()),'residual_sum':float(err.sum())},'top10_rows':[[int(i),float(r[i]),float(rows[i]),float(gn[i]),float(norm[i])] for i in top]}
 runs.append({'epoch':epoch,'checkpoint_sha256':sha(path),'old':result['old'],'new':result['new']})
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL_NODES.md'),'trace_sha256':sha(P/'trace.json'),'script_sha256':sha(Path(__file__)),'runs':runs}
with (P/'nodes.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for r in runs:
 print('epoch',r['epoch'])
 for k in ('old','new'):
  x=r[k];print(k,'rowsum',x['graph_row_sum'],'projection',x['projection_norm'],'corr',x['pearson_rowsum_projection_norm'],'n>2',x['projection_norm_gt_2'],'decomp',x['reconstruction_decomposition'],'top',x['top10_rows'][:3])
