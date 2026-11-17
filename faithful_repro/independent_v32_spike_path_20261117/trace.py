import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';CAP=FR/'independent_v32_refresh_capture_20261117/run'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import c4_model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
def stats(x):
 x=x.detach().double();return {'shape':list(x.shape),'norm':float(x.norm()),'rms':float(x.square().mean().sqrt()),'max_abs':float(x.abs().max()),'mean':float(x.mean())}
def rowstats(x):
 n=x.detach().double().norm(dim=1); q=torch.quantile(n,torch.tensor([.5,.9,.99,1.],dtype=torch.float64));return {'median':float(q[0]),'p90':float(q[1]),'p99':float(q[2]),'max':float(q[3]),'argmax':int(n.argmax())}
def main(epoch):
 ef=CAP/('event_%04d.json'%epoch);ev=json.loads(ef.read_text(encoding='utf-8'));path=CAP/ev['checkpoint'];assert sha(path)==ev['sha256'];s=torch.load(path,weights_only=True);c=s['cfg'];inputs=s['inputs'];old,new=s['old_graph'],s['new_graph']
 model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in inputs.items()},n_types=int(c['n_types']),hidden=int(c['hidden']),readout=int(c['readout']),proj=int(c['proj']),hgt_layers=int(c['hgt_layers']),hgt_heads=int(c['hgt_heads']),hgt_dropout=float(c['hgt_dropout']),readout_dropout=float(c['readout_dropout']),leaky_slope=float(c['leaky_slope']),temperature_init=float(c['temperature_init']),temperature_min=float(c['temperature_min']),temperature_max=float(c['temperature_max']),recon_enabled=float(c['lam_recon'])!=0,recon_seed=int(c['seed']))
 model.load_state_dict(s['model_state'],strict=True);model.train()
 def forward(graph):
  captured={};handles=[]
  def hook(name):
   def save(module,inp,out):captured[name]=out.detach().clone()
   return save
  for name,module in [('conv1',model.mi_view.conv1),('conv2',model.mi_view.conv2),('fc1',model.recon.heads['mi'].fc1),('fc2',model.recon.heads['mi'].fc2)]:handles.append(module.register_forward_hook(hook(name)))
  try:
   torch.set_rng_state(s['torch_rng'].clone())
   with torch.no_grad():
    out=model(inputs,graph[0],graph[1],s['train_pairs']);r=model.recon(out['zm_pre'],out['zd_pre'],s['targets']['mi_target'],s['targets']['mi_valid'],s['targets']['dis_target'],s['targets']['dis_valid'])
   endrng=sha_tensor(torch.get_rng_state());z=out['zm_pre'];g=r['gram_mi'];target=s['targets']['mi_target'];mask=s['targets']['mi_valid'].bool();err=torch.zeros_like(g);err[mask]=(g[mask]-target[mask]).square();perrow=err.sum(dim=1).double();total=perrow.sum();rank=torch.argsort(perrow,descending=True);diagonal=torch.diagonal(err).sum()
   stages={k:{'all':stats(v),'rows':rowstats(v)} for k,v in {**captured,'view1':out['views'][0],'view2':out['views'][1],'fused':z}.items()}
   return {'stages':stages,'gates':out['gates'][0].detach().tolist(),'beta':model.mi_view.avf.beta.detach().tolist(),'gram':stats(g),'target_valid':stats(target[mask]),'gram_valid':stats(g[mask]),'residual_sum':float(total),'valid_count':int(mask.sum()),'diag_residual_fraction':float(diagonal/total),'top1_row_fraction':float(perrow[rank[:1]].sum()/total),'top5_row_fraction':float(perrow[rank[:5]].sum()/total),'top10_row_fraction':float(perrow[rank[:10]].sum()/total),'top_rows':[[int(i),float(perrow[i]),float(g[i,i])] for i in rank[:10]],'rng_end_sha256':endrng}
  finally:
   for h in handles:h.remove()
 def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 a=forward(old);b=forward(new);rep=forward(new);ref=json.loads((CAP/('paired_%04d.json'%epoch)).read_text(encoding='utf-8'))['paired']
 for v,key in [(a,'old'),(b,'new')]:assert abs(v['residual_sum']-ref[key]['details']['recon_sum_mi'])/max(1,abs(v['residual_sum']))<1e-5
 assert b['rng_end_sha256']==rep['rng_end_sha256']==a['rng_end_sha256']
 assert abs(b['residual_sum']-rep['residual_sum'])/max(1,b['residual_sum'])<1e-5
 return {'epoch':epoch,'checkpoint_sha256':sha(path),'old':a,'new':b,'repeat_new_residual':rep['residual_sum']}
runs=[main(e) for e in (226,591)]
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'runs':runs}
with (P/'trace.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
for r in runs:
 print('epoch',r['epoch'])
 for k in ('old','new'):
  a=r[k];print(k,'gates',a['gates'],'beta',a['beta'],'fused',a['stages']['fused']['rows'],'decoder',a['stages']['fc2']['rows'],'gram',a['gram_valid'],'resid',a['residual_sum'],'top1/5/diag',a['top1_row_fraction'],a['top5_row_fraction'],a['diag_residual_fraction'],'rows',a['top_rows'][:3])
