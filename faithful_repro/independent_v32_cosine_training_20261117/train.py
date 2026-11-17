import hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,torch_geometric
P=Path(__file__).resolve().parent; FR=P.parent; SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source'; DATA=FR/'paper_guided/candidate_fold0_selection_rev1';sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import u1_runner as u,runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
original=u._recon_terms
def cosine_terms(model,out,tensors,cfg):
 r=original(model,out,tensors,cfg)
 if r is None:return None
 sums=[]
 for mod,z in [('mi',out['zm_pre']),('dis',out['zd_pre'])]:
  p=model.recon.heads[mod](z); n=F.normalize(p,p=2,dim=1,eps=1e-8);g=n@n.T;v=tensors[mod+'_valid'].bool();t=tensors[mod+'_target'].detach();sums.append((g[v]-t[v]).square().sum())
 a,b=sums;ca,cb=r['recon_valid_mi'],r['recon_valid_dis'];r.update(L_recon=(a+b)/(ca+cb),recon_sum_mi=float(a.detach()),recon_sum_dis=float(b.detach()),recon_mean_mi=float((a/ca).detach()),recon_mean_dis=float((b/cb).detach()))
 return r
def test():
 from reconstruction_adapter import ReconstructionAdapter
 class M:pass
 torch.manual_seed(791);m=M();m.recon=ReconstructionAdapter(4,3,123,True);out={'zm_pre':torch.randn(4,4,requires_grad=True),'zd_pre':torch.randn(3,4,requires_grad=True)};t={}
 for mod,n in [('mi',4),('dis',3)]:t[mod+'_target']=torch.randn(n,n);t[mod+'_valid']=~torch.eye(n,dtype=torch.bool)
 cfg={'lam_recon':1.};rng=torch.get_rng_state().clone();raw=original(m,out,t,cfg); c=cosine_terms(m,out,t,cfg);assert torch.equal(rng,torch.get_rng_state())
 for mod in ('mi','dis'):assert torch.equal(raw['gram_'+mod],c['gram_'+mod])
 direct=[]
 for mod in ('mi','dis'):
  z=out['zm_pre' if mod=='mi' else 'zd_pre'];p=m.recon.heads[mod](z);n=p/p.norm(dim=1,keepdim=True).clamp_min(1e-8);v=t[mod+'_valid'];direct.append(((n@n.T)[v]-t[mod+'_target'][v]).square().sum())
 expected=sum(direct)/(c['recon_valid_mi']+c['recon_valid_dis']);assert torch.allclose(c['L_recon'],expected,atol=1e-7,rtol=1e-6)
 params=list(m.recon.parameters());g1=torch.autograd.grad(c['L_recon'],params,retain_graph=True);g2=torch.autograd.grad(expected,params);assert all(torch.allclose(x,y,atol=1e-7,rtol=1e-5) for x,y in zip(g1,g2))
 assert abs(float(raw['L_recon']-c['L_recon']))>1e-4
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available();test();write(P/'unit_pass.json',{'status':'PASS','script_sha256':sha(__file__),'protocol_sha256':sha(P/'PROTOCOL.md'),'checks':['cosine independent loss and gradients','raw graph grams','RNG unchanged','intervention changes loss']})
 historical=FR/'independent_v32_seed_stability_20261117/runs/seed42/out/manifest.json'; base=json.loads(historical.read_text());pins={str(SRC/k):v for k,v in {**base['source_hashes'],**base['imported_pure_hashes']}.items()};pins.update({str(DATA/'candidate_fold_data.npz'):'ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677',str(DATA/'recon_meta.json'):'b4c969512755ac6a813d43cc0fc99b5b822b12365ef0bfa23fbab974eff1e84b',str(DATA/'recon_audits.json'):'7ed36536a5afbc7ba63465c8a1f8cd83bb7d91f2685da2e0a507248b977de33e'})
 for path,h in pins.items():assert sha(path)==h,path
 write(P/'launch.json',{'script_sha256':sha(__file__),'protocol_sha256':sha(P/'PROTOCOL.md'),'historical_manifest_sha256':sha(historical),'pins':pins,'seeds':[1234,42,2025],'epoch':650,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 data=runner.load_data_npz(str(DATA/'candidate_fold_data.npz'));u._recon_terms=cosine_terms
 for seed in (1234,42,2025):
  cfg=dict(base['cfg']);cfg['seed']=cfg['inter_seed']=seed;assert cfg['max_epochs']==650 and cfg['dynamic_enabled'];out=P/f'seed{seed}';out.mkdir(exist_ok=False)
  exp=u.expected_manifest_for_u1(cfg,data,meta_path=str(DATA/'recon_meta.json'),audit_path=str(DATA/'recon_audits.json'),data_file_sha=sha(DATA/'candidate_fold_data.npz'));write(out/'expected.json',exp)
  with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
   def log(s):
    print(s,file=f,flush=True)
    if s.startswith('epoch ') and int(s.split()[1])%100==0:print('seed',seed,s,flush=True)
   try:r=u.train_u1(cfg,data,str(out/'out'),expected_manifest=exp,meta_path=str(DATA/'recon_meta.json'),audit_path=str(DATA/'recon_audits.json'),data_file_sha=sha(DATA/'candidate_fold_data.npz'),log=log)
   except Exception:
    (out/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
  write(out/'completion.json',{'status':'PASS' if r['aborted'] is None and r['final_epoch']==650 else 'FAIL','aborted':r['aborted'],'epoch':r['final_epoch'],'manifest_sha256':sha(out/'out/manifest.json'),'best':r['best']});print('FINISHED',seed,r['final_epoch'],r['aborted'],flush=True)
  if r['aborted'] is not None:break
 for path,h in pins.items():assert sha(path)==h,path
