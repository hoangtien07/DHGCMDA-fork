import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch,torch_geometric
P=Path(__file__).resolve().parent;O=P.parent;F=O.parent;SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)];import runner,u1_runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True,choices=range(5));ap.add_argument('--seed',type=int,required=True,choices=(1234,42,2025));a=ap.parse_args();key=f'fold{a.fold}_seed{a.seed}'
torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
pf=read(P/'preflight.json');assert pf['status']=='PASS' and pf['protocol_sha256']==sha(P/'PROTOCOL.md') and pf['train_script_sha256']==sha(__file__) and pf['preflight_script_sha256']==sha(P/'preflight.py')
e=next(x for x in pf['entries'] if x['key']==key);C=P/'contracts'/key;B=Path(e['bundle']).parent
for name,path in [('bundle',B/'candidate_fold_data.npz'),('meta',B/'recon_meta.json'),('audit',B/'recon_audits.json'),('cfg',C/'cfg.json'),('expected',C/'expected.json'),('old_manifest',Path(e['old_manifest']))]:assert sha(path)==e[name+'_sha256'],name
cfg=read(C/'cfg.json');exp=read(C/'expected.json');old=read(e['old_manifest']);assert cfg['seed']==cfg['inter_seed']==a.seed and cfg['inter_cap_mult']==1 and cfg['max_epochs']==650 and exp['neg_bank']['sha256']==e['bank_sha256'];assert all(cfg[k]==old['cfg'][k] for k in cfg if k!='inter_cap_mult')
for name,h in {**old['source_hashes'],**old['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
data=runner.load_data_npz(str(B/'candidate_fold_data.npz'));out=P/'runs'/key;out.mkdir(parents=True,exist_ok=False);write(out/'launch.json',{'key':key,'preflight_sha256':sha(P/'preflight.json'),'protocol_sha256':sha(P/'PROTOCOL.md'),'trainer_sha256':sha(__file__),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
try:
 with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
  def log(x):
   print(x,file=f,flush=True)
   if x.startswith('epoch ') and int(x.split()[1])%100==0:print(key,x,flush=True)
  res=u1_runner.train_u1(cfg,data,str(out/'out'),expected_manifest=exp,meta_path=str(B/'recon_meta.json'),audit_path=str(B/'recon_audits.json'),data_file_sha=sha(B/'candidate_fold_data.npz'),log=log)
 for name,h in {**old['source_hashes'],**old['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
 for name,path in [('bundle',B/'candidate_fold_data.npz'),('meta',B/'recon_meta.json'),('audit',B/'recon_audits.json'),('cfg',C/'cfg.json'),('expected',C/'expected.json')]:assert sha(path)==e[name+'_sha256']
 d={'status':'PASS' if res['aborted'] is None and res['final_epoch']==650 else 'FAIL','key':key,'final_epoch':res['final_epoch'],'aborted':res['aborted'],'manifest_sha256':sha(out/'out/manifest.json'),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'completion.json',d);print('DONE',key,d['status'],flush=True)
 if d['status']!='PASS':raise RuntimeError('aborted training')
except BaseException:
 (out/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
