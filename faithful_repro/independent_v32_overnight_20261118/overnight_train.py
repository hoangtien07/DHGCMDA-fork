import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch,torch_geometric
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';D=FR/'paper_guided/candidate_fold0_selection_rev1';COS=FR/'independent_v32_cosine_training_20261117';sys.path[:0]=[str(SRC/'stages'),str(SRC),str(COS)]
import u1_runner as u,runner,train as cosine
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(1234,42,2025),required=True);ap.add_argument('--arm',choices=('raw01','cos01','coslong'),required=True);a=ap.parse_args();seed=a.seed;name=f'{a.arm}_seed{seed}'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
 cosine.test();old=read(COS/'launch.json');assert sha(COS/'train.py')==old['script_sha256'] and sha(COS/'PROTOCOL.md')==old['protocol_sha256']
 for path,h in old['pins'].items():assert sha(path)==h,path
 ref=read(FR/'paper_guided/u1_horizon650_fold0_rev1/out/manifest.json');cfg=dict(ref['cfg']);cfg['seed']=cfg['inter_seed']=seed;cfg['lam_recon']=1. if a.arm=='coslong' else .1;cfg['max_epochs']=1300 if a.arm=='coslong' else 650
 if a.arm!='raw01':u._recon_terms=cosine.cosine_terms
 out=P/'runs'/name;out.mkdir(parents=True,exist_ok=False)
 launch={'arm':a.arm,'seed':seed,'script_sha256':sha(__file__),'protocol_sha256':sha(P/'PROTOCOL_TRAIN.md'),'plan_sha256':sha(P/'PLAN.md'),'cosine_wrapper_sha256':sha(COS/'train.py'),'base_manifest_sha256':sha(FR/'paper_guided/u1_horizon650_fold0_rev1/out/manifest.json'),'input_pins':old['pins'],'cfg':cfg,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'launch.json',launch)
 data=runner.load_data_npz(str(D/'candidate_fold_data.npz'));expected=u.expected_manifest_for_u1(cfg,data,meta_path=str(D/'recon_meta.json'),audit_path=str(D/'recon_audits.json'),data_file_sha=sha(D/'candidate_fold_data.npz'));write(out/'expected.json',expected)
 try:
  with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
   def log(x):
    print(x,file=f,flush=True)
    if x.startswith('epoch ') and int(x.split()[1])%100==0:print(name,x,flush=True)
   res=u.train_u1(cfg,data,str(out/'out'),expected_manifest=expected,meta_path=str(D/'recon_meta.json'),audit_path=str(D/'recon_audits.json'),data_file_sha=sha(D/'candidate_fold_data.npz'),log=log)
  for path,h in old['pins'].items():assert sha(path)==h,path
  assert sha(__file__)==launch['script_sha256'] and sha(P/'PROTOCOL_TRAIN.md')==launch['protocol_sha256'] and sha(COS/'train.py')==launch['cosine_wrapper_sha256']
  done={'status':'PASS' if res['aborted'] is None and res['final_epoch']==cfg['max_epochs'] else 'FAIL','aborted':res['aborted'],'final_epoch':res['final_epoch'],'manifest_sha256':sha(out/'out/manifest.json'),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'completion.json',done);print('DONE',name,json.dumps(done),flush=True)
  if done['status']!='PASS':raise RuntimeError('training aborted')
 except BaseException:
  (out/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
