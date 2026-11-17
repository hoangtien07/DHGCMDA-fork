import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;FR=P.parent;D=FR/'independent_v32_fold_build_20261117';REF=FR/'paper_guided/u1_horizon650_fold0_rev1';SRC=REF/'source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(42,2025),required=True);ap.add_argument('--fold',type=int,choices=(1,2,3,4),required=True);a=ap.parse_args();fold=a.fold;seed=a.seed
 bundle=FR/'paper_guided/candidate_fold1_selection_rev1' if fold==1 else D/'bundles'/f'fold{fold}_selection';datafile=bundle/'candidate_fold_data.npz';meta=bundle/'recon_meta.json';audit=bundle/'recon_audits.json'
 baselines=json.loads((D/'validation_baselines.json').read_text(encoding='utf-8'))['folds'];assert sha(datafile)==baselines[fold]['bundle_sha256'] and json.loads(meta.read_text(encoding='utf-8'))['npz_sha256']==sha(datafile)
 if fold>=2:
  pre=json.loads((D/'audit.json').read_text(encoding='utf-8'));q=next(x for x in pre['folds'] if x['fold']==fold);assert pre['status']==q['status']=='PASS' and q['bundle_sha256']==sha(datafile)
 ref=json.loads((REF/'out/manifest.json').read_text(encoding='utf-8'));cfg=dict(ref['cfg']);assert cfg['seed']==cfg['inter_seed']==1234 and cfg['max_epochs']==650 and cfg['eval_every']==10;cfg['seed']=cfg['inter_seed']=seed
 prior=FR/'paper_guided/u1_fold1_650_selection_rev1/out/manifest.json' if fold==1 else D/'u1_selection'/f'fold{fold}/out/manifest.json';old=json.loads(prior.read_text(encoding='utf-8'))
 for name,h in {**ref['source_hashes'],**ref['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
 sys.path[:0]=[str(SRC/'stages'),str(SRC)];import torch,torch_geometric,runner,u1_runner
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
 data=runner.load_data_npz(str(datafile));expected=u1_runner.expected_manifest_for_u1(cfg,data,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(datafile));assert expected['data_hashes']==old['data_hashes'] and expected['view_hashes']==old['view_hashes']
 out=P/'runs'/f'fold{fold}_seed{seed}';out.mkdir(parents=True,exist_ok=False);write(out/'expected.json',expected)
 launch={'fold':fold,'seed':seed,'bundle_sha256':sha(datafile),'meta_sha256':sha(meta),'audit_sha256':sha(audit),'protocol_sha256':sha(P/'PROTOCOL.md'),'runner_sha256':sha(Path(__file__)),'reference_manifest_sha256':sha(prior),'python':sys.version,'torch':torch.__version__,'pyg':torch_geometric.__version__,'numpy':np.__version__,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'inner-validation only; no outer-test'};write(out/'launch.json',launch)
 with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
  def log(x):
   print(x,file=f,flush=True)
   if x.startswith('epoch ') and (int(x.split()[1])%50==0):print(f'fold {fold} seed {seed}: {x}',flush=True)
  result=u1_runner.train_u1(cfg,data,str(out/'out'),expected_manifest=expected,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(datafile),log=log)
 assert sha(datafile)==launch['bundle_sha256'] and sha(meta)==launch['meta_sha256'] and sha(audit)==launch['audit_sha256']
 for name,h in {**ref['source_hashes'],**ref['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
 done={'fold':fold,'seed':seed,'aborted':result['aborted'],'final_epoch':result['final_epoch'],'manifest_sha256':sha(out/'out/manifest.json'),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'completion.json',done);print('DONE',json.dumps(done),flush=True);assert done['aborted'] is None and done['final_epoch']==650
if __name__=='__main__':main()
