import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;FR=P.parent;SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC),str(P)]
import torch,torch_geometric,u1_refit_noval_rev10 as refit
from contract import make_cfg,validate_refit_keys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=range(5),required=True);ap.add_argument('--seed',type=int,choices=(1234,42,2025),required=True);ap.add_argument('--arm',choices=('dynamic','static'),required=True);a=ap.parse_args();key=f'fold{a.fold}_seed{a.seed}_{a.arm}'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
 pf=json.loads((P/'preflight.json').read_text());assert pf['status']=='PASS' and pf['protocol_sha256']==sha(P/'PROTOCOL.md') and pf['contract_sha256']==sha(P/'contract.py') and pf['source_map_sha256']==sha(P/'source_map.json') and pf['bundle_audit_sha256']==sha(P/'bundle_audit.json');item=next(e for e in pf['entries'] if e['key']==key);C=P/'contracts'/key
 for f in ('cfg','binding','expected'):assert sha(C/(f+'.json'))==item[f+'_sha256']
 cfg=json.loads((C/'cfg.json').read_text());binding=json.loads((C/'binding.json').read_text());expected=json.loads((C/'expected.json').read_text());assert cfg['seed']==cfg['inter_seed']==a.seed and cfg['max_epochs']==650 and cfg['dynamic_enabled']==(a.arm=='dynamic') and binding['fixed_horizon_rule']=='fixed-final-650-not-best-validation'
 B=P/'bundles'/f'fold{a.fold}_refit';datafile=B/'candidate_fold_data.npz';meta=B/'recon_meta.json';audit=B/'recon_audits.json';assert sha(datafile)==item['bundle_sha256']==binding['bundle_npz_sha256'] and sha(meta)==binding['meta_sha256'] and sha(audit)==binding['audit_sha256'];data=refit.load_refit_npz(str(datafile),expect_sha=item['bundle_sha256']);validate_refit_keys(data);source=json.loads((P/'source_map.json').read_text());refit.verify_source_pins(source)
 out=P/'runs'/key;out.mkdir(parents=True,exist_ok=False);launch={'key':key,'fold':a.fold,'seed':a.seed,'arm':a.arm,'protocol_sha256':sha(P/'PROTOCOL.md'),'runner_sha256':sha(Path(__file__)),'preflight_sha256':sha(P/'preflight.json'),'bundle_sha256':sha(datafile),'meta_sha256':sha(meta),'audit_sha256':sha(audit),'cfg_sha256':sha(C/'cfg.json'),'binding_sha256':sha(C/'binding.json'),'expected_sha256':sha(C/'expected.json'),'source_map_sha256':sha(P/'source_map.json'),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'launch.json',launch)
 with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
  def log(s):
   print(s,file=f,flush=True)
   if s.startswith('epoch ') and int(s.split()[1])%100==0:print(key,s,flush=True)
  result=refit.train_refit_noval(cfg,data,str(out/'out'),log=log,expected_manifest=expected,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(datafile),expected_source_hashes=source,binding=binding)
 assert sha(datafile)==launch['bundle_sha256'] and sha(meta)==launch['meta_sha256'] and sha(audit)==launch['audit_sha256'];refit.verify_source_pins(source)
 assert result['aborted'] is None and result['completed_epochs']==650 and result['validation_consumed'] is False
 h=result['dynamic_history'];assert len(h)==(130 if a.arm=='dynamic' else 0),(key,len(h));ck=out/'out/ckpt_final.pt';checkpoint=refit.load_refit_ckpt(str(ck),sha(ck));assert checkpoint['epoch']==650 and checkpoint['cfg']==cfg
 if a.arm=='static':
  G,_=refit.s1r.build_view_graphs(data['views'],int(cfg['knn_k']))
  assert all(torch.equal(G[k],checkpoint['G'][k]) for k in G)
 done={'key':key,'status':'PASS','fold':a.fold,'seed':a.seed,'arm':a.arm,'completed_epochs':650,'validation_consumed':False,'dynamic_refreshes':len(h),'manifest_sha256':sha(out/'out/manifest_noval.json'),'checkpoint_sha256':sha(ck),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'completion.json',done);print('DONE',json.dumps(done),flush=True)
if __name__=='__main__':main()
