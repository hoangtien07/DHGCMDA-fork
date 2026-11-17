import hashlib,json,sys
from pathlib import Path
import numpy as np,torch,torch_geometric
P=Path(__file__).resolve().parent;O=P.parent;F=O.parent;D=F/'independent_v32_fold_build_20261117';A=F/'independent_v32_5fold3seed_20261117';SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)];import runner,u1_runner,c2_runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
base=read(A/'results.json');bankcheck=read(O/'bank_size.json');assert base['status']==bankcheck['status']=='PASS';entries=[];(P/'contracts').mkdir(exist_ok=False)
for fold in range(5):
 B=F/'paper_guided'/f'candidate_fold{fold}_selection_rev1' if fold<2 else D/'bundles'/f'fold{fold}_selection';bundle=B/'candidate_fold_data.npz';meta=B/'recon_meta.json';audit=B/'recon_audits.json';data=runner.load_data_npz(str(bundle))
 for seed in (1234,42,2025):
  row=next(r for r in base['rows'] if r['fold']==fold and r['seed']==seed);assert sha(bundle)==row['bundle_sha256']
  run=((F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}') if seed==1234 else (F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}' if fold==0 else A/'runs'/f'fold{fold}_seed{seed}')
  old=read(run/'out/manifest.json');assert sha(run/'out/manifest.json')==row['manifest_sha256'] and old['cfg']['seed']==old['cfg']['inter_seed']==seed and old['cfg']['inter_cap_mult']==10 and old['cfg']['max_epochs']==650
  for name,h in {**old['source_hashes'],**old['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
  cfg=dict(old['cfg']);cfg['inter_cap_mult']=1;expected=u1_runner.expected_manifest_for_u1(cfg,data,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(bundle));assert expected['data_hashes']==old['data_hashes'] and expected['view_hashes']==old['view_hashes']
  bc=next(x for x in bankcheck['rows'] if x['fold']==fold and x['seed']==seed);assert expected['neg_bank']['sha256']==bc['bank1_sha256']
  key=f'fold{fold}_seed{seed}';c=P/'contracts'/key;c.mkdir();write(c/'cfg.json',cfg);write(c/'expected.json',expected)
  entries.append({'key':key,'fold':fold,'seed':seed,'bundle':str(bundle.resolve()),'bundle_sha256':sha(bundle),'meta_sha256':sha(meta),'audit_sha256':sha(audit),'old_manifest':str((run/'out/manifest.json').resolve()),'old_manifest_sha256':sha(run/'out/manifest.json'),'old_prediction_sha256':row['prediction_sha256'],'cfg_sha256':sha(c/'cfg.json'),'expected_sha256':sha(c/'expected.json'),'bank_sha256':bc['bank1_sha256']})
out={'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'preflight_script_sha256':sha(__file__),'train_script_sha256':sha(P/'train.py'),'bank_size_sha256':sha(O/'bank_size.json'),'prior_results_sha256':sha(A/'results.json'),'entries':entries}
write(P/'preflight.json',out);print('PREFLIGHT PASS',len(entries))
