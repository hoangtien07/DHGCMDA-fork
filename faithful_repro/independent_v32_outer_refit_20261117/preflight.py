import hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;D=F/'independent_v32_fold_build_20261117';OLD=F/'independent_v32_5fold3seed_20261117';REF=F/'paper_guided/u1_horizon650_fold0_rev1';SRC=REF/'source';sys.path[:0]=[str(SRC/'stages'),str(SRC),str(P)]
import torch,torch_geometric,u1_refit_noval_rev10 as runner
from contract import make_cfg,validate_refit_keys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
audit=json.loads((P/'bundle_audit.json').read_text(encoding='utf-8'));assert audit['status']=='PASS' and audit['protocol_sha256']==sha(P/'PROTOCOL.md');prev=json.loads((OLD/'results.json').read_text(encoding='utf-8'));assert prev['status']=='PASS';reference=json.loads((REF/'out/manifest.json').read_text(encoding='utf-8'))
for name,h in {**reference['source_hashes'],**reference['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
source=runner.effective_source_map();write(P/'source_map.json',source);contracts=P/'contracts';contracts.mkdir(exist_ok=False);entries=[]
for fold in range(5):
 rec=audit['folds'][fold];B=P/'bundles'/f'fold{fold}_refit';bundle=B/'candidate_fold_data.npz';meta=B/'recon_meta.json';aud=B/'recon_audits.json';assert sha(bundle)==rec['bundle_sha256'];data=runner.load_refit_npz(str(bundle),expect_sha=sha(bundle));validate_refit_keys(data)
 for seed in (1234,42,2025):
  if seed==1234:run=(F/'paper_guided'/('u1_horizon650_fold0_rev1' if fold==0 else 'u1_fold1_650_selection_rev1')) if fold<2 else D/'u1_selection'/f'fold{fold}'
  elif fold==0:run=F/'independent_v32_seed_stability_20261117/runs'/f'seed{seed}'
  else:run=OLD/'runs'/f'fold{fold}_seed{seed}'
  selection=run/'out/manifest.json';r=next(x for x in prev['rows'] if x['fold']==fold and x['seed']==seed);assert sha(selection)==r['manifest_sha256']
  for arm in ('dynamic','static'):
   cfg=make_cfg(reference['cfg'],seed,arm);binding={'selection_manifest_sha256':sha(selection),'selected_epoch':650,'bundle_npz_sha256':sha(bundle),'meta_sha256':sha(meta),'audit_sha256':sha(aud),'fixed_horizon_rule':'fixed-final-650-not-best-validation','protocol_sha256':sha(P/'PROTOCOL.md'),'fold':fold,'seed':seed,'arm':arm}
   expected=runner.expected_manifest_for_refit(cfg,data,binding,meta_path=str(meta),audit_path=str(aud),data_file_sha=sha(bundle));key=f'fold{fold}_seed{seed}_{arm}';dest=contracts/key;dest.mkdir();write(dest/'cfg.json',cfg);write(dest/'binding.json',binding);write(dest/'expected.json',expected)
   entries.append({'key':key,'fold':fold,'seed':seed,'arm':arm,'bundle_sha256':sha(bundle),'selection_manifest_sha256':sha(selection),'cfg_sha256':sha(dest/'cfg.json'),'binding_sha256':sha(dest/'binding.json'),'expected_sha256':sha(dest/'expected.json')})
 print('PREFLIGHT_FOLD',fold,'contracts',len(entries),flush=True)
assert len(entries)==30;write(P/'preflight.json',{'status':'PASS','protocol_sha256':sha(P/'PROTOCOL.md'),'preflight_sha256':sha(Path(__file__)),'contract_sha256':sha(P/'contract.py'),'source_map_sha256':sha(P/'source_map.json'),'bundle_audit_sha256':sha(P/'bundle_audit.json'),'entries':entries});print('ALL_PREFLIGHT_PASS',len(entries),flush=True)
