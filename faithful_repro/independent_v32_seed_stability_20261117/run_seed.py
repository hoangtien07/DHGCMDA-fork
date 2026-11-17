"""Run fixed archived U1 on candidate fold0 with a preregistered seed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
REF=FR/'paper_guided/u1_horizon650_fold0_rev1';SOURCE=REF/'source'
BUNDLE=FR/'paper_guided/candidate_fold0_selection_rev1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[42,2025],required=True);seed=ap.parse_args().seed
    datafile=BUNDLE/'candidate_fold_data.npz';meta=BUNDLE/'recon_meta.json';audit=BUNDLE/'recon_audits.json'
    assert sha(datafile)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
    assert json.loads(meta.read_text(encoding='utf-8'))['npz_sha256']==sha(datafile)
    old=json.loads((REF/'out/manifest.json').read_text(encoding='utf-8'))
    for name,h in {**old['source_hashes'],**old['imported_pure_hashes']}.items():assert sha(SOURCE/name)==h,name
    cfg=dict(old['cfg']);assert cfg['seed']==cfg['inter_seed']==1234 and cfg['max_epochs']==650 and cfg['eval_every']==10
    cfg['seed']=seed;cfg['inter_seed']=seed
    sys.path[:0]=[str(SOURCE/'stages'),str(SOURCE)]
    import torch,torch_geometric,runner,u1_runner
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4'
    assert not torch.cuda.is_available()
    loaded=runner.load_data_npz(str(datafile))
    expected=u1_runner.expected_manifest_for_u1(cfg,loaded,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(datafile))
    assert expected['data_hashes']==old['data_hashes'] and expected['view_hashes']==old['view_hashes']
    out=HERE/'runs'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    dump(out/'expected.json',expected)
    launch={'seed':seed,'inter_seed':seed,'bundle_sha256':sha(datafile),'meta_sha256':sha(meta),'audit_sha256':sha(audit),
       'protocol_sha256':sha(HERE/'PROTOCOL.md'),'runner_sha256':sha(Path(__file__)),
       'reference_manifest_sha256':sha(REF/'out/manifest.json'),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scope':'exploratory inner-validation; no outer-test predictions','python':sys.version,
       'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'numpy':np.__version__}
    dump(out/'launch.json',launch)
    with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
        def log(message):print(message,flush=True);print(message,file=f,flush=True)
        result=u1_runner.train_u1(cfg,loaded,str(out/'out'),expected_manifest=expected,meta_path=str(meta),
                                  audit_path=str(audit),data_file_sha=sha(datafile),log=log)
    assert sha(datafile)==launch['bundle_sha256'] and sha(meta)==launch['meta_sha256'] and sha(audit)==launch['audit_sha256']
    for name,h in {**old['source_hashes'],**old['imported_pure_hashes']}.items():assert sha(SOURCE/name)==h,name
    summary={'seed':seed,'aborted':result['aborted'],'final_epoch':result['final_epoch'],
        'best_epoch':result['best']['epoch'] if result['best'] else None,
        'manifest_sha256':sha(out/'out/manifest.json'),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/'completion.json',summary);print(json.dumps(summary),flush=True)
    assert summary['aborted'] is None and summary['final_epoch']==650,summary
if __name__=='__main__':main()
