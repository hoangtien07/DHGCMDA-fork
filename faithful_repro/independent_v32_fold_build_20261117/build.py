"""Invoke the previously pinned fold-feature builder for selection folds 2–4."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PKG=ROOT/'faithful_repro/colab_package_f0_selection_20261115'
mf=json.loads((PKG/'package_manifest.json').read_text(encoding='utf-8'))['files']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
required=['src/build_candidate_fold_features_rev3.py','src/semantic.py','src/functional.py','src/contracts.py',
          'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz','inputs/pair_split_candidate.csv',
          'inputs/candidate_manifest.json','inputs/HMDDv3.xlsx','inputs/meshToID.csv','inputs/mtrees2019.bin']
for name in required:assert sha(PKG/name)==mf[name],('source pin',name)
IN=PKG/'inputs'; SRC=PKG/'src'; out=HERE/'bundles';out.mkdir(exist_ok=True)
for fold in (2,3,4):
    target=out/f'fold{fold}_selection';assert not target.exists() and not Path(str(target)+'.part').exists()
    command=[sys.executable,str(SRC/'build_candidate_fold_features_rev3.py'),
      '--candidate',str(IN),'--strict',str(IN/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'),
      '--fold',str(fold),'--phase','selection','--mesh-xlsx',str(IN/'HMDDv3.xlsx'),
      '--meshtoid',str(IN/'meshToID.csv'),'--mtrees',str(IN/'mtrees2019.bin'),'--out',str(target)]
    mapping={'candidate-csv':'inputs/pair_split_candidate.csv','candidate-manifest':'inputs/candidate_manifest.json',
      'strict':'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz','semantic':'src/semantic.py',
      'functional':'src/functional.py','contracts':'src/contracts.py','mesh-xlsx':'inputs/HMDDv3.xlsx',
      'meshtoid':'inputs/meshToID.csv','mtrees':'inputs/mtrees2019.bin'}
    for flag,rel in mapping.items():command.extend(['--'+flag+'-sha',mf[rel]])
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    print('BUILD_START',fold,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
    proc=subprocess.run(command,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (HERE/f'fold{fold}_builder.stdout.log').write_text(proc.stdout,encoding='utf-8')
    (HERE/f'fold{fold}_builder.stderr.log').write_text(proc.stderr,encoding='utf-8')
    assert proc.returncode==0,(fold,proc.returncode,proc.stderr[-2000:])
    assert target.is_dir() and not Path(str(target)+'.part').exists()
    print('BUILD_COMPLETE',fold,'bundle_sha256',sha(target/'candidate_fold_data.npz'),flush=True)
for name in required:assert sha(PKG/name)==mf[name],('source drift',name)
