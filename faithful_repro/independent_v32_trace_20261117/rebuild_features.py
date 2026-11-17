"""Rebuild pinned U1 fold0 semantic and functional blocks, without model scoring."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];FR=ROOT/'faithful_repro'
SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source'
IN=FR/'colab_package_f0_selection_20261115/inputs'
data=FR/'paper_guided/candidate_fold0_selection_rev1/candidate_fold_data.npz'
meta=FR/'paper_guided/candidate_fold0_selection_rev1/candidate_fold_data.meta.json'
strict=FR/'data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
data1=FR/'paper_guided/candidate_fold1_selection_rev1/candidate_fold_data.npz'
src1=FR/'paper_guided/u1_fold1_650_selection_rev1/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin=json.loads(meta.read_text(encoding='utf-8'))['pins']
files={SRC/'semantic.py':pin['semantic.py'],SRC/'functional.py':pin['functional.py'],
       IN/'HMDDv3.xlsx':pin['HMDDv3.xlsx'],IN/'meshToID.csv':pin['meshToID.csv'],
       IN/'mtrees2019.bin':pin['mtrees2019.bin'],strict:pin[strict.name],
       data:'ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677',
       data1:'30b0ea3c216dd3a3f1a853d72b86888d08c414040bafc0c49ffeb7b98ab852ea',
       src1/'semantic.py':pin['semantic.py'],src1/'functional.py':pin['functional.py']}
for p,h in files.items():assert sha(p)==h,(p,sha(p),h)
sys.path.insert(0,str(SRC));import semantic,functional
z=np.load(strict,allow_pickle=True); f=np.load(data,allow_pickle=False)
print('PINNED INPUTS VERIFIED; rebuilding disease semantics',flush=True)
ms=semantic.load_mesh_source(str(IN/'HMDDv3.xlsx'),str(IN/'meshToID.csv'),str(IN/'mtrees2019.bin'))
sem=semantic.semantic_similarity([str(x) for x in z['dis_names']],ms,multi_descriptor='expand')
print('SEMANTICS READY; rebuilding functional similarity from train mask',flush=True)
folds=[]
for i,fp in enumerate((data,data1)):
    f=np.load(fp,allow_pickle=False)
    res=functional.functional_similarity(f['train_mask'].astype(bool),sem)
    sv=sem.S.astype(np.float32);fv=res.sim.astype(np.float32)
    assert np.array_equal(sv,f['view_dis_v2'][:,:271])
    assert np.array_equal(fv,f['view_mi_v2'][:,:411])
    folds.append({'fold':i,'train_mask_nonzero':int(f['train_mask'].sum()),
                  'functional_nonempty_mirnas':int(res.nonempty.sum()),
                  'semantic_block_elementwise_equal':True,'functional_block_elementwise_equal':True})
for p,h in files.items():assert sha(p)==h,(p,'changed during rebuild')
out={'status':'PASS','scope':'rebuild U1 selection fold0 and fold1 feature blocks from pinned sources, no model scoring',
 'all_source_pins_verified':True,'semantic_block_elementwise_equal':True,'functional_block_elementwise_equal':True,
 'folds':folds,
 'semantic_covered_diseases':int(sem.coverage.sum()),'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):h for p,h in files.items()},
 'limits':'Input source provenance verified; exact scientific equivalence to author DHGCMDA feature recipe not established.'}
(HERE/'feature_rebuild.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2),flush=True)
