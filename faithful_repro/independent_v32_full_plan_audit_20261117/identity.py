import hashlib,json,pathlib,numpy as np
p=pathlib.Path('v3.2_spld_paper');z=np.load('faithful_repro/data/v3.2_ref_MDAv3.2-3.npz',allow_pickle=False);checks={}
for name,key in [('miRNA_names.txt','mir_names'),('disease_names.txt','dis_names')]:checks[key]=np.array_equal(np.array((p/name).read_text(encoding='utf-8-sig').splitlines()),z[key])
for name,key in [('M_MISIM.npy','sim_m2'),('D_MESH.npy','sim_d2')]:checks[key]=np.array_equal(np.load(p/name,allow_pickle=False),z[key])
assert all(checks.values());out={'status':'PASS','checks':checks,'files':{name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('miRNA_names.txt','disease_names.txt','M_MISIM.npy','D_MESH.npy')}}
with pathlib.Path('faithful_repro/independent_v32_full_plan_audit_20261117/identity.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
