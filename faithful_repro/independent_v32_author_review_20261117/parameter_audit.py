import hashlib,json,sys
from pathlib import Path
import torch,numpy as np
P=Path('faithful_repro/independent_v32_author_review_20261117');F=P.parent;SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str((SRC/'stages').resolve()),str(SRC.resolve())]
import c4_model,u1_refit_noval_rev10 as refit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=F/'independent_v32_outer_refit_20261117/runs/fold0_seed1234_dynamic';done=json.loads((run/'completion.json').read_text());ck=refit.load_refit_ckpt(str(run/'out/ckpt_final.pt'),done['checkpoint_sha256']);c=ck['cfg'];data=refit.load_refit_npz(str(F/'independent_v32_outer_refit_20261117/bundles/fold0_refit/candidate_fold_data.npz'))
def build(dims,types):return c4_model.PaperGuidedC4(dims,n_types=types,hidden=c['hidden'],readout=c['readout'],proj=c['proj'],hgt_layers=c['hgt_layers'],hgt_heads=c['hgt_heads'],hgt_dropout=c['hgt_dropout'],readout_dropout=c['readout_dropout'],leaky_slope=c['leaky_slope'],temperature_init=c['temperature_init'],temperature_min=c['temperature_min'],temperature_max=c['temperature_max'],recon_enabled=True,recon_seed=c['seed'])
model=build({k:v.shape[1] for k,v in data['views'].items()},5);model.load_state_dict(ck['model_state'],strict=True)
def inventory(m):
 groups={}
 for n,p in m.named_parameters():
  group=n.split('.')[0];groups[group]=groups.get(group,0)+p.numel()
 return {'total_trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'total_all_parameters':sum(p.numel() for p in m.parameters()),'groups':groups}
v2=build({k:878 for k in data['views']},4)
out={'status':'PASS','scope':'checkpoint architecture counts and dimension-only v2 counterfactual; paper param config unspecified','checkpoint_sha256':sha(run/'out/ckpt_final.pt'),'cfg':{'hidden':c['hidden'],'readout':c['readout'],'proj':c['proj'],'hgt_layers':c['hgt_layers'],'hgt_heads':c['hgt_heads']},'v32_dims':{k:list(v.shape) for k,v in data['views'].items()},'v32_actual':inventory(model),'v2_dim_only_counterfactual':inventory(v2),'paper_reported_approx':2550000,'script_sha256':sha(Path(__file__))}
with (P/'parameter_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,ensure_ascii=False))
