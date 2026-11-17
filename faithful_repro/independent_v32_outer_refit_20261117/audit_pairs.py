import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;F=P.parent;SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import torch,u1_refit_noval_rev10 as refit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();items=[];incomplete=[]
for fold in range(5):
 for seed in (1234,42,2025):
  dirs={a:P/'runs'/f'fold{fold}_seed{seed}_{a}' for a in ('dynamic','static')}
  if any(not (d/'completion.json').exists() for d in dirs.values()):incomplete.append([fold,seed]);continue
  vals={a:json.loads((d/'out/manifest_noval.json').read_text()) for a,d in dirs.items()};dyn,stat=vals['dynamic'],vals['static'];x,y=dyn['frozen'],stat['frozen'];assert x['data_hashes']==y['data_hashes'] and x['view_hashes']==y['view_hashes'] and x['neg_bank']==y['neg_bank'] and x['recon']==y['recon'];c,d=x['cfg'],y['cfg'];assert {k for k in set(c)|set(d) if c.get(k)!=d.get(k)}=={'dynamic_enabled'}
  dc=json.loads((dirs['dynamic']/'completion.json').read_text());sc=json.loads((dirs['static']/'completion.json').read_text());ck=refit.load_refit_ckpt(str(dirs['static']/'out/ckpt_final.pt'),sc['checkpoint_sha256']);assert dyn['result']['dynamic_history'][0]['input_G_sha']==ck['G_sha'] and dyn['result']['dynamic_history'][0]['input_ei_sha']==ck['ei_sha']
  tx=json.loads((dirs['dynamic']/'out/trajectory_noval.json').read_text());ty=json.loads((dirs['static']/'out/trajectory_noval.json').read_text());diff={k:max(abs(a[k]-b[k])/max(1.,abs(a[k]),abs(b[k])) for a,b in zip(tx[:5],ty[:5])) for k in ('train_loss','grad_norm','update_norm')};assert max(diff.values())<=1e-4,(fold,seed,diff)
  items.append({'fold':fold,'seed':seed,'status':'PASS','dynamic_manifest_sha256':dc['manifest_sha256'],'static_manifest_sha256':sc['manifest_sha256'],'prefix_max_scaled_difference':diff})
print(json.dumps({'passed_pairs':len(items),'remaining_pairs':len(incomplete),'first_pairs':items[:2]}),flush=True)
if not incomplete:
 out={'status':'PASS','protocol_sha256':sha(P/'PAIR_AUDIT_PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'pairs':items}
 with (P/'pair_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
