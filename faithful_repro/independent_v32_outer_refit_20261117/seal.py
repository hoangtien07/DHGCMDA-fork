import hashlib,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;F=P.parent;SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';sys.path[:0]=[str(SRC/'stages'),str(SRC),str(P)]
import u1_refit_noval_rev10 as runner
from gate import required_keys,check_keys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (P/'STOP_ON_FAILURE').exists():raise RuntimeError('batch failure, scoring cannot be sealed')
keys=required_keys();missing=[k for k in keys if not (P/'runs'/k/'completion.json').exists()]
if missing:raise RuntimeError('refusing to seal: %d incomplete jobs'%len(missing))
check_keys(keys);assert not (P/'evaluation').exists() and not (P/'seal.json').exists()
pf=json.loads((P/'preflight.json').read_text());pm=json.loads((P/'prior_manifest.json').read_text());pv=json.loads((P/'prior_validation_parity.json').read_text());source=json.loads((P/'source_map.json').read_text());assert pf['status']==pm['status']==pv['status']=='PASS' and pf['protocol_sha256']==pm['protocol_sha256']==pv['protocol_sha256']==sha(P/'PROTOCOL.md');assert pf['source_map_sha256']==sha(P/'source_map.json') and pf['contract_sha256']==sha(P/'contract.py') and pm['prior_sha256']==pv['prior_sha256']==sha(P/'prior.py');runner.verify_source_pins(source)
entries=[]
for k in keys:
 r=P/'runs'/k;l=json.loads((r/'launch.json').read_text());d=json.loads((r/'completion.json').read_text());expected=next(x for x in pf['entries'] if x['key']==k);c=P/'contracts'/k
 assert d['status']=='PASS' and d['completed_epochs']==650 and d['validation_consumed'] is False and l['protocol_sha256']==sha(P/'PROTOCOL.md') and l['runner_sha256']==sha(P/'train.py') and l['preflight_sha256']==sha(P/'preflight.json')
 assert d['manifest_sha256']==sha(r/'out/manifest_noval.json') and d['checkpoint_sha256']==sha(r/'out/ckpt_final.pt') and l['bundle_sha256']==expected['bundle_sha256']
 for name in ('cfg','binding','expected'):assert l[name+'_sha256']==expected[name+'_sha256']==sha(c/(name+'.json'))
 frozen=json.loads((r/'out/manifest_frozen.json').read_text());exp=json.loads((c/'expected.json').read_text());runner.check_expected_manifest_refit(frozen,exp);cfg=json.loads((c/'cfg.json').read_text());ck=runner.load_refit_ckpt(str(r/'out/ckpt_final.pt'),d['checkpoint_sha256']);assert ck['epoch']==650 and ck['cfg']==cfg
 assert d['dynamic_refreshes']==(130 if d['arm']=='dynamic' else 0)
 entries.append({'key':k,'manifest_sha256':d['manifest_sha256'],'checkpoint_sha256':d['checkpoint_sha256'],'completion_sha256':sha(r/'completion.json')})
for i in range(5):assert pm['folds'][i]['counts_sha256']==sha(P/'priors'/f'fold{i}_counts.npz')
pa=json.loads((P/'pair_audit.json').read_text());assert pa['status']=='PASS' and len(pa['pairs'])==15 and pa['protocol_sha256']==sha(P/'PAIR_AUDIT_PROTOCOL.md') and pa['script_sha256']==sha(P/'audit_pairs.py')
PKG=F/'colab_package_f0_selection_20261115';files=PKG/'inputs';pmf=json.loads((PKG/'package_manifest.json').read_text())['files']
for rel in ('inputs/pair_split_candidate.csv','inputs/candidate_manifest.json','inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'):assert sha(PKG/rel)==pmf[rel]
seal={'status':'PASS','scope':'30 complete refits frozen before outer scoring; historical exposure remains','protocol_sha256':sha(P/'PROTOCOL.md'),'preflight_sha256':sha(P/'preflight.json'),'bundle_audit_sha256':sha(P/'bundle_audit.json'),'source_map_sha256':sha(P/'source_map.json'),'prior_manifest_sha256':sha(P/'prior_manifest.json'),'prior_parity_sha256':sha(P/'prior_validation_parity.json'),'prior_sha256':sha(P/'prior.py'),'pair_audit_sha256':sha(P/'pair_audit.json'),'gate_sha256':sha(P/'gate.py'),'score_sha256':sha(P/'score.py'),'verify_eval_sha256':sha(P/'verify_eval.py'),'report_sha256':sha(P/'report.py'),'metrics_sha256':sha(F/'independent_v32_metric_baseline_20261117/analyze.py'),'candidate_csv_sha256':sha(files/'pair_split_candidate.csv'),'candidate_manifest_sha256':sha(files/'candidate_manifest.json'),'strict_sha256':sha(files/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'),'entries':entries,'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
with (P/'seal.json').open('x',encoding='utf-8') as f:json.dump(seal,f,indent=2);f.write('\n')
print('PASS 30 refits, source, checkpoints, priors; scoring seal written')
