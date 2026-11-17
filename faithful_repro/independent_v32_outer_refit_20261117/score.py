import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;F=P.parent;SRC=F/'paper_guided/u1_horizon650_fold0_rev1/source';PKG=F/'colab_package_f0_selection_20261115';IN=PKG/'inputs';sys.path[:0]=[str(SRC/'stages'),str(SRC),str(P),str(F/'independent_v32_metric_baseline_20261117')]
import torch,c4_model,u1_refit_noval_rev10 as refit
from prior import predict_counts
from analyze import metrics
from gate import check_keys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if not (P/'seal.json').is_file():raise RuntimeError('no complete-run scoring seal')
 seal=json.loads((P/'seal.json').read_text(encoding='utf-8'));assert seal['status']=='PASS';check_keys([r['key'] for r in seal['entries']]);assert not (P/'STOP_ON_FAILURE').exists()
 for key,p in [('protocol_sha256',P/'PROTOCOL.md'),('preflight_sha256',P/'preflight.json'),('bundle_audit_sha256',P/'bundle_audit.json'),('source_map_sha256',P/'source_map.json'),('prior_manifest_sha256',P/'prior_manifest.json'),('prior_parity_sha256',P/'prior_validation_parity.json'),('prior_sha256',P/'prior.py'),('pair_audit_sha256',P/'pair_audit.json'),('gate_sha256',P/'gate.py'),('score_sha256',Path(__file__)),('metrics_sha256',F/'independent_v32_metric_baseline_20261117/analyze.py'),('candidate_csv_sha256',IN/'pair_split_candidate.csv'),('candidate_manifest_sha256',IN/'candidate_manifest.json'),('strict_sha256',IN/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')]:assert seal[key]==sha(p),key
 source=json.loads((P/'source_map.json').read_text());refit.verify_source_pins(source);prior_manifest=json.loads((P/'prior_manifest.json').read_text());audit=json.loads((P/'bundle_audit.json').read_text());
 for entry in seal['entries']:
  d=P/'runs'/entry['key'];assert sha(d/'out/ckpt_final.pt')==entry['checkpoint_sha256'] and sha(d/'out/manifest_noval.json')==entry['manifest_sha256'] and sha(d/'completion.json')==entry['completion_sha256']
 # Only after the complete seal and all prechecks do we open held-out labels.
 with (IN/'pair_split_candidate.csv').open(encoding='utf-8-sig',newline='') as h:split=list(csv.DictReader(h))
 with np.load(IN/'v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz',allow_pickle=True) as z:Y=z['Y']
 out=P/'evaluation';out.mkdir(exist_ok=False);savejson(out/'launch.json',{'seal_sha256':sha(P/'seal.json'),'score_sha256':sha(Path(__file__)),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'post-development exploratory outer CV, not paper match'})
 rows=[];baselines=[];seen=[]
 for fold in range(5):
  B=P/'bundles'/f'fold{fold}_refit';bundle=B/'candidate_fold_data.npz';assert sha(bundle)==audit['folds'][fold]['bundle_sha256'];data=refit.load_refit_npz(str(bundle),expect_sha=sha(bundle));keys=sorted({(int(r['mir_idx']),int(r['dis_idx'])) for r in split if r[f'seg_fold{fold}']=='test'});pairs=np.asarray(keys,dtype=np.int64);assert len(pairs)==1747 and not data['train_mask'][pairs[:,0],pairs[:,1]].any();seen+=keys;y=np.asarray(Y[pairs[:,0],pairs[:,1]],dtype=np.int8);assert (y.sum(1)>0).all()
  priorfile=P/'priors'/f'fold{fold}_counts.npz';assert sha(priorfile)==prior_manifest['folds'][fold]['counts_sha256'];z=np.load(priorfile,allow_pickle=False);counts={k:z[k] for k in z.files};scores=predict_counts(counts,pairs);b={name:metrics(y,v) for name,v in scores.items()};baselines.append({'fold':fold,'metrics':b,'train_counts_sha256':sha(priorfile)})
  inp={k:torch.tensor(v,dtype=torch.float32) for k,v in data['views'].items()}
  for seed in (1234,42,2025):
   for arm in ('dynamic','static'):
    key=f'fold{fold}_seed{seed}_{arm}';entry=next(t for t in seal['entries'] if t['key']==key);run=P/'runs'/key;ck=refit.load_refit_ckpt(str(run/'out/ckpt_final.pt'),entry['checkpoint_sha256']);cfg=ck['cfg'];assert cfg['seed']==cfg['inter_seed']==seed and cfg['max_epochs']==650 and cfg['dynamic_enabled']==(arm=='dynamic')
    model=c4_model.PaperGuidedC4({k:v.shape[1] for k,v in inp.items()},n_types=int(cfg['n_types']),hidden=int(cfg['hidden']),readout=int(cfg['readout']),proj=int(cfg['proj']),hgt_layers=int(cfg['hgt_layers']),hgt_heads=int(cfg['hgt_heads']),hgt_dropout=float(cfg['hgt_dropout']),readout_dropout=float(cfg['readout_dropout']),leaky_slope=float(cfg['leaky_slope']),temperature_init=float(cfg['temperature_init']),temperature_min=float(cfg['temperature_min']),temperature_max=float(cfg['temperature_max']),recon_enabled=float(cfg['lam_recon'])!=0,recon_seed=seed)
    opt=refit.s1r.build_optimizer(model,cfg);state=refit.restore_refit_state(model,opt,ck);assert state['epoch']==650;logits=refit.forward_refit(model,inp,state['G'],state['ei'],pairs);q=metrics(y,logits);pred=out/(key+'.npz');np.savez_compressed(pred,pairs=pairs,y=y,logits=logits);rows.append({'fold':fold,'seed':seed,'arm':arm,'metrics':q,'checkpoint_sha256':entry['checkpoint_sha256'],'prediction_sha256':sha(pred)});print('SCORED',key,'macro',q['macro_f1'],flush=True)
 assert len(seen)==8735 and len(set(seen))==8735 and len(rows)==30 and len(baselines)==5;refit.verify_source_pins(source)
 result={'status':'PASS','scope':'exploratory historical-exposure outer CV; not paper-equivalent','seal_sha256':sha(P/'seal.json'),'scorer_sha256':sha(Path(__file__)),'protocol_sha256':sha(P/'PROTOCOL.md'),'rows':rows,'baselines':baselines,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};savejson(out/'results.json',result);print('ALL_SCORED',len(rows),flush=True)
if __name__=='__main__':main()
