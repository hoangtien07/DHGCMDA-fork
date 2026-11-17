import hashlib,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent;F=O.parent;sys.path.insert(0,str(F/'independent_v32_metric_baseline_20261117'));from analyze import metrics
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pf=read(P/'preflight.json');assert pf['status']=='PASS' and sha(P/'PROTOCOL.md')==pf['protocol_sha256'] and sha(P/'train.py')==pf['train_script_sha256'];prior=read(F/'independent_v32_5fold3seed_20261117/results.json');assert sha(F/'independent_v32_5fold3seed_20261117/results.json')==pf['prior_results_sha256'];rows=[];missing=[]
for e in pf['entries']:
 key=e['key'];d=P/'runs'/key;complete=d/'completion.json'
 if not complete.is_file():missing.append({'key':key,'state':'not_completed'});continue
 try:
  done=read(complete);assert done['status']=='PASS' and done['final_epoch']==650 and done['manifest_sha256']==sha(d/'out/manifest.json')
  old=read(e['old_manifest']);new=read(d/'out/manifest.json');assert sha(e['old_manifest'])==e['old_manifest_sha256'] and sha(e['bundle'])==e['bundle_sha256'];assert new['cfg']['inter_cap_mult']==1 and old['cfg']['inter_cap_mult']==10 and all(new['cfg'][k]==old['cfg'][k] for k in new['cfg'] if k!='inter_cap_mult')
  assert new['data_hashes']==old['data_hashes'] and new['view_hashes']==old['view_hashes'] and new['neg_bank']['sha256']==e['bank_sha256'] and new['final_epoch']==old['final_epoch']==650
  pred=d/'out/predictions_val.npz';op=Path(e['old_manifest']).parent/'predictions_val.npz';meta=read(d/'out/predictions_val.meta.json');assert sha(pred)==meta['npz_sha256'] and sha(op)==e['old_prediction_sha256']
  with np.load(pred,allow_pickle=False) as z, np.load(op,allow_pickle=False) as oz:
   assert np.array_equal(z['pairs'],oz['pairs']) and np.array_equal(z['y'],oz['y']);n=metrics(z['y'],z['logits']);b=metrics(oz['y'],oz['logits'])
  ref=next(x for x in prior['rows'] if x['fold']==e['fold'] and x['seed']==e['seed'])
  for k in ('macro_f1','micro_f1','any_hit'):
   assert abs(b[k]-ref['best'][k])<1e-12 and abs(n[k]-new['best']['val_'+k])<1e-12
  def points(m):
   t=m['trajectory'];assert len(t)==650;return {'epoch_best':m['best']['epoch'],'final':{k:t[-1]['val_'+k] for k in ('macro_f1','micro_f1','any_hit')},'spikes':sum(x['train_loss']>2*t[i-1]['train_loss'] for i,x in enumerate(t) if i),'max_grad':max(x['grad_norm'] for x in t)}
  rows.append({'key':key,'fold':e['fold'],'seed':e['seed'],'raw10':{'best':b,**points(old)},'raw1':{'best':n,**points(new)},'prediction_sha256':sha(pred),'manifest_sha256':sha(d/'out/manifest.json')})
 except Exception as ex:missing.append({'key':key,'state':'FAILED_VERIFICATION','error':str(ex)})
summary={}
if rows:
 for arm in ('raw10','raw1'):
  summary[arm]={f'{phase}_{k}':statistics.mean(r[arm][phase][k] for r in rows) for phase in ('best','final') for k in ('macro_f1','micro_f1','any_hit')}
result={'status':'PASS' if len(rows)==15 and not missing else 'PARTIAL','scope':'exploratory same-fold inner-validation; prior exposure; historical raw10 baseline','protocol_sha256':sha(P/'PROTOCOL.md'),'preflight_sha256':sha(P/'preflight.json'),'report_script_sha256':sha(__file__),'rows':rows,'missing':missing,'summary':summary}
tmp=P/'results.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,P/'results.json')
lines=['# Giới hạn negative bank 10 so với 1: validation 5 fold × 3 seed','','Trạng thái: **'+result['status']+'**; đã xác minh %d/15 cặp.'%len(rows),'','Đây là thí nghiệm hậu nghiệm, inner-validation đã dùng phát triển; không phải 15 quan sát độc lập, test mù hoặc tái hiện chính xác paper. Cap1 thay cả số negatives và objective, không chứng minh riêng tác động false-negative.','', '| Fold | Seed | Macro best cap10 | Macro best cap1 | Macro epoch650 cap10 | Macro epoch650 cap1 | Spike cap10 / cap1 |','|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 a,b=r['raw10'],r['raw1'];lines.append('| %d | %d | %.4f | %.4f | %.4f | %.4f | %d / %d |'%(r['fold'],r['seed'],a['best']['macro_f1'],b['best']['macro_f1'],a['final']['macro_f1'],b['final']['macro_f1'],a['spikes'],b['spikes']))
if summary:
 lines+=['','Trung bình mô tả trên các ô đã xong (không phải CI):', '', '| Arm | Best macro | Best micro | Best any-hit | Final macro | Final micro | Final any-hit |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in ('raw10','raw1'):
  s=summary[arm];lines.append('| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |'%(arm,s['best_macro_f1'],s['best_micro_f1'],s['best_any_hit'],s['final_macro_f1'],s['final_micro_f1'],s['final_any_hit']))
lines+=['','Kết quả best được chọn trên cùng validation; final là mốc cố định 650. Per-class, max gradient, hash và những ô lỗi/chưa hoàn tất trong `results.json`. `preflight.json` khóa nguồn, bundle, seed và hash trước fit. Không dùng outer-test trong thí nghiệm này.','', 'Các ô chưa hoàn thành/kiểm tra lỗi: '+(json.dumps(missing,ensure_ascii=False) if missing else 'không có')]
tmp=P/'REPORT.tmp';tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(tmp,P/'REPORT.md');print('REPORT',result['status'],len(rows),'/',15,flush=True)
