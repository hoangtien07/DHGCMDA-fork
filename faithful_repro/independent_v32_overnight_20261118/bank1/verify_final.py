import hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
P=Path(__file__).resolve().parent;O=P.parent;F=O.parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pf=read(P/'preflight.json');rr=read(P/'results.json');status=read(P/'status.json');assert rr['protocol_sha256']==sha(P/'PROTOCOL.md') and rr['report_script_sha256']==sha(P/'report.py') and rr['preflight_sha256']==sha(P/'preflight.json');checks=[]
for row in rr['rows']:
 e=next(x for x in pf['entries'] if x['key']==row['key']);d=P/'runs'/row['key'];done=read(d/'completion.json');assert done['status']=='PASS' and done['manifest_sha256']==sha(d/'out/manifest.json')==row['manifest_sha256'];m=read(d/'out/manifest.json');old=read(e['old_manifest']);assert m['cfg']['inter_cap_mult']==1 and old['cfg']['inter_cap_mult']==10 and all(m['cfg'][k]==old['cfg'][k] for k in m['cfg'] if k!='inter_cap_mult')
 assert sha(d/'out/predictions_val.npz')==row['prediction_sha256'] and sha(Path(e['old_manifest']).parent/'predictions_val.npz')==e['old_prediction_sha256'] and sha(e['bundle'])==e['bundle_sha256']
 with np.load(d/'out/predictions_val.npz',allow_pickle=False) as z,np.load(Path(e['old_manifest']).parent/'predictions_val.npz',allow_pickle=False) as q,np.load(e['bundle'],allow_pickle=False) as b:
  assert np.array_equal(z['pairs'],q['pairs']) and np.array_equal(z['pairs'],b['val_pairs']) and np.array_equal(z['y'],q['y']) and np.array_equal(z['y'],b['y_val'])
  for arm,score in [('raw1',z['logits']),('raw10',q['logits'])]:
   y=z['y'];pred=score.argmax(1);one=np.eye(5,dtype=int)[pred];macro=f1_score(y,one,average='macro',zero_division=0);micro=f1_score(y,one,average='micro',zero_division=0);hit=float(y[np.arange(len(y)),pred].mean());r=row[arm]['best'];assert abs(macro-r['macro_f1'])<1e-12 and abs(micro-r['micro_f1'])<1e-12 and abs(hit-r['any_hit'])<1e-12
 checks.append(row['key'])
out={'status':'PASS' if rr['status']=='PASS' and len(checks)==15 and status['status'] in ('TRAINED','PASS') else 'PARTIAL','results_sha256':sha(P/'results.json'),'script_sha256':sha(__file__),'checked_keys':checks,'missing':rr['missing'],'scope':'independent sklearn metrics and paired data/hash check; not blind paper replication'}
tmp=P/'verification.tmp';tmp.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');os.replace(tmp,P/'verification.json');print('VERIFY',out['status'],len(checks))
if out['status']!='PASS':sys.exit(2)
