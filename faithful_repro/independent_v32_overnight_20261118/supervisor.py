import ctypes,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
uv=['uv','run','--cache-dir',str(R/'.t3tmp/independent-objective-uv-cache'),'--python','C:/Users/HoangTien07/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe','--no-project','--with','torch==2.5.1','--with','torch-geometric==2.6.1','--with','numpy==1.26.4','--with','scipy==1.13.1','--with','scikit-learn==1.5.2','--with','pandas==2.2.3','python','-X','utf8']
jobs=[(a,s) for s in (1234,42,2025) for a in ('raw01','cos01')]+[('coslong',s) for s in (1234,42,2025)]
state={'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'supervisor_sha256':sha(__file__),'trainer_sha256':sha(P/'overnight_train.py'),'protocol_sha256':sha(P/'PROTOCOL_TRAIN.md'),'jobs':[{'arm':a,'seed':s,'status':'pending'} for a,s in jobs]}
def save():
 temp=P/'status.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');os.replace(temp,P/'status.json')
if (P/'status.json').exists():raise RuntimeError('status namespace occupied')
assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)!=0,'could not request system-awake state'
try:
 save()
 for j in state['jobs']:
  name=f"{j['arm']}_seed{j['seed']}";j['status']='running';j['start_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save();print('START',name,flush=True)
  try:
   assert sha(P/'overnight_train.py')==state['trainer_sha256'] and sha(P/'PROTOCOL_TRAIN.md')==state['protocol_sha256']
   with (P/(name+'.stdout.log')).open('x',encoding='utf-8') as stdout,(P/(name+'.stderr.log')).open('x',encoding='utf-8') as stderr:
    r=subprocess.run(uv+[str(P/'overnight_train.py'),'--arm',j['arm'],'--seed',str(j['seed'])],cwd=R,stdout=stdout,stderr=stderr,check=False)
   j['returncode']=r.returncode;j['status']='PASS' if r.returncode==0 else 'FAIL'
  except Exception:j['status']='FAIL';j['error']=traceback.format_exc()
  j['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save();print('END',name,j['status'],flush=True)
 state['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());state['status']='PASS' if all(j['status']=='PASS' for j in state['jobs']) else 'PARTIAL';save()
finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
