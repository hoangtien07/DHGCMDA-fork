import concurrent.futures,hashlib,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
PY=str(ROOT/'.venv-bank1/bin/python') if (ROOT/'.venv-bank1/bin/python').exists() else sys.executable
env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
pf=read(P/'preflight.json');assert pf['status']=='PASS' and pf['train_script_sha256']==sha(P/'train.py') and pf['protocol_sha256']==sha(P/'PROTOCOL.md');assert not (P/'status.json').exists();lock=threading.Lock()
state={'status':'RUNNING','supervisor':'linux','supervisor_sha256':sha(__file__),'python':PY,'preflight_sha256':sha(P/'preflight.json'),'report_sha256':sha(P/'report.py'),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'jobs':[{'key':e['key'],'fold':e['fold'],'seed':e['seed'],'status':'pending'} for e in pf['entries']]}
def save():
 tmp=P/'status.tmp';tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');os.replace(tmp,P/'status.json')
def run(j):
 key=j['key']
 with lock:j['status']='running';j['start_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save();print('START',key,flush=True)
 try:
  assert sha(P/'train.py')==pf['train_script_sha256'] and sha(P/'preflight.json')==state['preflight_sha256']
  with (P/(key+'.stdout.log')).open('x',encoding='utf-8') as out,(P/(key+'.stderr.log')).open('x',encoding='utf-8') as err:r=subprocess.run([PY,'-X','utf8',str(P/'train.py'),'--fold',str(j['fold']),'--seed',str(j['seed'])],cwd=ROOT,env=env,stdout=out,stderr=err)
  status='PASS' if r.returncode==0 else 'FAIL';code=r.returncode;error=None
 except Exception:status='FAIL';code=None;error=traceback.format_exc()
 with lock:
  j.update(status=status,returncode=code,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  if error:j['error']=error
  save();print('END',key,status,flush=True)
try:
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,state['jobs']))
 state['status']='TRAINED' if all(j['status']=='PASS' for j in state['jobs']) else 'PARTIAL';save();assert sha(P/'report.py')==state['report_sha256']
 with (P/'report.stdout.log').open('x',encoding='utf-8') as out,(P/'report.stderr.log').open('x',encoding='utf-8') as err:r=subprocess.run([PY,'-X','utf8',str(P/'report.py')],cwd=ROOT,env=env,stdout=out,stderr=err)
 state['report_returncode']=r.returncode;state['status']='PASS' if state['status']=='TRAINED' and r.returncode==0 and read(P/'results.json')['status']=='PASS' else 'PARTIAL';state['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save();print('COMPLETE',state['status'],flush=True)
finally:pass
