import ctypes,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();s=P/'status.json';ver=P/'verify_final.py';pin=sha(ver)
uv=['uv','run','--cache-dir',str(ROOT/'.t3tmp/independent-objective-uv-cache'),'--python','C:/Users/HoangTien07/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe','--no-project','--with','torch==2.5.1','--with','torch-geometric==2.6.1','--with','numpy==1.26.4','--with','scipy==1.13.1','--with','scikit-learn==1.5.2','--with','pandas==2.2.3','python','-X','utf8']
assert not (P/'watcher.json').exists();(P/'watcher.json').write_text(json.dumps({'status':'WAITING','watcher_sha256':sha(__file__),'verify_script_sha256':pin,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
while True:
 try:
  state=json.loads(s.read_text(encoding='utf-8'));done=state.get('status') in ('PASS','PARTIAL') and 'report_returncode' in state
  if done:break
 except (OSError,ValueError):pass
 time.sleep(20)
assert sha(ver)==pin
with (P/'verify.stdout.log').open('x',encoding='utf-8') as out,(P/'verify.stderr.log').open('x',encoding='utf-8') as err:r=subprocess.run(uv+[str(ver)],cwd=ROOT,stdout=out,stderr=err)
(P/'watcher_done.json').write_text(json.dumps({'status':'PASS' if r.returncode==0 else 'PARTIAL_OR_ERROR','returncode':r.returncode,'verification_sha256':sha(P/'verification.json') if (P/'verification.json').exists() else None,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
