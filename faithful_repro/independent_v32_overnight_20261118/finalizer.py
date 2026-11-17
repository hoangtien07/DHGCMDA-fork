import hashlib,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;B=P/'bank1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();overview=P/'overview.py';verify=P/'verify_core.py';pins={'overview':sha(overview),'verify':sha(verify),'watcher':sha(__file__)}
(P/'finalizer_started.json').write_text(json.dumps({'status':'WAITING','pins':pins,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
start=time.monotonic();deadline=6*3600
while not (B/'watcher_done.json').exists() and time.monotonic()-start<deadline:time.sleep(30)
assert sha(overview)==pins['overview'] and sha(verify)==pins['verify'];r1=subprocess.run([sys.executable,'-X','utf8',str(verify)],cwd=P.parent.parent,capture_output=True,text=True);r2=subprocess.run([sys.executable,'-X','utf8',str(overview)],cwd=P.parent.parent,capture_output=True,text=True)
out={'status':'PASS' if (B/'watcher_done.json').exists() and json.loads((B/'watcher_done.json').read_text(encoding='utf-8'))['status']=='PASS' and r1.returncode==r2.returncode==0 else 'PARTIAL','bank_watcher_done':(B/'watcher_done.json').exists(),'core_returncode':r1.returncode,'overview_returncode':r2.returncode,'core_output':r1.stdout[-2000:],'core_error':r1.stderr[-2000:],'overview_output':r2.stdout[-2000:],'overview_error':r2.stderr[-2000:],'report_sha256':sha(P/'REPORT_OVERNIGHT.md') if (P/'REPORT_OVERNIGHT.md').exists() else None,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(P/'finalizer_done.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
