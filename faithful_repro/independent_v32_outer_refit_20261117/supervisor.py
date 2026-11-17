import json,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from gate import required_keys
keys=required_keys();start=time.monotonic()
print('SUPERVISOR_START waiting for 30 refits; no outer-test before seal',flush=True)
while True:
 if (P/'STOP_ON_FAILURE').exists():print('STOP_ON_FAILURE detected; no scoring',flush=True);sys.exit(1)
 done=[k for k in keys if (P/'runs'/k/'completion.json').is_file()]
 if len(done)==len(keys):break
 if time.monotonic()-start>24*3600:
  (P/'SUPERVISOR_TIMEOUT').write_text('24h elapsed; no outer scoring\n',encoding='utf-8');print('TIMEOUT',len(done),flush=True);sys.exit(1)
 time.sleep(30)
for name in ('audit_pairs.py','seal.py','score.py','verify_eval.py','report.py'):
 print('START_PHASE',name,flush=True)
 p=subprocess.run([sys.executable,'-X','utf8',str(P/name)],text=True,capture_output=True,encoding='utf-8',errors='replace')
 (P/('phase_'+name+'.stdout.log')).write_text(p.stdout,encoding='utf-8');(P/('phase_'+name+'.stderr.log')).write_text(p.stderr,encoding='utf-8')
 if p.returncode:
  (P/'SUPERVISOR_FAILURE.json').write_text(json.dumps({'status':'FAIL','phase':name,'exit_code':p.returncode,'stderr_tail':p.stderr[-4000:]},indent=2)+'\n',encoding='utf-8');print('PHASE_FAILED',name,p.returncode,p.stderr[-1000:],flush=True);sys.exit(p.returncode)
 print('PHASE_PASS',name,p.stdout[-300:],flush=True)
plan=P.parent.parent/'docs/superpowers/plans/2026-11-17-outer-refit-evaluation.md';text=plan.read_text(encoding='utf-8-sig');assert text.count('- [ ]')==5;plan.write_text(text.replace('- [ ]','- [x]'),encoding='utf-8')
(P/'SUPERVISOR_COMPLETE.json').write_text(json.dumps({'status':'PASS','refits':30,'score':'evaluation/results.json','verification':'evaluation/verification.json','report':'REPORT.md'},indent=2)+'\n',encoding='utf-8')
print('SUPERVISOR_COMPLETE',flush=True)
