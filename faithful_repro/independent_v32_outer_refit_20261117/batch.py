import argparse,json,subprocess,sys,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('dynamic','static'),required=True);a=ap.parse_args();stop=P/'STOP_ON_FAILURE'
for fold in range(5):
 for seed in (1234,42,2025):
  if stop.exists():print('STOP_FLAG',a.arm,flush=True);sys.exit(1)
  key=f'fold{fold}_seed{seed}_{a.arm}';print('START',key,flush=True)
  r=subprocess.run([sys.executable,'-X','utf8',str(P/'train.py'),'--fold',str(fold),'--seed',str(seed),'--arm',a.arm])
  if r.returncode:
   try:stop.write_text(f'{key} exit {r.returncode}\n',encoding='utf-8')
   except OSError:pass
   print('FAILED',key,r.returncode,flush=True);sys.exit(r.returncode)
print('BATCH_DONE',a.arm,flush=True)
