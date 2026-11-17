import argparse,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(42,2025),required=True);a=ap.parse_args()
for fold in (1,2,3,4):
 print('START fold',fold,'seed',a.seed,flush=True)
 subprocess.run([sys.executable,'-X','utf8',str(p/'train.py'),'--fold',str(fold),'--seed',str(a.seed)],check=True)
print('BATCH_COMPLETE seed',a.seed,flush=True)
