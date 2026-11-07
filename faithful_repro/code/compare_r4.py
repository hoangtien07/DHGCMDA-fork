import json,os,numpy as np
B=os.path.expanduser('~/dhg_runs/faithful')
def L(r,d,p,s):
    f=f'{B}/{r}/{d}_{p}_none_s{s}.json'
    return json.load(open(f)) if os.path.exists(f) else None
S=[1234,1,2]
def m(js,k): v=[np.mean([f[k] for f in j['folds']]) for j in js]; return np.mean(v),np.std(v),v
out=[]
P=lambda *a: out.append(' '.join(str(x) for x in a))
P('## r3 (round-3 faithful) vs r4 (same model, eval additions), seed-mean±sd over seeds',S)
for d in ['v2.0','v3.2_cuilab_deg2','v3.2_ref_MDAv3.2-3']:
  for p,ks in [('type',['top1_precision','top1_recall','top1_f1']),('triplet',['aupr','auc','f1'])]:
    a=[L('results',d,p,s) for s in S]; b=[L('results_r4',d,p,s) for s in S]
    if None in a or None in b: P(d,p,'MISSING'); continue
    for k in ks:
      ma=m(a,k); mb=m(b,k)
      ex=all(np.allclose([f[k] for f in x['folds']],[f[k] for f in y['folds']]) for x,y in zip(a,b))
      extra=''
      if p=='triplet': extra=f" | r4 draw0 {m(b,k+'_draw0')[0]:.4f}"
      P(f'{d} {p} {k}: r3 {ma[0]:.4f}±{ma[1]:.4f}  r4 {mb[0]:.4f}±{mb[1]:.4f}  identical_folds={ex}{extra}')
    if p=='type':
      for k in ['top1_recall_micro','top1_f1_micro']:
        mb=m(b,k); P(f'{d} {p} {k}: r4 {mb[0]:.4f}±{mb[1]:.4f}')
P('\n## 4-view vs 2-view cuilab (both r4), paired per seed')
for p,ks in [('type',['top1_precision','top1_recall','top1_f1','top1_recall_micro','top1_f1_micro']),('triplet',['aupr','auc','f1','aupr_draw0','auc_draw0','f1_draw0'])]:
  a=[L('results_r4','v3.2_cuilab_deg2',p,s) for s in S]; b=[L('results_r4','v3.2_cuilab_deg2_4view',p,s) for s in S]
  if None in a or None in b: P(p,'MISSING'); continue
  for k in ks:
    ma=m(a,k); mb=m(b,k); dif=np.array(mb[2])-np.array(ma[2])
    P(f'{p} {k}: 2view {ma[0]:.4f}±{ma[1]:.4f}  4view {mb[0]:.4f}±{mb[1]:.4f}  diff/seed {np.round(dif,4).tolist()} mean {dif.mean():+.4f}')
print('\n'.join(out))
