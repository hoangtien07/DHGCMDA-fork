import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
P=Path(__file__).resolve().parent;F=P.parent;ref=F/'data/v3.2_ref_MDAv3.2-3.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();old=json.loads((P/'triplet_baseline.json').read_text());assert old['status']=='PASS' and sha(P/'triplet_baseline.py')==old['script_sha256'] and sha(P/'PROTOCOL_TRIPLET_BASELINE.md')==old['protocol_sha256']
with np.load(ref,allow_pickle=False) as z:y=z['Y'].astype(bool)
pos=np.argwhere(y);unk=np.argwhere(~y);order=np.random.default_rng(0).permutation(len(pos));splits=np.array_split(order,5);rows=[]
for f,idx in enumerate(splits):
 keep=np.ones(len(pos),bool);keep[idx]=False;p=pos[keep];v=pos[idx];rng=np.random.default_rng(1000+f);ids=rng.choice(len(unk),size=len(pos),replace=False);vn=unk[ids[:len(v)]];pn=unk[ids[len(v):]];train=np.concatenate([p,pn]);test=np.concatenate([v,vn]);yt=np.r_[np.ones(len(p),int),np.zeros(len(pn),int)];yv=np.r_[np.ones(len(v),int),np.zeros(len(vn),int)]
 nm,nd,nc=y.shape;mi=np.bincount(p[:,0]*nc+p[:,2],minlength=nm*nc).reshape(nm,nc);di=np.bincount(p[:,1]*nc+p[:,2],minlength=nd*nc).reshape(nd,nc);g=np.bincount(p[:,2],minlength=nc);m=mi.sum(1);d=di.sum(1);pair=np.bincount(p[:,0]*nd+p[:,1],minlength=nm*nd).reshape(nm,nd)
 def feat(a,ex):
  i,j,k=a.T;e=np.asarray(ex,dtype=np.int64);counts=np.array([mi[i,k],di[j,k],g[k],m[i],d[j]]).T-e[:,None];assert (counts>=0).all();return np.c_[np.log1p(counts),(pair[i,j]-e>0).astype(int),np.eye(nc)[k]]
 X=feat(train,yt);V=feat(test,np.zeros(len(test),int));model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=0)).fit(X,yt);s=model.predict_proba(V)[:,1]
 q=precision_recall_fscore_support(yv,s>=.5,average='binary',zero_division=0);r=old['rows'][f];actual={'auc':roc_auc_score(yv,s),'average_precision':average_precision_score(yv,s),'precision_at_fixed_0_5':q[0],'recall_at_fixed_0_5':q[1],'f1_at_fixed_0_5':q[2]}
 for k,z in actual.items():assert abs(z-r[k])<1e-10,(f,k,z,r[k])
 h=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();assert h(train)==r['train_index_sha256'] and h(test)==r['test_index_sha256'];seen=pair[test[:,0],test[:,1]]>0;groups={}
 for name,mask in [('seen_pair',seen),('unseen_pair',~seen)]:
  yy=yv[mask];ss=s[mask];pr=ss>=.5;assert len(np.unique(yy))==2;p_,r_,f_,_=precision_recall_fscore_support(yy,pr,average='binary',zero_division=0);groups[name]={'n':len(yy),'positive':int(yy.sum()),'unknown':int((1-yy).sum()),'auc':float(roc_auc_score(yy,ss)),'ap':float(average_precision_score(yy,ss)),'precision':float(p_),'recall':float(r_),'f1':float(f_)}
 rows.append({'fold':f,'groups':groups});print('fold',f,'seen',groups['seen_pair']['positive'],'/',groups['seen_pair']['n'],'unseen',groups['unseen_pair']['positive'],'/',groups['unseen_pair']['n'],flush=True)
out={'status':'PASS','scope':'independent recount and post-hoc observed-pair stratification; sampled unknowns, not verified negatives','protocol_sha256':sha(P/'PROTOCOL_TRIPLET_VERIFY.md'),'script_sha256':sha(__file__),'baseline_sha256':sha(P/'triplet_baseline.json'),'rows':rows}
with (P/'triplet_verify.json').open('x',encoding='utf-8') as h_:json.dump(out,h_,indent=2);h_.write('\n')
print('independent 5-fold metrics and split hash PASS')
