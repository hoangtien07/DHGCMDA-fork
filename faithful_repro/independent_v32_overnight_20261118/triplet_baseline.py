import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
P=Path(__file__).resolve().parent;F=P.parent;ref=F/'data/v3.2_ref_MDAv3.2-3.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts(trip,shape):
 nm,nd,nc=shape;mt=np.zeros((nm,nc),dtype=np.int64);dt=np.zeros((nd,nc),dtype=np.int64);gg=np.zeros(nc,dtype=np.int64);mp=np.zeros(nm,dtype=np.int64);dp=np.zeros(nd,dtype=np.int64);pc=np.zeros((nm,nd),dtype=np.int64)
 i,j,k=trip.T;np.add.at(mt,(i,k),1);np.add.at(dt,(j,k),1);np.add.at(gg,k,1);np.add.at(mp,i,1);np.add.at(dp,j,1);np.add.at(pc,(i,j),1);return mt,dt,gg,mp,dp,pc
def features(trip,c,exclude_self):
 i,j,k=trip.T;mt,dt,gg,mp,dp,pc=c;e=np.asarray(exclude_self,dtype=np.int64);assert len(e)==len(trip) and np.isin(e,[0,1]).all();a=np.column_stack([mt[i,k]-e,dt[j,k]-e,gg[k]-e,mp[i]-e,dp[j]-e]);assert np.all(a>=0) and np.all(pc[i,j]-e>=0)
 return np.column_stack([np.log1p(a),(pc[i,j]-e>0).astype(float),np.eye(len(gg))[k]])
def fixture():
 t=np.array([[0,0,0],[0,0,1],[1,1,1]]);c=counts(t,(2,2,2));x=features(t,c,np.ones(3,int));assert np.allclose(x[0,:5],np.log1p([0,0,0,1,1])) and x[0,5]==1 and x[2,5]==0
fixture()
with np.load(ref,allow_pickle=False) as z:y=z['Y'].astype(bool)
pos=np.argwhere(y);unknown=np.argwhere(~y);assert len(pos)==11748 and len(unknown)==545157;order=np.random.default_rng(0).permutation(len(pos));parts=np.array_split(order,5);rows=[]
for fold,idx in enumerate(parts):
 keep=np.ones(len(pos),bool);keep[idx]=False;trpos=pos[keep];tepos=pos[idx];rng=np.random.default_rng(1000+fold);chosen=rng.choice(len(unknown),size=len(tepos)+len(trpos),replace=False);teneg=unknown[chosen[:len(tepos)]];trneg=unknown[chosen[len(tepos):]]
 train=np.concatenate([trpos,trneg]);test=np.concatenate([tepos,teneg]);assert not set(map(tuple,train))&set(map(tuple,test)) and not y[tuple(trneg.T)].any() and not y[tuple(teneg.T)].any()
 c=counts(trpos,y.shape);x=features(train,c,np.r_[np.ones(len(trpos),int),np.zeros(len(trneg),int)]);xv=features(test,c,np.zeros(len(test),int));yt=np.r_[np.ones(len(trpos),int),np.zeros(len(trneg),int)];yv=np.r_[np.ones(len(tepos),int),np.zeros(len(teneg),int)]
 clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=0));clf.fit(x,yt);score=clf.predict_proba(xv)[:,1];assert np.isfinite(score).all();pred=score>=.5;prec,rec,f1,_=precision_recall_fscore_support(yv,pred,average='binary',zero_division=0);seen=c[-1][tepos[:,0],tepos[:,1]]>0
 row={'fold':fold,'train_positive':len(trpos),'train_unknown':len(trneg),'test_positive':len(tepos),'test_unknown':len(teneg),'positive_test_pair_seen_train':int(seen.sum()),'pair_seen_fraction':float(seen.mean()),'auc':float(roc_auc_score(yv,score)),'average_precision':float(average_precision_score(yv,score)),'precision_at_fixed_0_5':float(prec),'recall_at_fixed_0_5':float(rec),'f1_at_fixed_0_5':float(f1),'max_iterations_used':int(clf.named_steps['logisticregression'].n_iter_.max()),'train_index_sha256':hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),'test_index_sha256':hashlib.sha256(np.ascontiguousarray(test).tobytes()).hexdigest()};rows.append(row);print('fold',fold,'AUROC',round(row['auc'],4),'AP',round(row['average_precision'],4),'F1',round(row['f1_at_fixed_0_5'],4),flush=True)
out={'status':'PASS','scope':'exploratory baseline on simulated triplet splits; transductive unknown definition; not author pipeline or score','protocol_sha256':sha(P/'PROTOCOL_TRIPLET_BASELINE.md'),'script_sha256':sha(__file__),'ref_sha256':sha(ref),'fixture':'PASS','rows':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in ('auc','average_precision','precision_at_fixed_0_5','recall_at_fixed_0_5','f1_at_fixed_0_5','pair_seen_fraction')}}
with (P/'triplet_baseline.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print('mean',json.dumps(out['mean']),flush=True)
