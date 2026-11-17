"""Re-score existing validation exports and the declared train-only disease prior."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];FR=ROOT/'faithful_repro'
TYPES=['genetics','epigenetics','circulation','target','tissue']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def harm(p,r):return float(2*p*r/(p+r)) if p+r else 0.
def metrics(y,s):
    y=np.asarray(y);s=np.asarray(s)
    assert y.ndim==2 and y.shape==s.shape and y.shape[1]==5 and len(y)>0
    assert np.isin(y,[0,1]).all() and (y.sum(1)>0).all() and np.isfinite(s).all()
    pred=s.argmax(1);one=np.eye(5,dtype=int)[pred];y=y.astype(int)
    tp=(one*y).sum(0);fp=(one*(1-y)).sum(0);fn=((1-one)*y).sum(0)
    d=2*tp+fp+fn;f=np.divide(2*tp,d,out=np.zeros(5,dtype=float),where=d!=0)
    macro=float(f.mean());micro=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
    assert np.isclose(macro,f1_score(y,one,average='macro',zero_division=0),rtol=0,atol=1e-12)
    assert np.isclose(micro,f1_score(y,one,average='micro',zero_division=0),rtol=0,atol=1e-12)
    first=y.argmax(1);p=float(np.mean(first==pred));r=float(np.mean([(pred[first==k]==k).mean() for k in np.unique(first)]))
    return {'macro_f1':macro,'micro_f1':micro,'any_hit':float(y[np.arange(len(y)),pred].mean()),
            'first_label_hybrid_f1':harm(p,r),'ties':int(((s==s.max(1,keepdims=True)).sum(1)>1).sum()),
            'per_class':{TYPES[k]:{'tp':int(tp[k]),'fp':int(fp[k]),'fn':int(fn[k]),'f1':float(f[k]),
                'support':int(y[:,k].sum()),'predicted':int(one[:,k].sum())} for k in range(5)}}
def prior(train_pairs,train_y,val_pairs,n_dis):
    c=np.zeros((n_dis,5));np.add.at(c,train_pairs[:,1],train_y);g=train_y.sum(0).astype(float)
    s=c[val_pairs[:,1]].copy();missing=s.sum(1)==0;s[missing]=g;s+=(g/g.sum())*1e-12
    return s,int(missing.sum())

def main():
    # Known multihot fixture: predicted type 1 twice, one true extra label in row 1.
    y=np.array([[1,1,0,0,0],[0,1,0,0,0]]);s=np.array([[0,2,0,0,0],[0,2,0,0,0]])
    z=metrics(y,s);assert abs(z['macro_f1']-.2)<1e-12 and abs(z['micro_f1']-.8)<1e-12 and z['any_hit']==1
    assert abs(z['first_label_hybrid_f1']-.5)<1e-12
    Y=np.load(FR/'data/v3.2_ref_MDAv3.2-3.npz',allow_pickle=False)['Y']
    prior_audit=json.loads((FR/'independent_reproduction_audit_20261116/metrics_and_objective.json').read_text(encoding='utf-8'))
    out={'scope':'archived inner-validation only; checkpoint selected on same validation; no outer-test metrics',
         'types':TYPES,'fixture':'PASS','folds':[]};classrows=[]
    for i,name in enumerate(('u1_horizon650_fold0_rev1','u1_fold1_650_selection_rev1')):
        run=FR/'paper_guided'/name/'out';bundle=FR/'paper_guided'/f'candidate_fold{i}_selection_rev1/candidate_fold_data.npz'
        f=np.load(bundle,allow_pickle=False);exp=run/'predictions_val.npz';m=json.loads((run/'predictions_val.meta.json').read_text())
        assert sha(exp)==m['npz_sha256'];v=np.load(exp,allow_pickle=False)
        pairs,yval,s=v['pairs'],v['y'],v['logits'];assert np.array_equal(pairs,f['val_pairs']) and np.array_equal(yval,f['y_val'])
        assert np.array_equal(yval,Y[pairs[:,0],pairs[:,1]])
        assert not set(map(tuple,f['train_pairs'])) & set(map(tuple,pairs))
        ps,unseen=prior(f['train_pairs'],f['y_train'],pairs,Y.shape[1])
        model,base=metrics(yval,s),metrics(yval,ps)
        prev=prior_audit[name+'/val']['metrics'];old=prior_audit[name+'/disease_prior_val']['metrics']
        for k in ('macro_f1','micro_f1','any_hit'):
            assert np.isclose(model[k],prev[k],atol=1e-12,rtol=0) and np.isclose(base[k],old[k],atol=1e-12,rtol=0)
        p1=s.argmax(1);p2=ps.argmax(1);hit1=yval[np.arange(len(yval)),p1]>0;hit2=yval[np.arange(len(yval)),p2]>0
        disagreement={'both_hit':int((hit1&hit2).sum()),'model_only':int((hit1&~hit2).sum()),
                      'baseline_only':int((~hit1&hit2).sum()),'neither':int((~hit1&~hit2).sum())}
        row={'fold':i,'validation_pairs':len(pairs),'best_epoch':m['best_epoch'],'model':model,'disease_prior':base,
             'unseen_disease_pairs':unseen,'any_hit_overlap':disagreement,'inputs':{'bundle_sha256':sha(bundle),'predictions_sha256':sha(exp)}}
        out['folds'].append(row)
        for method,result in [('U1',model),('disease_prior',base)]:
            for cls,v in result['per_class'].items():classrows.append({'fold':i,'method':method,'type':cls,**v})
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    with (HERE/'per_class.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['fold','method','type','tp','fp','fn','f1','support','predicted']);w.writeheader();w.writerows(classrows)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
