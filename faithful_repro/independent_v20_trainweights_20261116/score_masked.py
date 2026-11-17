"""Recompute released metric and multilabel metrics from captured test pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'independent_v20_release_20261116'/'pinned_release'))
from Calculate_Metrics import Metric_fun


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','full'],required=True)
    mode=ap.parse_args().mode;out=HERE/'runs'/mode
    complete=json.loads((out/'completion.json').read_text(encoding='utf-8'))
    assert complete['status']=='complete' and complete['n_folds']==5
    assert complete['epoch_per_fold']==(1 if mode=='smoke' else 650)
    log=(out/'console.log').read_text(encoding='utf-8')
    bad={phrase:len(re.findall(re.escape(phrase),log,flags=re.I)) for phrase in [
      'Forward pass error:', 'Error in dual view construction:', 'Falling back to single source',
      'Error in evaluation seed', 'NaN detected!', 'NaN detected before backward!',
      'Error updating heterogeneous graph:', 'Error creating heterogeneous graph:']}
    folds=[];all_test=[];train_sets=[]
    for i in range(1,6):
        p=out/f'fold_{i}.npz';assert sha(p)==complete['fold_sha256'][p.name]
        d=np.load(p);pos=d['test_pos'];truth=d['true_scalar_pos'];scores=d['score_pos'];Y=d['y_multihot']
        assert scores.shape==(len(pos),5) and Y.shape==(len(pos),4) and (Y.sum(1)>0).all()
        assert np.count_nonzero(d['feature_scalar_pos'])==0
        pred=scores[:,1:5].argmax(1);one=np.eye(4,dtype=int)[pred]
        scalar_truth=truth.astype(int)-1
        assert np.array_equal(Y[np.arange(len(pos)),scalar_truth],np.ones(len(pos)))
        released=Metric_fun().compute_top1_metrics(truth,scores)
        accuracy=float((pred==scalar_truth).mean())
        recall=float(np.mean([(pred[scalar_truth==k]==k).mean() for k in np.unique(scalar_truth)]))
        harmonic=2*accuracy*recall/(accuracy+recall) if accuracy+recall else 0.
        assert np.allclose([accuracy,recall,harmonic], [released['top1_precision'],released['top1_recall'],released['top1_f1']],rtol=0,atol=1e-12)
        train=set(map(tuple,d['train_pos']));test=set(map(tuple,pos))
        assert len(test)==len(pos) and not train & test
        all_test.append(test);train_sets.append(train)
        row={'fold':i,'n_test_positive':len(pos),'n_test_negative':len(d['test_neg']),
          'n_multilabel':int((Y.sum(1)>1).sum()),'n_test_scalar_types_in_feature':int((d['feature_scalar_pos']==truth).sum()),
          'n_ties':int(((scores[:,1:5]==scores[:,1:5].max(1,keepdims=True)).sum(1)>1).sum()),
          'released':{'precision':accuracy,'recall':recall,'f1':harmonic,'valid_samples':int(released['valid_samples'])},
          'multihot':{'macro_f1':float(f1_score(Y,one,average='macro',zero_division=0)),
                      'micro_f1':float(f1_score(Y,one,average='micro',zero_division=0)),
                      'any_hit':float(Y[np.arange(len(pos)),pred].mean()),
                      'per_class_f1':f1_score(Y,one,average=None,zero_division=0).tolist(),
                      'support':Y.sum(0).astype(int).tolist()},
          'score_unique_rows':int(len(np.unique(scores,axis=0))), 'npz_sha256':sha(p)}
        displayed=re.search(rf'Fold {i} completed - AUC: [0-9.]+, Top-1 F1: ([0-9.]+)',log)
        assert displayed and abs(float(displayed.group(1))-harmonic)<0.000051,(i,displayed,harmonic)
        folds.append(row)
    assert all(not all_test[i] & all_test[j] for i in range(5) for j in range(i+1,5))
    for i in range(5):
        assert train_sets[i]==set().union(*(all_test[j] for j in range(5) if j!=i))
    means={group:{metric:float(np.mean([f[group][metric] for f in folds])) for metric in metrics}
           for group,metrics in [('released',['precision','recall','f1']),('multihot',['macro_f1','micro_f1','any_hit'])]}
    for key in ('precision','recall','f1'):
        assert abs(means['released'][key]-complete['top1']['top1_'+key])<1e-12
    result={'status':'PASS','mode':mode,'epochs':complete['epoch_per_fold'],'folds':folds,
      'means_across_folds':means,'full_model_default_seed_1234_but_split_seed_0':True,
      'unique_test_positives_across_five_folds':len(set().union(*all_test)),
      'all_train_sets_are_other_four_test_sets':True,'flagged_log_phrases':bad,
      'different_score_rows_each_fold':all(f['score_unique_rows']>1 for f in folds),
      'limits':'Train-only association matrix, GIP and class weights; static similarity feature provenance remains unverified. Paper folds and metric conventions not recovered.'}
    (out/'metric_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','mode':mode,'means':means,'folds':len(folds),'flagged_log_phrases':bad},indent=2))


if __name__=='__main__':main()
