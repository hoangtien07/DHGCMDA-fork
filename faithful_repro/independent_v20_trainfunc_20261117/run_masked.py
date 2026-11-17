"""Replay upstream main_optimized; capture returned fold arrays with a read-only hook.

Run separately with --smoke (1 epoch) or --full (650 epochs). The imported
upstream source is byte-identical to git; each run has its own data/cache cwd.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from mask import train_matrix, binary_gip
from weights import train_weights
from functional import functional_similarity

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
RELEASE=HERE.parent/'independent_v20_release_20261116'/'pinned_release'
BASELINE=HERE.parent/'independent_v20_release_20261116'/'runs'/'full'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_release():
    manifest=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(RELEASE/item['path'])==item['sha256'],item['path']
    return manifest


def state_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


class Tee:
    def __init__(self,a,b):self.a=a;self.b=b
    def write(self,s):self.a.write(s);self.b.write(s);self.b.flush()
    def flush(self):self.a.flush();self.b.flush()
    def isatty(self):return False


def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().numpy().copy()
    return np.asarray(x).copy()


def labels(csv_path):
    tensor=np.zeros((495,383,4),dtype=np.uint8)
    scalar=np.zeros((495,383),dtype=np.uint8)
    with csv_path.open(encoding='utf-8-sig',newline='') as h:
        for row in csv.reader(h):
            mi,di,typ=map(int,row[:3]);mi-=1;di-=1
            assert 0<=mi<495 and 0<=di<383 and 1<=typ<=4
            tensor[mi,di,typ-1]=1;scalar[mi,di]=typ
    assert int(tensor.sum())==1679 and int((tensor.sum(2)>0).sum())==1498
    return tensor,scalar


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True)
    mode=parser.parse_args().mode
    manifest=verify_release()
    out=HERE/'runs'/mode;out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(RELEASE/'v2.0_495m383D',out/'v2.0_495m383D')
    for item in manifest['files']:
        if item['path'].startswith('v2.0_495m383D/'):
            assert sha(out/item['path'])==item['sha256']
    Y,scalar=labels(out/'v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import torch_geometric,scipy,sklearn,pandas
    versions={'torch':torch.__version__,'pyg':torch_geometric.__version__,'numpy':np.__version__,
              'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__,
              'python':sys.version,'device':'cpu','threads':torch.get_num_threads()}
    assert versions['torch'].split('+')[0]=='2.5.1' and versions['pyg']=='2.6.1'
    assert (np.__version__,scipy.__version__,sklearn.__version__,pandas.__version__)==('1.26.4','1.13.1','1.5.2','2.2.3')
    assert not torch.cuda.is_available()
    os.chdir(out);sys.path.insert(0,str(RELEASE));sys.argv=[sys.argv[0]]
    stream=(out/'console.log').open('x',encoding='utf-8',buffering=1)
    sys.stdout=Tee(sys.stdout,stream);sys.stderr=Tee(sys.stderr,stream)
    print('REPLAY START',mode,versions,flush=True)
    import main_experiments_hetero1 as author
    from param import parameter_parser
    args=parameter_parser()
    assert args.validation==5 and args.n_head==8 and args.epoch==650 and args.seed==1234
    if mode=='smoke':args.epoch=1
    cfg=vars(args).copy()
    contract={'mode':mode,'source_commit':manifest['source_commit'],'source_manifest_sha256':sha(HERE/'source_manifest.json'),
      'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PLAN.md'),
      'config':cfg,'versions':versions,'truth_channel_order':['circulation','epigenetics','target','genetics'],
      'author_loader_rule':'last CSV type per cell','hook':'capture outputs of original train_epoch; original evaluator untouched',
      'dataset_note':'association, GIP, functional similarity, and class weights from train only; disease static source provenance unverified',
      'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    orig=author.train_epoch_optimized; counter={'n':0}
    def capture(model,fold_data,opt,arguments):
        from prepareData import Gauss_D_optimized, Gauss_M_optimized
        index=counter['n']+1
        reference=np.load(BASELINE/f'fold_{index}.npz')
        train_pos,train_neg=map(cpu,fold_data[2]);test_pos,test_neg=map(cpu,fold_data[3])
        for name, arr in [('train_pos',train_pos),('train_neg',train_neg),('test_pos',test_pos),('test_neg',test_neg)]:
            assert np.array_equal(arr,reference[name]),(index,name)
        full=cpu(fold_data[4]); masked=train_matrix(full,train_pos,test_pos)
        altered=full.copy();altered[test_pos[:,0],test_pos[:,1]]=0
        assert np.array_equal(masked,train_matrix(altered,train_pos,test_pos))
        assert np.count_nonzero(masked[test_pos[:,0],test_pos[:,1]])==0
        dd=cpu(fold_data[0]);original_mm=cpu(fold_data[1])
        mm=functional_similarity(masked,dd)
        assert np.isfinite(mm).all() and np.allclose(mm,mm.T,atol=1e-6)
        assert np.array_equal(mm,functional_similarity(train_matrix(altered,train_pos,test_pos),dd))
        matrix_path=out/f'functional_fold_{index}.npy';np.save(matrix_path,mm)
        print('FUNCTION AUDIT',index,'sha256',sha(matrix_path),'zero_degree_mirnas',int((masked!=0).sum(1).__eq__(0).sum()),'mean_abs_diff_release',float(np.abs(mm-original_mm).mean()),flush=True)
        dg=Gauss_D_optimized(masked,masked.shape[1]);mg=Gauss_M_optimized(masked,masked.shape[0])
        fold_data=list(fold_data);fold_data[4]=torch.from_numpy(masked).float()
        fold_data[1]=torch.from_numpy(mm).float()
        fold_data[9]=torch.from_numpy(binary_gip(masked)).float()
        fold_data[10]=torch.from_numpy(np.where(dd==0,dg,(dd+dg)/2)).float()
        fold_data[11]=torch.from_numpy(np.where(mm==0,mg,(mm+mg)/2)).float()
        print('MASK AUDIT',index,'train edges',np.count_nonzero(masked),'test type exposures',np.count_nonzero(masked[test_pos[:,0],test_pos[:,1]]),flush=True)
        counts,weights=train_weights(masked,train_pos)
        assert np.array_equal(counts, np.bincount(masked[train_pos[:,0],train_pos[:,1]].astype(int),minlength=5)[1:])
        original_loss=author.SimplifiedMultiTypeAssociationLoss
        original_init=original_loss.__init__
        def fold_init(self,*a,**kw):
            original_init(self,*a,**kw)
            self.class_weights.copy_(torch.tensor(weights,dtype=self.class_weights.dtype,device=self.class_weights.device))
            assert np.allclose(self.class_weights.detach().cpu().numpy(),weights,atol=1e-7)
            print('TRAIN WEIGHT AUDIT',index,'counts',counts.tolist(),'weights',weights.tolist(),flush=True)
        original_loss.__init__=fold_init
        try:
            result=orig(model,tuple(fold_data),opt,arguments)
        finally:
            original_loss.__init__=original_init
        np0=np.random.get_state();t0=torch.get_rng_state().clone();py0=random.getstate()
        fold=counter['n'];counter['n']+=1
        train_pos,train_neg=map(cpu,fold_data[2]);test_pos,test_neg=map(cpu,fold_data[3])
        truth_pos,truth_neg,score_pos,score_neg=map(cpu,result)
        assert test_pos.shape==(len(truth_pos),2) and test_neg.shape==(len(truth_neg),2)
        assert score_pos.ndim==2 and score_pos.shape==(len(test_pos),5)
        assert score_neg.ndim==2 and score_neg.shape==(len(test_neg),5)
        assert np.array_equal(truth_pos,scalar[test_pos[:,0],test_pos[:,1]])
        assert np.array_equal(truth_neg,scalar[test_neg[:,0],test_neg[:,1]])
        feature_matrix=cpu(fold_data[4])
        feature_scalar_pos=feature_matrix[test_pos[:,0],test_pos[:,1]]
        assert np.count_nonzero(feature_scalar_pos)==0
        assert np.isfinite(score_pos).all() and np.isfinite(score_neg).all()
        Ypos=Y[test_pos[:,0],test_pos[:,1]]
        assert (Ypos.sum(1)>0).all()
        p=out/f'fold_{fold+1}.npz'
        assert not p.exists()
        np.savez_compressed(p,train_pos=train_pos,train_neg=train_neg,test_pos=test_pos,test_neg=test_neg,
            true_scalar_pos=truth_pos,true_scalar_neg=truth_neg,feature_scalar_pos=feature_scalar_pos,
            score_pos=score_pos,score_neg=score_neg,y_multihot=Ypos)
        assert state_equal(np0,np.random.get_state()) and torch.equal(t0,torch.get_rng_state()) and py0==random.getstate()
        print('AUDIT CAPTURE',fold+1,'pairs',len(test_pos),'sha256',sha(p),flush=True)
        return result
    author.train_epoch_optimized=capture
    try:
        result=author.main_optimized(args)
        assert result is not None and counter['n']==5
        binary,cvtype,top1=result[:3]
        final={'status':'complete','n_folds':counter['n'],'mode':mode,'epoch_per_fold':args.epoch,
          'top1':{k:float(v) for k,v in top1.items()},'binary':np.asarray(binary).tolist(),
          'cvtype':np.asarray(cvtype).tolist(),
          'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'fold_sha256':{f'fold_{i}.npz':sha(out/f'fold_{i}.npz') for i in range(1,6)}}
        verify_release()
        for item in manifest['files']:
            if item['path'].startswith('v2.0_495m383D/'):
                assert sha(out/item['path'])==item['sha256']
        (out/'completion.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
        print('REPLAY COMPLETE',json.dumps(final,ensure_ascii=False),flush=True)
    except BaseException:
        traceback.print_exc()
        print('REPLAY PARTIAL: captured folds',counter['n'],flush=True)
        raise
    finally:
        author.train_epoch_optimized=orig
        sys.stdout.flush();sys.stderr.flush()
        sys.stdout=sys.__stdout__;sys.stderr=sys.__stderr__;stream.close()


if __name__=='__main__':main()
