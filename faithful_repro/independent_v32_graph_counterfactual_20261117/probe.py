"""Replay seed42 and probe pre/post-refresh graphs at the same state/RNG."""
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from paired import run_pair

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;FR=HERE.parents[0]
REF=FR/'paper_guided/u1_horizon650_fold0_rev1';SRC=REF/'source'
DATA=FR/'paper_guided/candidate_fold0_selection_rev1';BUNDLE=DATA/'candidate_fold_data.npz'
BASE=FR/'independent_v32_seed_stability_20261117/runs/seed42'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest_tensor(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def graph_hash(g,e):
    return {'G':{k:digest_tensor(v) for k,v in g.items()},'ei':{str(k):digest_tensor(v) for k,v in e.items()}}
def save_new(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def main():
    assert sha(BUNDLE)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
    bmeta=DATA/'recon_meta.json';baudit=DATA/'recon_audits.json';basefile=BASE/'out/manifest.json'
    base=json.loads(basefile.read_text(encoding='utf-8'));complete=json.loads((BASE/'completion.json').read_text(encoding='utf-8'))
    assert sha(basefile)==complete['manifest_sha256'] and base['cfg']['seed']==base['cfg']['inter_seed']==42 and base['final_epoch']==650
    for name,h in {**base['source_hashes'],**base['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
    cfg=dict(base['cfg']);cfg['max_epochs']=230;assert cfg['eval_every']==10
    sys.path[:0]=[str(SRC/'stages'),str(SRC)]
    import torch_geometric,runner,u1_runner,losses
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4'
    assert not torch.cuda.is_available()
    data=runner.load_data_npz(str(BUNDLE))
    exp=u1_runner.expected_manifest_for_u1(cfg,data,meta_path=str(bmeta),audit_path=str(baudit),data_file_sha=sha(BUNDLE))
    assert exp['data_hashes']==base['data_hashes'] and exp['view_hashes']==base['view_hashes']
    lines,first=inspect.getsourcelines(u1_runner.train_u1)
    def lineno(marker):
        found=[first+i for i,s in enumerate(lines) if s.strip()==marker]
        assert len(found)==1,(marker,found)
        return found[0]
    old_line=lineno('old_g_sha = {k: _sha_bytes(v.numpy().tobytes()) for k,v in G.items()}')
    forward_line=lineno("out = model(inputs, G, ei, tp_t)")
    out=HERE/'replay_retry2';out.mkdir(exist_ok=False)
    save_new(out/'expected.json',exp)
    launch={'source_manifest_sha256':sha(basefile),'bundle_sha256':sha(BUNDLE),'meta_sha256':sha(bmeta),'audit_sha256':sha(baudit),
            'protocol_sha256':sha(HERE/'PROTOCOL_RETRY2.md'),'probe_sha256':sha(Path(__file__)),'paired_sha256':sha(HERE/'paired.py'),
            'seed':42,'max_epochs':230,'scope':'single train-state probe; no outer-test',
            'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save_new(out/'launch.json',launch)
    saved={};code=u1_runner.train_u1.__code__
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        if event!='line':return trace
        L=frame.f_locals;epoch=L.get('epoch',0)
        if frame.f_lineno==old_line and epoch==225 and 'old' not in saved:
            saved['old']=({k:v.detach().clone() for k,v in L['G'].items()},
                          {k:v.detach().clone() for k,v in L['ei'].items()})
            saved['old_hash']=graph_hash(*saved['old'])
            print('CAPTURED_PRE_REFRESH',flush=True)
        if frame.f_lineno==forward_line and epoch==226 and 'paired' not in saved:
            assert 'old' in saved
            new=({k:v.detach().clone() for k,v in L['G'].items()},
                 {k:v.detach().clone() for k,v in L['ei'].items()})
            saved['new_hash']=graph_hash(*new)
            hist=next(x for x in base['dynamic_history'] if x['epoch']==225)
            saved['historical_graph_match']={'old_G':saved['old_hash']['G']==hist['input_G_sha'],
                                            'old_ei':saved['old_hash']['ei']==hist['input_ei_sha'],
                                            'new_G':all(saved['new_hash']['G'][k]==hist['G_'+k+'_sha256'] for k in new[0])}
            edge_matches=[]
            for key,name in [(('mir','sim','mir'),'edges_sim_mi_sha256'),(('dis','sim','dis'),'edges_sim_dis_sha256'),
                             (('mir','assoc','dis'),'edges_assoc_sha256'),(('dis','rev_assoc','mir'),'edges_rev_assoc_sha256')]:
                edge_matches.append(digest_tensor(new[1][key])==hist[name])
            saved['historical_graph_match']['new_ei']=all(edge_matches)
            model=L['model'];rng_before=torch.get_rng_state().clone();nstate=np.random.get_state();pstate=random.getstate()
            snapshot={'epoch':226,'model_state':{k:v.detach().clone() for k,v in model.state_dict().items()},
                      'old_graph':saved['old'],'new_graph':new,'torch_rng':rng_before,
                      'inputs':{k:v.detach().clone() for k,v in L['inputs'].items()},
                      'train_pairs':L['tp_t'].detach().clone(),'train_y':torch.as_tensor(L['y_tr']).clone(),
                      'weights':L['w_t'].detach().clone(),'positive_mask':L['pos_t'].detach().clone(),
                      'negative_mask':L['neg_t'].detach().clone(),
                      'targets':{k:v.detach().clone() for k,v in L['tensors'].items()},
                      'unit_mass':L['unit_mass'],'cfg':cfg}
            snapshot_path=out/'probe_state.pt';torch.save(snapshot,snapshot_path);saved['snapshot_sha256']=sha(snapshot_path)
            def objective(m,graph):
                pred=m(L['inputs'],graph[0],graph[1],L['tp_t'])
                typ=losses.type_loss(pred['logits'],L['y_tr'],L['w_t'])
                intr=u1_runner.c3r._intra_terms(m,pred,cfg)
                inter=u1_runner._inter_terms_u1(pred,L['pos_t'],L['neg_t'],cfg)
                rec=u1_runner._recon_terms(m,pred,L['tensors'],cfg)
                total=u1_runner.combine_objectives(typ,intr,inter,rec,L['unit_mass'],cfg)
                fac=(rec['recon_valid_mi']+rec['recon_valid_dis'])/L['unit_mass']
                detail={'type_loss':float(typ.detach()),'intra_mean':float(intr['L_intra'].detach()),
                        'inter_mean':float(inter['L_inter'].detach()),'recon_mean':float(rec['L_recon'].detach()),
                        'recontribution':float(rec['lam']*fac*rec['L_recon'].detach()),
                        'recon_sum_mi':float(rec['recon_sum_mi']),'recon_sum_dis':float(rec['recon_sum_dis'])}
                return total,detail
            saved['paired']=run_pair(model,saved['old'],new,objective)
            assert torch.equal(rng_before,torch.get_rng_state()) and random.getstate()==pstate
            ns=np.random.get_state();assert nstate[0]==ns[0] and np.array_equal(nstate[1],ns[1]) and nstate[2:]==ns[2:]
            saved['edge_counts']={str(k):{'old':int(saved['old'][1][k].shape[1]),'new':int(new[1][k].shape[1])} for k in new[1]}
            print('COUNTERFACTUAL_COMPUTED',json.dumps(saved['paired']),flush=True)
        return trace
    with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
        def log(msg):print(msg,flush=True);print(msg,file=f,flush=True)
        sys.settrace(trace)
        try:result=u1_runner.train_u1(cfg,data,str(out/'out'),expected_manifest=exp,meta_path=str(bmeta),audit_path=str(baudit),data_file_sha=sha(BUNDLE),log=log)
        finally:sys.settrace(None)
    assert 'paired' in saved and result['aborted'] is None and result['final_epoch']==230
    actual=json.loads((out/'out/manifest.json').read_text(encoding='utf-8'))
    differences=[]
    for a,b in zip(actual['trajectory'],base['trajectory'][:230]):
        assert a['epoch']==b['epoch']
        for key in ('train_loss','grad_norm','update_norm'):
            av,bv=float(a[key]),float(b[key]);scale=max(1.,abs(bv));differences.append(abs(av-bv)/scale)
    worst=max(differences)
    actual226=actual['trajectory'][225];pair=saved['paired']
    for key,observed,tol in [('loss',actual226['train_loss'],1e-5),('grad_norm',actual226['grad_norm'],1e-4)]:
        assert abs(pair['new'][key]-observed)/max(1.,abs(observed))<tol,(key,pair['new'][key],observed)
    assert pair['repeat_within_tolerance'] and sha(BUNDLE)==launch['bundle_sha256'] and sha(basefile)==launch['source_manifest_sha256']
    for name,h in {**base['source_hashes'],**base['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
    summary={'status':'PASS','scope':'one state, seed42, epoch226, same weights and starting RNG; no optimizer in paired probes',
             'old_graph':saved['old_hash'],'new_graph':saved['new_hash'],'edges':saved['edge_counts'],'paired':pair,
             'historical_graph_match':saved['historical_graph_match'],
             'snapshot_sha256':saved['snapshot_sha256'],
             'historical_replay_within_1e-5':bool(worst<1e-5),
             'historical_replay_max_scaled_abs_difference':worst,'historical_values_checked':len(differences),
             'actual_epoch226_train_loss':actual226['train_loss'],'actual_epoch226_grad_norm':actual226['grad_norm'],
             'replay_manifest_sha256':sha(out/'out/manifest.json'),
             'limits':'Local intervention changes G and edge indices together, conditional on one selected spike; does not establish population-level or individual component causality.'}
    save_new(HERE/'result_retry2.json',summary)
    print('PROBE_PASS',json.dumps(summary),flush=True)
if __name__=='__main__':main()
