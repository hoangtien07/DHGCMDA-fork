"""G7-A2 — checkpoint-conditioned analysis of sigmoid-BCE checkpoints.
ZERO optimizer steps. Uses the ACTUAL A2 objective (BCE-with-logits on
last_logits) and the fail-closed restore_checkpoint entrypoint.
Label: CHECKPOINT_CONDITIONED_GRADIENT_DIAGNOSTIC."""
import json, os, sys, types, hashlib
import numpy as np, torch
sys.path.insert(0, '.')
import p5_runner as pr

h = lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
ctx = pr.bind_context('P5_BENCHMARK_UNKNOWN_PROVENANCE', seed=1234)
Y = ctx['Y']
LOCKED = dict(hidden=256, proj=64, heads=4, nlayer=2, dropout=0.3, K=13,
    t=0.5, lam1=1.0, lam2=0.3, lam3=1.0, margin=0.5, margin_w=0.1,
    exist_w=0.3, neg_w=0.5, beta=0.99999, lr=1e-4, wd=1e-5, neg_ratio=10,
    update_every=5, theta=0.5, val_frac=0.1, eval_every=10, patience=10,
    epochs=650, n_neg_draws=10, predictor='diag', recon_reduction='mean',
    type_objective='sigmoid_bce',
    early_stop=1, ablation='none', protocol='type', folds=5, only_fold=-1,
    cap_threads=0, threads=1, diag_leak=0, seed=1234)
a = types.SimpleNamespace(**LOCKED)

def build_inputs(D, fold):
    Yt = np.zeros_like(Y); tr = fold['train']
    Yt[tr[:,0],tr[:,1]] = Y[tr[:,0],tr[:,1]]
    A = (Yt.sum(2)>0).astype(np.float32)
    sims = {k:D[k].astype(np.float32) for k in ('sim_m1','sim_m2','sim_d1','sim_d2')}
    X = {'m1':np.hstack([sims['sim_m1'],A]),'m2':np.hstack([sims['sim_m2'],A]),
         'd1':np.hstack([sims['sim_d1'],A.T]),'d2':np.hstack([sims['sim_d2'],A.T])}
    Xt = {k:torch.tensor(v) for k,v in X.items()}
    At,Ytt = torch.tensor(A),torch.tensor(Yt)
    S_M,S_D = torch.tensor(sims['sim_m2']),torch.tensor(sims['sim_d2'])
    excl = np.zeros(Y.shape[:2],bool)
    excl[fold['test'][:,0],fold['test'][:,1]]=True; excl[fold['val'][:,0],fold['val'][:,1]]=True
    neg_pool = np.argwhere((A==0)&~excl)
    rng = np.random.default_rng(a.seed)
    pos = torch.tensor(np.argwhere(A>0))
    neg = torch.tensor(neg_pool[rng.choice(len(neg_pool),min(len(neg_pool),a.neg_ratio*len(pos)),False)])
    n_k = Yt.sum((0,1)); w=(1-a.beta)/(1-a.beta**np.maximum(n_k,1))
    w = torch.tensor((w*5/w.sum()),dtype=torch.float32)
    return dict(Xt=Xt,At=At,Yt=Ytt,S_M=S_M,S_D=S_D,pos=pos,neg=neg,w=w)

GROUPS = {'encoder':['inp.','clm.','cld.','avf_'],'graph_layers':['hgt.','hgcn'],
          'predictor':['pred.'],'recon_heads':['rec_']}
def group_of(n):
    for g,pref in GROUPS.items():
        if any(n.startswith(p) for p in pref): return g
    return 'other'

out = {}
for f in range(1,5):
    ckp = f'runs/g7a2_fold{f}_s1234/checkpoint.pt'
    sha = h(ckp)
    fold = ctx['folds'][f]
    inp = build_inputs(ctx['D'], fold)
    nm,nd,C = Y.shape
    # fail-closed restore: metadata-driven semantics + sigmoid patch
    model, ck = pr.restore_checkpoint(ckp, a, nm, nd, C)
    G,ei = ck['G'], ck['ei']
    torch.manual_seed(0)
    with torch.no_grad():
        ex,tp,*_ = model(inp['Xt'],G,ei)
    tp_np = tp.numpy()
    te = fold['test']; taus = np.array(ck['thresholds'])
    yhat = (tp_np[te[:,0],te[:,1]] >= taus).astype(int)
    yt = Y[te[:,0],te[:,1]]
    tp_= ((yhat==1)&(yt==1)).sum(0); fp_=((yhat==1)&(yt==0)).sum(0); fn_=((yhat==0)&(yt==1)).sum(0)
    r = json.load(open(f'runs/g7a2_fold{f}_s1234/result.json'))
    replay_exact = bool((tp_==np.array(r['per_class_tp'])).all() and (fp_==np.array(r['per_class_fp'])).all() and (fn_==np.array(r['per_class_fn'])).all())
    # metadata consistency vs saved manifest
    man = json.load(open(f'runs/g7a2_fold{f}_s1234/resolved_manifest.json'))
    meta_ok = (ck['type_objective'] == man['resolved_args']['type_objective'] == 'sigmoid_bce'
               and ck['amendment'] == man['amendment'] == 'G7_A2_TYPE_SIGMOID_BCE'
               and man['resolved_args']['recon_reduction'] == 'mean')
    # gradient attribution with the ACTUAL A2 sigmoid-BCE objective
    ex,tp_,lintra,zm,zd,Sm,Sd = model(inp['Xt'],G,ei)
    p_s,n_s = ex[inp['pos'][:,0],inp['pos'][:,1]], ex[inp['neg'][:,0],inp['neg'][:,1]]
    l_ex = ((-(1-p_s)**2*torch.log(p_s+1e-7)).sum()+a.neg_w*(-(n_s**2)*torch.log(1-n_s+1e-7)).sum())/(len(p_s)+len(n_s))
    yp = inp['Yt'][inp['pos'][:,0],inp['pos'][:,1]]
    lg = model.pred.last_logits[inp['pos'][:,0],inp['pos'][:,1]]
    l_type = torch.nn.functional.binary_cross_entropy_with_logits(
        lg, yp, pos_weight=inp['w'], reduction='none').mean()
    l_inter = pr.fz.inter_loss(zm,zd,inp['At'],a.t,a.margin,a.margin_w)
    l_rec = ((inp['S_M']-Sm)**2).mean()+((inp['S_D']-Sd)**2).mean()
    comps = {'type':(l_type,1.0),'recon':(l_rec,a.lam3),'intra':(lintra,a.lam1),
             'inter':(l_inter,a.lam2),'exist':(l_ex,a.exist_w)}
    gn = {}; gvecs = {}
    for name,(l,wgt) in comps.items():
        model.zero_grad(set_to_none=True); (wgt*l).backward(retain_graph=True)
        tot = 0.0; by = {g:0.0 for g in GROUPS}; vec={}
        for n_,p_ in model.named_parameters():
            if p_.grad is None: continue
            n2 = float(p_.grad.norm()**2); tot += n2
            by[group_of(n_)] = by.get(group_of(n_),0)+n2; vec[n_]=p_.grad.detach().clone()
        gn[name]={'total':tot**0.5,'by_group':{k:v**0.5 for k,v in by.items()}}
        gvecs[name]=vec
    shared = set(gvecs['type'])&set(gvecs['recon'])
    dot=sum(float((gvecs['type'][n]*gvecs['recon'][n]).sum()) for n in shared)
    nt=sum(float((gvecs['type'][n]**2).sum()) for n in shared)**0.5
    nr=sum(float((gvecs['recon'][n]**2).sum()) for n in shared)**0.5
    cos = dot/max(nt*nr,1e-12)
    dens={}
    for rel,v in ei.items():
        dens[str(rel)]={'edges':int(v.shape[1]) if hasattr(v,'shape') else int(len(v))}
    out[f] = {'ckp_sha256':sha,'graph_state':ck['graph_state'],
        'selected_epoch':ck['selected_epoch'],'replay_exact':replay_exact,
        'meta_consistent':meta_ok,'grad':gn,'cos_type_recon':cos,'ei_edges':dens}
    print(f,'replay',replay_exact,'meta',meta_ok,'cos',round(cos,4),
          {k:round(v['total'],3) for k,v in gn.items()})
json.dump(out,open('G7_A2_ATTRIBUTION_RAW.json','w'),indent=2,default=str)
