import copy,hashlib,inspect,json,random,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;FR=HERE.parent
SRC=FR/'paper_guided/u1_horizon650_fold0_rev1/source';D=FR/'paper_guided/candidate_fold0_selection_rev1';BASE=FR/'independent_v32_seed_stability_20261117/runs/seed42'
sys.path[:0]=[str(SRC/'stages'),str(SRC)]
import torch_geometric,runner,u1_runner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 assert torch.__version__.split('+')[0]=='2.5.1' and torch_geometric.__version__=='2.6.1' and np.__version__=='1.26.4' and not torch.cuda.is_available()
 basefile=BASE/'out/manifest.json';base=json.loads(basefile.read_text(encoding='utf-8'));cfg=dict(base['cfg']);assert cfg['seed']==cfg['inter_seed']==42 and cfg['max_epochs']==650
 for name,h in {**base['source_hashes'],**base['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
 bundle=D/'candidate_fold_data.npz';meta=D/'recon_meta.json';audit=D/'recon_audits.json';assert sha(bundle)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
 data=runner.load_data_npz(str(bundle));exp=u1_runner.expected_manifest_for_u1(cfg,data,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(bundle));assert exp['data_hashes']==base['data_hashes'] and exp['view_hashes']==base['view_hashes']
 out=HERE/'run';out.mkdir(exist_ok=False);write(out/'expected.json',exp)
 write(out/'launch.json',{'protocol_sha256':sha(HERE/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'bundle_sha256':sha(bundle),'historical_manifest_sha256':sha(basefile),'python':sys.version,'torch':torch.__version__,'pyg':torch_geometric.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 lines,start=inspect.getsourcelines(u1_runner.train_u1)
 def line(s):
  a=[start+i for i,x in enumerate(lines) if x.strip()==s];assert len(a)==1,(s,a);return a[0]
 refresh=line('old_g_sha = {k: _sha_bytes(v.numpy().tobytes()) for k,v in G.items()}');forward=line('out = model(inputs, G, ei, tp_t)');lossline=line('if not torch.isfinite(loss):')
 state={'events':[],'capture_count':0};code=u1_runner.train_u1.__code__
 def graph(G,ei):return ({k:v.detach().clone() for k,v in G.items()},{k:v.detach().clone() for k,v in ei.items()})
 def trace(frame,event,arg):
  if frame.f_code is not code:return None
  if event!='line':return trace
  L=frame.f_locals;e=L.get('epoch',0)
  if frame.f_lineno==refresh and state.get('refresh_epoch')!=e:
   state['refresh_epoch']=e;state['old']=graph(L['G'],L['ei'])
  if frame.f_lineno==forward and state.get('refresh_epoch')==e-1 and state.get('capture_epoch')!=e:
   state['capture_epoch']=e;state['capture_count']+=1;ns=np.random.get_state()
   state['capture']={'epoch':e,'model_state':copy.deepcopy(L['model'].state_dict()),'optimizer_state':copy.deepcopy(L['opt'].state_dict()),'old_graph':state['old'],'new_graph':graph(L['G'],L['ei']),
     'torch_rng':torch.get_rng_state().clone(),'python_rng':random.getstate(),'numpy_rng':{'algorithm':ns[0],'keys':torch.from_numpy(ns[1].astype(np.int64)),'position':ns[2],'has_gauss':ns[3],'cached_gaussian':ns[4]},
     'inputs':{k:v.detach().clone() for k,v in L['inputs'].items()},'train_pairs':L['tp_t'].detach().clone(),'train_y':torch.as_tensor(L['y_tr']).clone(),'weights':L['w_t'].detach().clone(),'positive_mask':L['pos_t'].detach().clone(),'negative_mask':L['neg_t'].detach().clone(),'targets':{k:v.detach().clone() for k,v in L['tensors'].items()},'unit_mass':L['unit_mass'],'cfg':cfg}
  if frame.f_lineno==lossline and state.get('capture_epoch')==e and state.get('checked_epoch')!=e:
   state['checked_epoch']=e;current=float(L['loss'].detach());previous=float(L['trajectory'][-1]['train_loss']);is_spike=current>2*previous
   if is_spike or e==6:
    path=out/('spike_%04d.pt'%e if is_spike else 'control_0006.pt');torch.save(state['capture'],path)
    record={'epoch':e,'previous_loss':previous,'loss':current,'ratio':current/previous,'spike':is_spike,'checkpoint':path.name,'sha256':sha(path)};state['events'].append(record);write(out/('event_%04d.json'%e),record);print('CAPTURE',json.dumps(record),flush=True)
   del state['capture']
  return trace
 with (out/'progress.log').open('x',encoding='utf-8',buffering=1) as f:
  def log(m):print(m,flush=True);print(m,file=f,flush=True)
  sys.settrace(trace)
  try:r=u1_runner.train_u1(cfg,data,str(out/'out'),expected_manifest=exp,meta_path=str(meta),audit_path=str(audit),data_file_sha=sha(bundle),log=log)
  finally:sys.settrace(None)
 assert r['aborted'] is None and r['final_epoch']==650 and state['capture_count']==129
 assert sha(bundle)=='ca14679af63042a7fce40a627da0f8fb04f2be46dd53d13cd4e68a356a689677'
 for name,h in {**base['source_hashes'],**base['imported_pure_hashes']}.items():assert sha(SRC/name)==h,name
 result={'status':'PASS','final_epoch':650,'capture_count':state['capture_count'],'events':state['events'],'manifest_sha256':sha(out/'out/manifest.json'),'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'completion.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
