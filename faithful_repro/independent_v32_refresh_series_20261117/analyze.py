import hashlib,json,math,statistics,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;R=P/'run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def th(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
d=json.loads((R/'completion.json').read_text());assert d['status']=='PASS' and d['final_epoch']==650 and d['capture_count']==129 and d['manifest_sha256']==sha(R/'out/manifest.json')
m=json.loads((R/'out/manifest.json').read_text());hist=m['dynamic_history'];traj=m['trajectory'];assert len(hist)==130 and len(traj)==650
records=[];prevH=None
for h in hist:
 e=h['epoch'];f=R/('refresh_%04d.pt'%e);s=torch.load(f,weights_only=True);assert s['epoch']==e;a=s['refresh_artifacts'];prod=s['producer'];ng=s['new_G'];og=s['old_G'];row={'refresh_epoch':e,'file_sha256':sha(f)}
 for mod in ('mi','dis'):
  assert th(prod['gram_'+mod])==h['producer_gram_'+mod+'_sha256'];assert th(a['normalized_'+mod])==h['normalized_'+mod+'_sha256'];assert th(a['incidence_'+mod])==h['incidence_'+mod+'_sha256']
  for v in (1,2):assert th(ng[mod+'_v'+str(v)])==h['G_'+mod+'_v'+str(v)+'_sha256'] and th(og[mod+'_v'+str(v)])==h['input_G_sha'][mod+'_v'+str(v)]
 H=a['incidence_mi'];dv=H.sum(1);norm=a['normalized_mi'].clone();norm.fill_diagonal_(float('-inf'));sv=torch.sort(norm,dim=1,descending=True).values;k=int(m['cfg']['knn_k']);gap=sv[:,k-1]-sv[:,k];G=ng['mi_v1'];old=og['mi_v1'];change=float((G-old).double().norm()/old.double().norm())
 row.update({'G_mi_relative_change':change,'incidence_top13_share':float(torch.topk(dv,13).values.sum()/dv.sum()),'exact_boundary_ties':int((gap==0).sum()),'near_boundary_1e-6':int((gap.abs()<=1e-6).sum()),'H_membership_turnover_fraction':float((H!=prevH).sum()/(2*H.sum())) if prevH is not None else None,'H_degree_max':float(dv.max())});prevH=H
 if e<650:
  before=traj[e-1];after=traj[e];assert before['epoch']==e and after['epoch']==e+1 and before['dynamic_refreshed'];row.update({'loss_before':before['train_loss'],'loss_after':after['train_loss'],'loss_ratio':after['train_loss']/before['train_loss'],'spike':after['train_loss']>2*before['train_loss']})
 records.append(row)
sp=[x for x in records if x.get('spike')];assert [x['refresh_epoch']+1 for x in sp]==[x['epoch'] for x in d['events'] if x['spike']]
series=[x for x in records if x['refresh_epoch']<650];base=[x for x in series if not x['spike']];summary={}
for k in ('G_mi_relative_change','incidence_top13_share','exact_boundary_ties','near_boundary_1e-6','H_membership_turnover_fraction','loss_ratio'):
 a=[x[k] for x in base if x[k] is not None];a.sort();summary[k]={'other_n':len(a),'other_median':statistics.median(a),'other_max':max(a),'spike_values':[{'epoch':x['refresh_epoch']+1,'value':x[k],'fraction_other_at_or_below':sum(v<=x[k] for v in a)/len(a) if x[k] is not None else None} for x in sp]}
out={'status':'PASS','scope':'one instrumented trajectory; 129 refresh with next epoch; exploratory association not causation','protocol_sha256':sha(P/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'capture_sha256':sha(P/'capture.py'),'manifest_sha256':sha(R/'out/manifest.json'),'records':records,'summary':summary}
with (P/'series.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'n_refresh':len(records),'compared':len(series),'spikes':[x['refresh_epoch']+1 for x in sp],'summary':summary}),flush=True)
