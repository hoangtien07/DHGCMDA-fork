import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
items=[('diagnostics','PLAN.md'),('strata','PROTOCOL_STRATA.md'),('negative_overlap','PROTOCOL_NEGATIVES.md'),('bank_size','PROTOCOL_BANK_SIZE.md'),('canary','PROTOCOL_CANARY.md'),('data_audit','PROTOCOL_DATA.md'),('triplet_audit','PROTOCOL_TRIPLET.md'),('triplet_baseline','PROTOCOL_TRIPLET_BASELINE.md'),('triplet_verify','PROTOCOL_TRIPLET_VERIFY.md'),('coverage','PROTOCOL_COVERAGE.md'),('split_audit','PROTOCOL_SPLIT.md'),('upstream_graph_fixture','PROTOCOL_UPSTREAM_GRAPH.md')]
checks=[]
for name,protocol in items:
 j=read(P/(name+'.json'));assert j['status']=='PASS' and j['script_sha256']==sha(P/(name+'.py'));field='plan_sha256' if name=='diagnostics' else 'protocol_sha256';assert j[field]==sha(P/protocol),(name,field);checks.append(name)
a=read(P/'data_audit.json');assert a['shape']==[411,271,5] and a['typed_counts']==[1155,403,2293,3997,3900] and a['n_pairs']==8735 and a['n_multitype']==2110 and sum(a['n_pairs_by_type_count'].values())==8735
n=read(P/'negative_overlap.json');s=read(P/'bank_size.json');assert len(n['rows'])==len(s['rows'])==15
for r in s['rows']:assert r['subset'] and r['val_in_bank1']<=r['val_in_bank10'] and r['n_bank1']<=r['n_bank10']
c=read(P/'canary.json');assert len(c['rows'])==5 and all(r['changed_val_labels']>0 and r['verified_train_mask_calls']==2 for r in c['rows'])
t=read(P/'triplet_audit.json');assert t['known_positive_triplets']+t['unknown_triplets']==t['possible_triplets'] and t['unknown_triplets_known_pair_other_type']+t['unknown_triplets_no_known_pair']==t['unknown_triplets'] and len(t['rows'])==25
v=read(P/'triplet_verify.json');assert v['baseline_sha256']==sha(P/'triplet_baseline.json') and len(v['rows'])==5
u=read(P/'upstream_graph_fixture.json');assert u['association_edges_unchanged'] and len(u['baseline_mi_edges'])==0 and len(u['reconstructed_mi_edges'])==2
out={'status':'PASS','scope':'integrity and cross-check of local evidence; not independent reproduction','script_sha256':sha(__file__),'checked':checks}
(P/'core_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks),'evidence artifacts and cross-checks')
