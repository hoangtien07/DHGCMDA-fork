"""Validate the audit outputs and seal their hashes for review."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
summary=json.loads((P/'summary.json').read_text(encoding='utf-8'))
rebuild=json.loads((P/'feature_rebuild.json').read_text(encoding='utf-8'))
assert summary['status']==rebuild['status']=='PASS'
assert len(rebuild['folds'])==2 and all(x['functional_block_elementwise_equal'] and x['semantic_block_elementwise_equal'] for x in rebuild['folds'])
with (P/'pair_ledger.csv').open(encoding='utf-8',newline='') as f:pairs=list(csv.DictReader(f))
with (P/'triplet_ledger.csv').open(encoding='utf-8',newline='') as f:trips=list(csv.DictReader(f))
assert len(pairs)==8735 and len(trips)==11748
assert len({(r['mi_idx'],r['di_idx']) for r in pairs})==len(pairs)
assert len({(r['mi_idx'],r['di_idx'],r['type_idx']) for r in trips})==len(trips)
assert len(summary['u1_fold0_selection']['candidate_grouped_split'])==5
assert all(x['cross_segment_groups']==0 for x in summary['u1_fold0_selection']['candidate_grouped_split'])
assert summary['u1_fold0_selection']['heldout_positive_entries_in_mask']==0
assert summary['u1_fold1_selection']['heldout_positive_entries_in_mask']==0
names=['PLAN.md','REPORT.md','audit.py','rebuild_features.py','verify.py','summary.json','feature_rebuild.json','pair_ledger.csv','triplet_ledger.csv']
manifest={}
for name in names:
    p=P/name;manifest[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(P/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('PASS: 8,735 unique pairs; 11,748 unique triplets; 5 grouped splits; 2 rebuilt U1 bundles; 9 sealed files')
