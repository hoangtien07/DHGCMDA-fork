"""Validate reported counts and seal the independently calculated results."""
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
res=json.loads((P/'results.json').read_text(encoding='utf-8'))
assert res['fixture']=='PASS' and len(res['folds'])==2
for fold in res['folds']:
    n=fold['validation_pairs'];assert sum(fold['any_hit_overlap'].values())==n
    for method in ('model','disease_prior'):
        classes=fold[method]['per_class'];assert len(classes)==5
        assert sum(v['predicted'] for v in classes.values())==n
        assert all(v['support']==v['tp']+v['fn'] for v in classes.values())
        assert all(v['predicted']==v['tp']+v['fp'] for v in classes.values())
with (P/'per_class.csv').open(encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
assert len(rows)==20 and len({(r['fold'],r['method'],r['type']) for r in rows})==20
names=['CONTRACT.md','REPORT.md','analyze.py','verify.py','results.json','per_class.csv']
manifest={n:{'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in names}
(P/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('PASS: 2 validation exports; 20 per-class rows; contingency counts consistent; 6 files hashed')
