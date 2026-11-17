"""Hash the reproducibility evidence, excluding generated caches and this manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
paths = sorted(root.glob('*.py')) + [root / x for x in (
    'PROTOCOL.md', 'INVENTORY.md', 'REPORT.md', 'source_manifest.json',
    'existing_baseline_audit.json', 'split_audit.json', 'label_collapse_audit.json',
    'feature_oracle.json',
)]
paths += [root / 'runs/full' / x for x in (
    'contract.json', 'completion.json', 'console.log', 'metric_audit.json',
    'partial_metric_audit.json', 'fold_1.npz', 'fold_2.npz', 'fold_3.npz',
    'fold_4.npz', 'fold_5.npz',
)]
assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
data = {str(p.relative_to(root)).replace('\\', '/'): {
    'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
} for p in paths}
target = root / 'evidence_manifest.json'
target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Sealed {len(data)} evidence files in {target}')
