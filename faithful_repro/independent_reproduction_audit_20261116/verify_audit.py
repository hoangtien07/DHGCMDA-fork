"""Independent numerical checks with scikit-learn; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = json.loads((HERE / 'metrics_and_objective.json').read_text(encoding='utf-8'))
    checks = []
    for key, row in results.items():
        if 'path' not in row:
            continue
        z = np.load(ROOT / row['path'])
        y = z['y']; pred = z['logits'].argmax(1)
        onehot = np.eye(y.shape[1], dtype=int)[pred]
        first = y.argmax(1)
        p = accuracy_score(first, pred)
        r = balanced_accuracy_score(first, pred)
        harmonic = 2 * p * r / (p + r) if p + r else 0.
        expected = row['metrics']
        library = {'macro_f1':f1_score(y, onehot, average='macro', zero_division=0),
                   'micro_f1':f1_score(y, onehot, average='micro', zero_division=0),
                   'pair_average_f1':f1_score(y, onehot, average='samples', zero_division=0),
                   'first_label_hybrid_f1':harmonic}
        audited = {'macro_f1':expected['macro_f1'], 'micro_f1':expected['micro_f1'],
                   'pair_average_f1':expected['pair_average']['f1'],
                   'first_label_hybrid_f1':expected['first_label_hybrid']['f1']}
        for name in library:
            assert abs(library[name] - audited[name]) < 1e-12, (key, name, library, audited)
        checks.append({'export':key, 'independent_library_metrics':library})
    inputs = json.loads((HERE / 'input_sha256.json').read_text(encoding='utf-8'))
    for path, expected in inputs.items():
        assert digest(ROOT / path) == expected, f'Input drift: {path}'
    with (HERE / 'FIDELITY_MATRIX.csv').open(encoding='utf-8-sig', newline='') as h:
        matrix = list(csv.DictReader(h))
    assert len(matrix) == 18 and len({r['id'] for r in matrix}) == 18
    assert all(None not in r and all(r.values()) for r in matrix)
    source = json.loads((HERE / 'source_checks.json').read_text(encoding='utf-8'))
    assert source['recursive_tree_complete'] and source['v32_named_paths'] == []
    result = {'status':'PASS', 'independent_sklearn_version':sklearn.__version__,
              'export_checks':checks, 'input_sha256_rechecks':len(inputs),
              'fidelity_rows':len(matrix), 'no_checkpoint_or_outer_predictions_loaded':True}
    (HERE / 'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    inventory = {str(p.relative_to(HERE)).replace('\\','/'):digest(p)
                 for p in sorted(HERE.rglob('*'))
                 if p.is_file() and p.name != 'ARTIFACT_SHA256.json' and '__pycache__' not in p.parts}
    (HERE / 'ARTIFACT_SHA256.json').write_text(json.dumps(inventory, indent=2)+'\n', encoding='utf-8')
    for path, expected in inventory.items():
        assert digest(HERE / path) == expected
    print(json.dumps({'status':'PASS', 'exports_verified':len(checks),
                      'metrics_per_export':4, 'input_hashes_verified':len(inputs),
                      'fidelity_rows':len(matrix), 'artifacts_hashed':len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
