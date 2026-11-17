"""Read-only reproduction audit. Run with uv --with numpy --with pymupdf.

No torch/checkpoint loading, fitting, outer-test predictions, or pipeline writes.
Metric definitions are fixed in PLAN.md before this script was run.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import urllib.request

import numpy as np
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = HERE / 'sources'
READS = {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(path):
    path = Path(path)
    b = path.read_bytes()
    h = sha(b)
    key = str(path.relative_to(ROOT)).replace('\\', '/')
    if key in READS:
        assert READS[key] == h, f'Input changed during audit: {key}'
    READS[key] = h
    return b


def js(path):
    return json.loads(read(path).decode('utf-8-sig'))


def load(path, allow_pickle=False):
    read(path)
    return np.load(path, allow_pickle=allow_pickle)


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                       allow_nan=False) + '\n', encoding='utf-8')


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'DHGCMDA-independent-readonly-audit'})
    with urllib.request.urlopen(req, timeout=35) as response:
        return response.read()


def collect_sources():
    SNAP.mkdir(exist_ok=True)
    inventory = SNAP / 'inventory.json'
    if inventory.exists():
        result = js(inventory)
        for item in result['files']:
            assert sha(read(SNAP / item['file'])) == item['sha256']
        return result
    base = 'https://api.github.com/repos/CDMBlab/DHGCMDA'
    commit_bytes = fetch(base + '/commits/main')
    commit = json.loads(commit_bytes)
    rev = commit['sha']
    requests = [
        ('github_commit.json', base + '/commits/' + rev, commit_bytes),
        ('github_tree.json', base + '/git/trees/' + rev + '?recursive=1', None),
    ]
    for name in ['Calculate_Metrics.py', 'hetero_model.py', 'main_experiments_hetero1.py',
                 'prepareData.py', 'param.py', 'README.md', 'trainData.py']:
        requests.append((name, f'https://raw.githubusercontent.com/CDMBlab/DHGCMDA/{rev}/{name}', None))
    records = []
    for name, url, payload in requests:
        b = payload if payload is not None else fetch(url)
        (SNAP / name).write_bytes(b)
        records.append({'file': name, 'url': url, 'sha256': sha(b), 'bytes': len(b)})
    result = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'commit': rev,
              'files': records, 'scope': 'public main tree, not private branches or author machine'}
    save('sources/inventory.json', result)
    return result


def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(y, scores):
    y, s = np.asarray(y, dtype=int), np.asarray(scores)
    assert y.ndim == 2 and y.shape == s.shape and len(y)
    assert np.isin(y, [0, 1]).all() and (y.sum(1) > 0).all() and np.isfinite(s).all()
    n, c = y.shape
    pred = s.argmax(1)
    one = np.eye(c, dtype=int)[pred]
    tp = (one & y).sum(0)
    fp = (one & (1 - y)).sum(0)
    fn = ((1 - one) & y).sum(0)
    p = np.divide(tp, tp + fp, out=np.zeros(c), where=tp + fp > 0)
    r = np.divide(tp, tp + fn, out=np.zeros(c), where=tp + fn > 0)
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(c), where=2 * tp + fp + fn > 0)
    hit = y[np.arange(n), pred]
    first = y.argmax(1)
    first_p = float(np.mean(pred == first))
    first_r = float(np.mean([np.mean(pred[first == k] == k) for k in np.unique(first)]))
    any_p = float(hit.mean())
    any_r = float(r[y.sum(0) > 0].mean())
    counts = y.sum(1)
    return {
        'n_pairs': n, 'n_typed_positives': int(y.sum()), 'multitype_pairs': int((counts > 1).sum()),
        'ties_for_max': int(((s == s.max(1, keepdims=True)).sum(1) > 1).sum()),
        'first_label_hybrid': {'precision': first_p, 'recall': first_r, 'f1': harmonic(first_p, first_r)},
        'any_hit_hybrid': {'precision': any_p, 'recall': any_r, 'f1': harmonic(any_p, any_r)},
        'macro_f1': float(f.mean()),
        'micro_f1': float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
        'pair_average': {'precision': any_p, 'recall': float(np.mean(hit / counts)),
                         'f1': float(np.mean(2 * hit / (1 + counts)))},
        'any_hit': any_p,
        'per_class': {'tp': tp.tolist(), 'fp': fp.tolist(), 'fn': fn.tolist(),
                      'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(),
                      'support': y.sum(0).tolist(), 'predicted': one.sum(0).tolist()},
    }


def released_method():
    tree = ast.parse(read(SNAP / 'Calculate_Metrics.py').decode('utf-8-sig'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_top1_metrics')
    ns = {'np': np, 'defaultdict': defaultdict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned released method>', 'exec'), ns)
    local = ast.parse(read(ROOT / 'Calculate_Metrics.py').decode('utf-8-sig'))
    local_node = next(n for n in ast.walk(local) if isinstance(n, ast.FunctionDef) and n.name == node.name)
    return ns['compute_top1_metrics'], ast.dump(node) == ast.dump(local_node)


def fixtures(original):
    checks = {}
    eye4, eye5 = np.eye(4), np.eye(5)
    a = original(None, np.arange(1, 5), np.column_stack([np.ones(4), eye4]))
    b = original(None, np.arange(1, 6), np.column_stack([np.ones(5), eye5]))
    assert a['top1_f1'] == 1 and a['valid_samples'] == 4
    assert b['top1_f1'] == 0 and b['valid_samples'] == 0
    checks['released_4_types_perfect'] = float(a['top1_f1'])
    checks['released_5_types_perfect'] = {'f1': float(b['top1_f1']), 'valid': b['valid_samples']}
    for c in [4, 5]:
        v = evaluate(np.eye(c, dtype=int), np.eye(c))
        assert v['macro_f1'] == v['micro_f1'] == v['first_label_hybrid']['f1'] == v['pair_average']['f1'] == 1
    y = np.array([[1, 1], [0, 1]])
    s = np.array([[0., 2.], [0., 1.]])
    v = evaluate(y, s)
    assert np.isclose(v['macro_f1'], .5) and np.isclose(v['micro_f1'], .8)
    assert np.isclose(v['pair_average']['f1'], 5 / 6)
    checks['multihot_hand_fixture'] = v
    rng = np.random.default_rng(20261116)
    for _ in range(30):
        y = rng.integers(0, 2, (40, 4)); y[np.arange(40), rng.integers(4, size=40)] = 1
        s = rng.normal(size=y.shape)
        a = original(None, y, np.column_stack([np.ones(len(y)), s]))
        v = evaluate(y, s)['first_label_hybrid']
        assert np.allclose([a['top1_precision'], a['top1_recall'], a['top1_f1']],
                           [v['precision'], v['recall'], v['f1']], atol=1e-14)
    checks['generalization_agrees_released_K4'] = '30 randomized multihot fixtures passed'
    # The order-sensitive convention can change merely by relabelling channels.
    a = evaluate(np.array([[1,1],[0,1]]), np.array([[0.,2.],[0.,1.]]))
    b = evaluate(np.array([[1,1],[1,0]]), np.array([[2.,0.],[1.,0.]]))
    assert a['macro_f1'] == b['macro_f1'] and a['any_hit'] == b['any_hit']
    assert a['first_label_hybrid']['f1'] != b['first_label_hybrid']['f1']
    checks['simultaneous_permutation_changes_first_label_only'] = True
    # Common-denominator algebra: normalization of just the class weights changes
    # relative coefficients in a combined objective unless auxiliary terms scale too.
    raw_type, auxiliary, c, mass = 7., 11., 3., 17.
    executed = (c * raw_type + auxiliary) / mass
    assert np.isclose(executed * mass / c, raw_type + auxiliary / c)
    checks['objective_weight_scaling_hand_fixture'] = True
    return checks


def paper_and_data():
    import pymupdf
    pdf = next(ROOT.glob('*.pdf'))
    read(pdf)
    doc = pymupdf.open(pdf)
    for i in list(range(8, 25)) + [26, 27, 29]:
        (SNAP / f'paper_p{i+1:02}.txt').write_text(doc[i].get_text(), encoding='utf-8')
    for i in [22, 29]:
        doc[i].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(SNAP / f'paper_p{i+1:02}.png')
    doc[29].get_pixmap(matrix=pymupdf.Matrix(5, 5), clip=pymupdf.Rect(270, 40, 520, 250)).save(SNAP / 'figure4a_v32_detail.png')
    a = load(ROOT / 'faithful_repro/data/v3.2_ref_MDAv3.2-3.npz', allow_pickle=True)
    y = a['Y']; names = a['type_names'].astype(str).tolist()
    mats = []
    for name in names:
        p = ROOT / 'HMDD_data/MDAv3.2-3' / f'{name}.csv'
        rows = list(csv.reader(read(p).decode('utf-8-sig').splitlines()))
        mats.append(np.array([[float(v) for v in row[1:]] for row in rows[1:]]))
    assert np.array_equal(np.stack(mats, 2), y)
    snapshot_y = load(ROOT / 'v3.2_spld_paper/Y_multilabel.npy')
    assert np.array_equal(snapshot_y, y)
    strict = load(ROOT / 'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz', allow_pickle=True)
    assert np.array_equal(strict['Y'], y)
    for key in ['mir_names', 'dis_names', 'type_names']:
        assert np.array_equal(strict[key], a[key])
    identity = {}
    for key, filename in [('mir_names','mi_name_3.2_3.csv'), ('dis_names','dis_name_3.2_3.csv')]:
        rows = list(csv.reader(read(ROOT / 'HMDD_data/MDAv3.2-3' / filename).decode('utf-8-sig').splitlines()))
        values = [row[1] for row in rows[1:]]
        assert values == a[key].astype(str).tolist()
        identity[key] = {'n': len(values), 'ordered_sha256': sha('\n'.join(values).encode())}
    full = {'shape': list(y.shape), 'types': names, 'per_type': y.sum((0, 1)).astype(int).tolist(),
            'triplets': int(y.sum()), 'pairs': int((y.sum(2) > 0).sum()),
            'multitype_pairs': int((y.sum(2) > 1).sum()),
            'min_mirna_pair_degree': int((y.sum(2) > 0).sum(1).min()),
            'min_disease_pair_degree': int((y.sum(2) > 0).sum(0).min()),
            'min_mirna_typed_degree': int(y.sum((1, 2)).min()),
            'single_typed_link_mirnas': a['mir_names'][y.sum((1, 2)) < 2].astype(str).tolist(),
            'csv_snapshot_reference_strict_labels_equal': True, 'ordered_names_equal': identity,
            'DHGCMDA_author_tensor_identity': 'unknown: payload is recovered SPLD lineage, not a DHGCMDA v3.2 release'}
    save('data_identity.json', full)
    return y, names


def permutation_diagnostic(y, s):
    rows = []
    for perm in itertools.permutations(range(y.shape[1])):
        p = list(perm)
        v = evaluate(y[:, p], s[:, p])
        rows.append({'order': p, 'first_label_hybrid_f1': v['first_label_hybrid']['f1'],
                     'macro_f1': v['macro_f1'], 'any_hit': v['any_hit']})
    base = evaluate(y, s)
    if base['ties_for_max'] == 0:
        assert all(np.isclose(r['macro_f1'], base['macro_f1']) and np.isclose(r['any_hit'], base['any_hit']) for r in rows)
    return {'all_permutations': rows,
            'first_label_hybrid_range': [min(r['first_label_hybrid_f1'] for r in rows),
                                         max(r['first_label_hybrid_f1'] for r in rows)],
            'warning': 'ordering sensitivity diagnostic; no order selected by score'}


def disease_prior(train_pairs, train_y, val_pairs, n_dis):
    # Fixed beforehand: counts per disease, with 1e-12 global-frequency tie break;
    # unseen disease falls back to global train support. No validation labels used.
    counts = np.zeros((n_dis, train_y.shape[1]))
    np.add.at(counts, train_pairs[:, 1], train_y)
    glob = train_y.sum(0).astype(float)
    scores = counts[val_pairs[:, 1]].copy()
    missing = scores.sum(1) == 0
    scores[missing] = glob
    scores += (glob / glob.sum()) * 1e-12
    return scores, int(missing.sum())


def run_exports(y_full, names):
    all_results = {}
    for name in ['u1_horizon650_fold0_rev1', 'u1_fold1_650_selection_rev1']:
        out = ROOT / 'faithful_repro/paper_guided' / name / 'out'
        manifest = js(out / 'manifest.json')
        archived_source = out.parent / 'source'
        verified_modules = {}
        for name_, digest in {**manifest['source_hashes'], **manifest['imported_pure_hashes']}.items():
            assert sha(read(archived_source / name_)) == digest, f'Archived executed module hash mismatch: {name_}'
            verified_modules[name_] = digest
        arrays = {}
        for split in ['train', 'val']:
            p = out / f'predictions_{split}.npz'
            meta = js(out / f'predictions_{split}.meta.json')
            assert sha(read(p)) == meta['npz_sha256']
            z = load(p)
            pairs, y, scores = z['pairs'], z['y'], z['logits']
            assert len(np.unique(pairs, axis=0)) == len(pairs)
            assert np.array_equal(y, y_full[pairs[:, 0], pairs[:, 1]])
            h = sha(y.astype(np.float64).tobytes())
            # The manifest hashes pre-export float64 multihot labels.
            assert h == manifest['data_hashes'][f'y_{split}']
            assert sha(pairs.tobytes()) == manifest['data_hashes'][f'{split}_pairs']
            arrays[split] = (pairs, y, scores)
            v = evaluate(y, scores)
            assert abs(v['macro_f1'] - manifest['best'][f'{split}_macro_f1']) < 1e-12
            assert abs(v['micro_f1'] - manifest['best'][f'{split}_micro_f1']) < 1e-12
            # Cross-check the existing frozen metric implementation, never modify it.
            module_path = archived_source / 'metrics.py'
            read(module_path)
            spec = importlib.util.spec_from_file_location('frozen_metric_crosscheck', module_path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            existing = module.type_metrics(y, scores)
            assert abs(v['macro_f1'] - existing['primary']['macro_f1']) < 1e-12
            assert abs(v['first_label_hybrid']['f1'] - existing['legacy_first_label_hybrid']['f1']) < 1e-12
            all_results[f'{name}/{split}'] = {'metrics': v, 'epoch': meta['best_epoch'],
                'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'sha256': meta['npz_sha256'],
                'selection': 'best macro-F1 among 65 validation looks; train is in-sample; no outer score'}
        trp, try_, _ = arrays['train']; vp, vy, vs = arrays['val']
        assert not set(map(tuple, trp)) & set(map(tuple, vp))
        prior_s, missing = disease_prior(trp, try_, vp, y_full.shape[1])
        all_results[f'{name}/disease_prior_val'] = {'metrics': evaluate(vy, prior_s),
            'rule': 'train-only disease type counts; global train support tie-break; global fallback',
            'unseen_diseases': missing, 'warning': 'separate declared diagnostic; not necessarily archived prior recipe'}
        all_results[f'{name}/val']['permutation_diagnostic'] = permutation_diagnostic(vy, vs)
        counts = try_.sum(0)
        beta = manifest['cfg']['beta']
        raw_w = (1 - beta) / (1 - beta ** counts)
        c = len(names) / raw_w.sum()
        w = c * raw_w
        M = float((try_ * w).sum())
        telemetry = []
        for row in [manifest['trajectory'][0], manifest['best']]:
            mass = row['unit_mass']
            parts = {'type': row['L_type'],
                     'intra': manifest['cfg']['lam_intra'] * (row['intra_n_mi'] + row['intra_n_dis']) * row['L_intra'] / mass,
                     'inter': manifest['cfg']['lam_inter'] * (row['inter_info_sum'] + manifest['cfg']['inter_margin_coef'] * row['inter_margin_term']) / mass,
                     'reconstruction': manifest['cfg']['lam_recon'] * (row['recon_sum_mi'] + row['recon_sum_dis']) / mass}
            total = sum(parts.values())
            assert np.isclose(total, row['train_loss'], rtol=2e-6)
            telemetry.append({'epoch': row['epoch'], 'contributions': parts, 'sum': total,
                              'recorded_loss': row['train_loss'], 'share': {k:v/total for k,v in parts.items()}})
        all_results[f'{name}/objective'] = {'train_counts': counts.tolist(),
            'raw_eq34_weights': raw_w.tolist(), 'sumC_weights': w.tolist(), 'weight_scale_c': float(c),
            'mass_recomputed_float64': M, 'mass_manifest_float32': manifest['trajectory'][0]['unit_mass'],
            'effective_aux_coeff_vs_raw_weight_sum_objective': {k: manifest['cfg'][k]/c for k in ['lam_intra','lam_inter','lam_recon']},
            'telemetry': telemetry,
            'verified_executed_modules': verified_modules,
            'interpretation': 'Multiplying executed loss by M/c gives raw-weight type sum plus auxiliary sums with lambda/c. This is an algebraic comparison with literal raw-weight summed objective, not evidence of the unpublished author implementation.'}
    save('metrics_and_objective.json', all_results)
    columns = ['run', 'n_pairs', 'macro_f1', 'micro_f1', 'any_hit', 'first_label_hybrid_f1', 'any_hit_hybrid_f1', 'pair_average_f1']
    with (HERE / 'METRIC_COMPARISON.csv').open('w', newline='', encoding='utf-8-sig') as h:
        writer = csv.DictWriter(h, fieldnames=columns); writer.writeheader()
        for name, row in all_results.items():
            if 'metrics' not in row:
                continue
            v = row['metrics']
            writer.writerow({'run': name, **{k:v[k] for k in columns[1:5]},
                             'first_label_hybrid_f1': v['first_label_hybrid']['f1'],
                             'any_hit_hybrid_f1': v['any_hit_hybrid']['f1'], 'pair_average_f1': v['pair_average']['f1']})
    return all_results


def main():
    read(Path(__file__))
    read(HERE / 'PLAN.md')
    sources = collect_sources()
    original, same = released_method()
    checks = fixtures(original)
    y, names = paper_and_data()
    result = run_exports(y, names)
    tree = js(SNAP / 'github_tree.json')
    assert not tree.get('truncated', False)
    paths = [x['path'] for x in tree['tree']]
    source_result = {'commit': sources['commit'], 'recursive_tree_complete': True,
                     'paths': paths, 'v32_named_paths': [p for p in paths if '3.2' in p or 'v32' in p.lower()],
                     'released_metric_same_AST_as_local': same}
    save('source_checks.json', source_result)
    save('verification.json', {'fixtures': checks, 'source_checks': source_result,
                              'all_export_hash_pair_label_metric_checks_passed': True,
                              'no_pipeline_source_changed_during_run': True,
                              'numpy_version': np.__version__, 'python': sys.version,
                              'scored_exports': [k for k,v in result.items() if 'metrics' in v]})
    # Exact primary-source labels, visually verified from the page-30 render.
    published = {'paper_pdf_sha256': READS[str(next(ROOT.glob('*.pdf')).relative_to(ROOT)).replace('\\','/')],
        'table3_v32': {'precision':.7915,'recall':.9421,'f1':.8600},
        'figure4a_v32': {'precision':.6005,'recall':.5253,'f1':.5678},
        'table3_v20': {'precision':.5842,'recall':.6341,'f1':.5970},
        'figure4a_v20': {'precision':.5642,'recall':.6341,'f1':.5970},
        'table4_v32': {'aupr':.9271,'auc':.9181,'f1':.8674},
        'figure4b_v32': {'aupr':.9439,'auc':.9420,'f1':.8768},
        'warning':'Hand transcribed from rendered primary PDF; no metric/target selected by closeness.'}
    for key, values in list(published.items()):
        if isinstance(values,dict) and 'precision' in values:
            values['harmonic_of_displayed_P_R'] = harmonic(values['precision'],values['recall'])
    save('published_targets.json', published)
    for path, digest in READS.items():
        assert sha((ROOT / path).read_bytes()) == digest, f'Input changed: {path}'
    save('input_sha256.json', READS)
    print(json.dumps({'status':'PASS', 'commit':sources['commit'],
                      'v32_named_paths':source_result['v32_named_paths'],
                      'validation':{k:{'macro_f1':v['metrics']['macro_f1'],
                                       'first_label_hybrid_f1':v['metrics']['first_label_hybrid']['f1']}
                                    for k,v in result.items() if k.endswith('/val')},
                      'objective_scales':{k:v['weight_scale_c'] for k,v in result.items() if k.endswith('/objective')}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
