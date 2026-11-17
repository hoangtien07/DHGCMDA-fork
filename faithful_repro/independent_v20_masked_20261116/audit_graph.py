"""Rebuild fold graphs independently, checking actual PyG edges and attributes."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from mask import train_matrix

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'independent_v20_release_20261116'
sys.path.insert(0, str(BASE / 'pinned_release'))
from main_experiments_hetero1 import create_hetero_data_optimized
from prepareData import read_association_csv, read_txt

torch.set_num_threads(2)
data = BASE / 'pinned_release' / 'v2.0_495m383D'
full = read_association_csv(str(data / 'multi_all_mirna_disease_pairs_without_negative.csv')).numpy()
dd = read_txt(str(data / 'D_SSM1.txt'))
mm = read_txt(str(data / 'M_FSM.txt'))
rows = []
for i in range(1, 6):
    d = np.load(BASE / 'runs' / 'full' / f'fold_{i}.npz')
    masked = train_matrix(full, d['train_pos'], d['test_pos'])
    graph = create_hetero_data_optimized([dd, mm, None, None, torch.from_numpy(masked)])
    edge = graph['miRNA', 'associates', 'disease'].edge_index.numpy().T
    attr = graph['miRNA', 'associates', 'disease'].edge_attr.numpy().ravel()
    train = set(map(tuple, d['train_pos'])); test = set(map(tuple, d['test_pos']))
    assert set(map(tuple, edge)) == train and not set(map(tuple, edge)) & test
    assert np.array_equal(attr, masked[edge[:, 0], edge[:, 1]])
    rows.append({'fold': i, 'train_edges': len(edge), 'test_edges': 0,
                 'edge_attributes_equal_train_types': True})
out = {'status': 'PASS', 'scope': 'association edges in initial PyG graph; static similarities not audited', 'folds': rows}
(HERE / 'graph_audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
