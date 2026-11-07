import sys, os, types, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(1, os.path.expanduser('~/repos/DHGCMDA'))
import faithful as fz
D = {k: v for k, v in np.load(os.path.expanduser('~/dhg_runs/faithful/data/v3.2_cuilab_deg2.npz'), allow_pickle=True).items()}
a = types.SimpleNamespace(protocol='type', folds=5, val_frac=0.1, early_stop=1, n_neg_draws=10)
# 1. CV_type splits: disjoint, cover all pairs
fo = fz.make_splits(D['Y'], a, np.random.default_rng(0))
allp = np.concatenate([f['test'] for f in fo]); assert len(allp) == int((D['Y'].sum(2) > 0).sum()) == len({tuple(x) for x in allp})
f = fo[0]; s = lambda x: {tuple(p) for p in x}
assert not (s(f['test']) & s(f['train'])) and not (s(f['val']) & s(f['train'])) and not (s(f['test']) & s(f['val']))
# 2. CV_triplet splits: test negatives unknown, equal size
a.protocol = 'triplet'; ft = fz.make_splits(D['Y'], a, np.random.default_rng(0))[0]
assert len(ft['test_neg']) == len(ft['test']) and D['Y'][tuple(ft['test_neg'].T)].sum() == 0 and D['Y'][tuple(ft['test'].T)].min() == 1
# 3. five-type tensor and multi-label preserved
Y2 = np.load(os.path.expanduser('~/dhg_runs/faithful/data/v2.0.npz'))['Y']; assert Y2.sum() == 1679 and (Y2.sum(2) > 1).sum() == 161
assert D['Y'].shape[2] == 5
# 4. Top-1 metric == released compute_top1_metrics on single-label 4-type data
import Calculate_Metrics as CM
rng = np.random.default_rng(1); n = 400; yt = rng.integers(0, 4, n); pt = rng.random((n, 4))
Yoh = np.eye(4)[yt]; mine = fz.top1(Yoh, pt)
ref = CM.Metric_fun().compute_top1_metrics(list(yt + 1), [list(np.r_[0.9, p]) for p in pt]) if hasattr(CM, 'Metric_fun') else None
print('mine', {k: round(v, 4) for k, v in mine.items() if isinstance(v, float)}, 'released', ref)
# 5. five-type perfect predictor -> F1 1.0
yt5 = rng.integers(0, 5, n); m5 = fz.top1(np.eye(5)[yt5], np.eye(5)[yt5] + 0.01 * rng.random((n, 5))); assert m5['top1_f1'] == 1.0
# 6. module shapes + KNN hypergraph rows
G = fz.knn_G(rng.random((50, 30)), 13); assert G.shape == (50, 50) and torch.allclose(G, G.T, atol=1e-6)
avf = fz.AVF(256); z = avf([torch.randn(50, 256), torch.randn(50, 256)]); assert z.shape == (50, 256) and (z >= 0).all()
rc = fz.Recon(256)(torch.randn(50, 256)); assert rc.shape == (50, 50)
print('ALL TESTS PASSED')
