"""G4.2 — CLI entrypoint + failure-injection tests. Every blocker must
be unbypassable: tampered hashes, blocked profile, unauthorized
overrides, non-finite numerics, wall-cap, atomic writes."""
import os, json, types
import numpy as np
import torch
import pytest
import p5_runner as pr
import p5_evaluator as ev
import p5_diagnostics as dg


def _a(**kw):
    base = dict(hidden=16, proj=8, heads=2, nlayer=1, dropout=0.0, K=3,
                t=0.5, lam1=1.0, lam2=0.3, lam3=1.0, margin=0.5,
                margin_w=0.1, exist_w=0.3, neg_w=0.5, beta=0.99999,
                lr=1e-3, wd=1e-5, neg_ratio=3, update_every=99, theta=0.5,
                early_stop=1, eval_every=1, patience=50, n_neg_draws=2,
                predictor='diag', recon_reduction='sum', diag_leak=0,
                ablation='none', protocol='type', folds=5, only_fold=-1,
                cap_threads=0, threads=1, seed=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


DATA = os.path.join(pr.REPO,
    'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')
FOLDS = os.path.join(pr.REPO,
    'faithful_repro/independent_v32_outer_refit_20261117/evaluation')


def test_dry_run_never_trains(tmp_path, capsys):
    out = str(tmp_path / 'dry.json')
    rc = pr.main(['--dry-run', '--synthetic', '--out', out])
    assert rc == 0
    res = json.load(open(out))
    assert res['synthetic'] and 'resolved_args' in res
    stdout = capsys.readouterr().out
    assert '"optimizer_steps": 0' in stdout
    # no checkpoint or training artifact exists
    assert not os.path.exists(os.path.join(os.path.dirname(out), 'checkpoint.pt'))


def test_dry_run_real_binds_context(tmp_path):
    out = str(tmp_path / 'dry_real.json')
    rc = pr.main(['--dry-run', '--data', DATA, '--fold-dir', FOLDS,
                  '--profile', 'P5_BENCHMARK_UNKNOWN_PROVENANCE', '--out', out])
    assert rc == 0
    res = json.load(open(out))
    assert res['data_sha256'].startswith('85bb9ea4')
    assert len(res['split_sha256']) == 5
    assert res['threshold_grid'] == pr.THRESHOLD_GRID


def test_once_synthetic_end_to_end(tmp_path):
    outdir = str(tmp_path / 'run1')
    rc = pr.main(['--once', '--synthetic', '--seed', '7', '--out', outdir])
    assert rc == 0
    assert os.path.exists(os.path.join(outdir, 'result.json'))
    assert os.path.exists(os.path.join(outdir, 'checkpoint.pt'))
    assert os.path.exists(os.path.join(outdir, 'resolved_manifest.json'))
    ck = torch.load(os.path.join(outdir, 'checkpoint.pt'), weights_only=False)
    assert 'state_dict' in ck and 'thresholds' in ck
    res = json.load(open(os.path.join(outdir, 'result.json')))
    assert np.isfinite(res['macro_f1'])
    assert res['threshold_source'] == 'inner_validation_only'
    # atomic writes leave no .tmp debris
    assert not any(f.endswith('.tmp') for f in os.listdir(outdir))


def test_blocked_profile_rejected():
    with pytest.raises(pr.LockViolation, match='blocked'):
        pr.bind_context('P5_SPLIT_AWARE_EXTERNAL', data_path=DATA, fold_dir=FOLDS)


def test_unauthorized_override_rejected_real_context():
    cfg = pr.load_lock()
    with pytest.raises(pr.LockViolation, match='unauthorized override'):
        pr.args_from_lock(cfg, 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          overrides={'hidden': 999})
    # locked-value override rejected even if whitelisted in synthetic
    with pytest.raises(pr.LockViolation, match='unauthorized override'):
        pr.args_from_lock(cfg, 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          overrides={'lam1': 0.5})


def test_nonfinite_loss_hardstop(monkeypatch):
    D = pr.synthetic_fixture(seed=31)
    fold = pr.synthetic_fold(D['Y'], seed=31)
    a = _a(epochs=3)
    monkeypatch.setattr(pr.fz, 'inter_loss',
                        lambda *x, **k: torch.tensor(float('nan')))
    with pytest.raises(pr.HardStop, match='non_finite_loss'):
        pr.train_fold(D, fold, a, epochs=3)


def test_nonfinite_grad_hardstop(monkeypatch):
    D = pr.synthetic_fixture(seed=32)
    fold = pr.synthetic_fold(D['Y'], seed=32)
    a = _a(epochs=3)
    real_il = pr.fz.inter_loss
    def poisoned(*x, **k):
        t = real_il(*x, **k)
        return t * torch.tensor(float('inf')) if t.requires_grad else t
    monkeypatch.setattr(pr.fz, 'inter_loss', poisoned)
    with pytest.raises(pr.HardStop):
        pr.train_fold(D, fold, a, epochs=3)


def test_wallclock_cap_interrupts_safely():
    D = pr.synthetic_fixture(seed=33)
    fold = pr.synthetic_fold(D['Y'], seed=33)
    a = _a(epochs=500, patience=999)
    res = pr.train_fold(D, fold, a, epochs=500, max_wall_s=0.05)
    assert res['interrupted'] is not None
    assert 'wall_cap' in res['interrupted']
    # partial state returned, not raised — run is safe-interruptible
    assert 'macro_f1' in res


def test_pooled_macrof1_counts_correct():
    """Pooled = sum TP/FP/FN across folds, then class F1, then mean —
    not the mean of per-fold F1s."""
    mk = lambda tp, fp, fn, sup: {'macro_f1': 0.5, 'per_class_tp': tp,
        'per_class_fp': fp, 'per_class_fn': fn, 'per_class_support': sup,
        'per_class_f1': [0.5] * len(tp)}
    f1 = mk([10, 0], [2, 0], [5, 0], [15, 0])
    f2 = mk([4, 1], [1, 1], [1, 0], [5, 1])
    agg = ev.aggregate_fold_results([f1, f2])
    # class0 pooled: tp=14 fp=3 fn=6 -> p=14/17 r=14/20 -> f1=0.7568
    # class1 pooled: tp=1 fp=1 fn=0 -> p=0.5 r=1.0 -> f1=0.6667
    expected = (2 * (14 / 17) * (14 / 20) / (14 / 17 + 14 / 20)
                + 2 * 0.5 * 1.0 / 1.5) / 2
    assert abs(agg['pooled_macro_f1'] - expected) < 1e-6
    assert agg['mean_of_fold_macro_f1'] == 0.5
    assert agg['pooled_counts']['tp'] == [14, 1]


def test_tampered_fold_manifest_rejected(tmp_path):
    z = np.load(os.path.join(FOLDS, 'fold0_seed1234_static.npz'))
    bad = str(tmp_path / 'fold0_seed1234_static.npz')
    np.savez(bad, pairs=z['pairs'][:100], y=z['y'][:100], logits=z['logits'][:100])
    import shutil
    d = str(tmp_path)
    for f in range(1, 5):
        shutil.copy(os.path.join(FOLDS, f'fold{f}_seed1234_static.npz'), d)
    with pytest.raises(pr.LockViolation):
        pr.validate_hashes(DATA, d)
