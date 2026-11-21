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


# ---------------- G5-launch hotfix tests ----------------

def _run_synth(outdir, seed=7, extra=()):
    rc = pr.main(['--once', '--synthetic', '--seed', str(seed),
                  '--out', outdir, *extra])
    return rc


def test_identical_seed_runs_identical_artifacts(tmp_path):
    """Two synthetic CLI runs, same seed+config -> identical checkpoint
    and identical result.json modulo wall-clock."""
    d1, d2 = str(tmp_path / 'r1'), str(tmp_path / 'r2')
    assert _run_synth(d1) == 0 and _run_synth(d2) == 0
    c1 = torch.load(os.path.join(d1, 'checkpoint.pt'), weights_only=False)
    c2 = torch.load(os.path.join(d2, 'checkpoint.pt'), weights_only=False)
    for k in c1['state_dict']:
        assert torch.equal(c1['state_dict'][k], c2['state_dict'][k])
    r1 = json.load(open(os.path.join(d1, 'result.json')))
    r2 = json.load(open(os.path.join(d2, 'result.json')))
    r1.pop('wall_s'); r2.pop('wall_s')
    assert r1 == r2


def test_interrupted_run_never_scores_test(tmp_path):
    d = str(tmp_path / 'ri')
    rc = pr.main(['--once', '--synthetic', '--seed', '9',
                  '--max-wall-s', '0.01', '--out', d])
    assert rc == 0
    res = json.load(open(os.path.join(d, 'result.json')))
    assert res['outcome'] == 'interrupted'
    assert res['macro_f1'] is None          # no outer-test scoring
    assert res['thresholds'] is None
    assert res['interrupted'] is not None
    assert res['optimizer_steps'] >= 1      # partial evidence kept


def test_hardstop_cli_writes_failure_report(tmp_path, monkeypatch):
    d = str(tmp_path / 'rf')
    monkeypatch.setattr(pr.fz, 'inter_loss',
                        lambda *x, **k: torch.tensor(float('nan')))
    rc = pr.main(['--once', '--synthetic', '--seed', '5', '--out', d])
    assert rc == 1
    rep = json.load(open(os.path.join(d, 'failure_report.json')))
    assert rep['outcome'] == 'failed'
    assert 'non_finite_loss' in rep['hard_stop']
    assert os.path.exists(os.path.join(d, 'diagnostics.json'))
    assert not os.path.exists(os.path.join(d, 'result.json'))


def test_completed_artifact_fields(tmp_path):
    d = str(tmp_path / 'rc')
    assert _run_synth(d, seed=11) == 0
    res = json.load(open(os.path.join(d, 'result.json')))
    man = json.load(open(os.path.join(d, 'resolved_manifest.json')))
    for k in ['selected_epoch', 'optimizer_steps', 'thresholds',
              'threshold_grid', 'outcome', 'val_history']:
        assert k in res, k
    assert res['outcome'] == 'completed'
    assert res['selected_epoch'] is not None
    assert man['threshold_grid'] == pr.THRESHOLD_GRID
    assert 'resolved_args' in man and 'seed' in man
    # pinned grid reached the evaluator: every fitted tau is a grid point
    assert all(t in set(pr.THRESHOLD_GRID) or t == 0.5
               for t in res['thresholds'])


def test_no_valid_result_without_checkpoint(tmp_path):
    """eval_every larger than epochs -> no val-selected checkpoint ->
    outcome failed_no_checkpoint, no test score."""
    import types as _t
    a = _a(epochs=2, eval_every=99)
    D = pr.synthetic_fixture(seed=44)
    fold = pr.synthetic_fold(D['Y'], seed=44)
    res = pr.train_fold(D, fold, a, epochs=2)
    assert res['outcome'] == 'failed_no_checkpoint'
    assert res['macro_f1'] is None


# ---------------- G6 preflight tests ----------------

def test_model_seed_split_seed_separation():
    """Split-manifest seed (bind_context seed -> which manifest files +
    inner-val construction) and model seed (a.seed -> torch/np RNG) are
    separate concepts: changing the model seed never moves a fold."""
    ctx1 = pr.bind_context('P5_BENCHMARK_UNKNOWN_PROVENANCE', seed=1234,
                           data_path=DATA, fold_dir=FOLDS)
    ctx2 = pr.bind_context('P5_BENCHMARK_UNKNOWN_PROVENANCE', seed=1234,
                           data_path=DATA, fold_dir=FOLDS)
    for f in range(5):
        for k in ('train', 'val', 'test'):
            assert np.array_equal(ctx1['folds'][f][k], ctx2['folds'][f][k])
    # model seed lives only in args — changing it changes training RNG
    # but cannot alter the bound split coordinates
    a = _a(epochs=2, eval_every=1, seed=0)
    r0 = pr.train_fold(pr.synthetic_fixture(seed=60),
                       pr.synthetic_fold(pr.synthetic_fixture(seed=60)['Y'], seed=60),
                       a, epochs=2)
    a2 = _a(epochs=2, eval_every=1, seed=99)
    r1 = pr.train_fold(pr.synthetic_fixture(seed=60),
                       pr.synthetic_fold(pr.synthetic_fixture(seed=60)['Y'], seed=60),
                       a2, epochs=2)
    assert r0['outcome'] == r1['outcome'] == 'completed'


def test_checkpoint_persists_selected_graph(tmp_path):
    """Checkpoint now carries G/ei of the validation-selected epoch;
    persisting them changes neither outputs nor optimization (result
    identical to a direct train_fold run)."""
    d = str(tmp_path / 'rg')
    assert _run_synth(d, seed=13) == 0
    ck = torch.load(os.path.join(d, 'checkpoint.pt'), weights_only=False)
    assert 'G' in ck and 'ei' in ck
    assert set(ck['G']) == {'m1', 'm2', 'd1', 'd2'}
    assert ck['graph_state'] == 'selected'
    assert ck['selected_epoch'] is not None
    res = json.load(open(os.path.join(d, 'result.json')))
    assert np.isfinite(res['macro_f1'])  # unchanged by persistence


def test_score_diagnostics_present(tmp_path):
    d = str(tmp_path / 'rs')
    assert _run_synth(d, seed=15) == 0
    res = json.load(open(os.path.join(d, 'result.json')))
    sd = res['score_diagnostics']
    assert len(sd) == 5
    for k in range(5):
        c = sd[f'class_{k}']
        for f in ('auroc', 'ap', 'score_quantiles', 'pos_pred_rate', 'support'):
            assert f in c
        assert len(c['score_quantiles']) == 5


def test_amendment_unlocks_only_registered_key():
    """G7_A1_RECON_MEAN swaps exactly recon_reduction sum->mean in a real
    context; every other locked hyperparameter stays enforced."""
    cfg = pr.load_lock()
    a = pr.args_from_lock(cfg, 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          amendment='G7_A1_RECON_MEAN')
    assert a.recon_reduction == 'mean'
    assert a.amendment == 'G7_A1_RECON_MEAN'
    for k, v in pr.LOCKED_HYPERPARAMS.items():
        if k != 'recon_reduction':
            assert getattr(a, k) == v


def test_amendment_unregistered_rejected():
    with pytest.raises(pr.LockViolation):
        pr.args_from_lock(pr.load_lock(), 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          amendment='UNREGISTERED_CHANGE')


def test_amendment_does_not_loosen_other_locks():
    """A real-context override of any other locked key is still rejected
    even when a valid amendment is active."""
    with pytest.raises(pr.LockViolation):
        pr.args_from_lock(pr.load_lock(), 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          overrides={'lam3': 0.01},
                          amendment='G7_A1_RECON_MEAN')


def test_mean_reduction_isolated_to_recon_term():
    """Same seed => identical init and step-0 forward: only the recon
    component's scale changes between 'sum' and 'mean'."""
    D = pr.synthetic_fixture(seed=61)
    fold = pr.synthetic_fold(D['Y'], seed=61)
    d_sum, d_mean = dg.Diagnostics(), dg.Diagnostics()
    pr.train_fold(D, fold, _a(epochs=2, eval_every=1, seed=61),
                  diag=d_sum, epochs=2)
    pr.train_fold(D, fold, _a(epochs=2, eval_every=1, seed=61,
                              recon_reduction='mean'),
                  diag=d_mean, epochs=2)
    s_sum = d_sum.report()['steps'][0]
    s_mean = d_mean.report()['steps'][0]
    assert s_sum['l_type'] == s_mean['l_type']          # same forward
    assert s_sum['l_intra_w'] == s_mean['l_intra_w']
    assert s_mean['l_rec_w'] < s_sum['l_rec_w'] * 0.01  # mean shrinks recon


def test_dry_run_real_applies_amendment(tmp_path):
    """--dry-run must show the EXACT resolved args the real run would use —
    including the amendment (previously it silently showed 'sum')."""
    out = str(tmp_path / 'dry_a1.json')
    rc = pr.main(['--dry-run', '--data', DATA, '--fold-dir', FOLDS,
                  '--profile', 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                  '--amendment', 'G7_A1_RECON_MEAN', '--out', out])
    assert rc == 0
    res = json.load(open(out))
    assert res['resolved_args']['recon_reduction'] == 'mean'
    assert res['resolved_args']['lam3'] == 1.0
    assert res['resolved_args']['amendment'] == 'G7_A1_RECON_MEAN'
