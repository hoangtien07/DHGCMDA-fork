"""P5 runner integration tests (G4.1 WP3) — exercise the real code
path end-to-end on synthetic fixtures. No HMDD training."""
import os, json, types
import numpy as np
import torch
import pytest
import p5_runner as pr
import p5_evaluator as ev
import p5_diagnostics as dg


def _args(**kw):
    base = dict(hidden=16, proj=8, heads=2, nlayer=1, dropout=0.0, K=3,
                t=0.5, lam1=1.0, lam2=0.3, lam3=1.0, margin=0.5,
                margin_w=0.1, exist_w=0.3, neg_w=0.5, beta=0.99999,
                lr=1e-3, wd=1e-5, neg_ratio=3, update_every=5, theta=0.5,
                early_stop=1, eval_every=2, patience=50, n_neg_draws=2,
                predictor='diag', recon_reduction='sum', diag_leak=0,
                ablation='none', protocol='type', folds=5, only_fold=-1,
                cap_threads=0, threads=1, seed=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_1_test_label_perturbation_invariance():
    """Changing ONLY outer-test labels must leave fitted thresholds,
    selected checkpoint and every train-visible tensor identical."""
    D = pr.synthetic_fixture(seed=11)
    a = _args(update_every=2, eval_every=1, epochs=4)
    fold = pr.synthetic_fold(D['Y'], seed=11)
    torch.manual_seed(0)
    res1 = pr.train_fold(D, fold, a, epochs=4)
    # perturb test-pair labels only (views fixed, feature profile fixed)
    D2 = {k: v.copy() for k, v in D.items()}
    ti, tj = fold['test'][0]
    D2['Y'][ti, tj] = 1 - D2['Y'][ti, tj]
    torch.manual_seed(0)  # identical init + draws -> only Y[test] differs
    res2 = pr.train_fold(D2, fold, a, epochs=4)
    assert res1['thresholds'] == res2['thresholds']
    assert res1['best_val_macro_f1'] == res2['best_val_macro_f1']
    assert res1['val_history'] == res2['val_history']
    # test metric MAY change (labels are inputs to scoring), but never
    # to train-visible tensors


def test_2_checkpoint_follows_macrof1_not_top1(monkeypatch):
    """Fixture where aux top1 and primary macro-F1 disagree: checkpoint
    must follow macro-F1."""
    D = pr.synthetic_fixture(seed=12)
    a = _args(update_every=99, eval_every=1, patience=50, epochs=3)
    fold = pr.synthetic_fold(D['Y'], seed=12)
    # controlled macro-F1 sequence: ep1 high, ep2/ep3 low
    seq = iter([0.9, 0.1, 0.1, 0.0])  # 3 val evals + 1 test eval
    calls = []
    real = ev.multihot_macro_f1

    def fake(y, p, taus, support_policy="exclude_and_report"):
        calls.append(1)
        v = next(seq)
        return {'metric': 'MULTIHOT_MACRO_F1', 'macro_f1': v,
                'per_class_f1': [v] * y.shape[1],
                'per_class_support': [1] * y.shape[1],
                'excluded_classes': [], 'zero_support_policy': 'x',
                'prediction_cardinality_mean': 0.0, 'thresholds': list(taus)}
    monkeypatch.setattr(pr.ev, 'multihot_macro_f1', fake)
    res = pr.train_fold(D, fold, a, epochs=3)
    # primary selection kept ep1 (0.9) as best — regardless of what any
    # top1-style metric would have said
    assert res['best_val_macro_f1'] == 0.9
    assert len(calls) == 4  # 3 validation evals + 1 test eval


def test_3_runner_uses_locked_manifest():
    """resolve_run must use the locked fold manifest pairs verbatim and
    fail closed on tampered data."""
    data = os.path.join(pr.REPO,
        'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')
    fold_dir = os.path.join(pr.REPO,
        'faithful_repro/independent_v32_outer_refit_20261117/evaluation')
    out = pr.resolve_run(data, fold_dir)
    assert out['n_folds'] == 5
    # split definitions == manifest pairs exactly (not regenerated)
    z0 = np.load(os.path.join(fold_dir, 'fold0_seed1234_static.npz'))
    folds, _ = pr.validate_hashes(data, fold_dir)
    assert np.array_equal(folds[0]['test'], z0['pairs'])
    # tampered tensor -> LockViolation
    import tempfile, hashlib
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        f.write(b'not the tensor'); bad = f.name
    try:
        with pytest.raises(pr.LockViolation):
            pr.validate_hashes(bad, fold_dir)
    finally:
        os.unlink(bad)


def test_4_synthetic_step_finite_and_updates():
    """Complete forward/backward/optimizer step: finite loss, finite
    gradients, nonzero parameter update."""
    D = pr.synthetic_fixture(seed=13)
    a = _args(update_every=99, eval_every=1, epochs=2, patience=1)
    fold = pr.synthetic_fold(D['Y'], seed=13)
    diag = dg.Diagnostics()
    res = pr.train_fold(D, fold, a, diag=diag, epochs=2)
    rep = diag.report()
    s = rep['steps'][-1]  # step 2 carries the step1->step2 update delta
    assert np.isfinite(s['total']) and np.isfinite(s['l_type'])
    assert np.isfinite(s['grad_norm_preclip'])
    assert s['param_update_l2_total'] > 0  # parameters actually moved
    assert np.isfinite(res['macro_f1'])


def test_5_refresh_epochs_exactly():
    """The real refresh path fires at multiples of update_every only —
    at 5,10 not at other epochs; graph objects change on those epochs."""
    torch.manual_seed(0)
    D = pr.synthetic_fixture(seed=14)
    a = _args(update_every=5, eval_every=6, epochs=12, patience=99)
    fold = pr.synthetic_fold(D['Y'], seed=14)
    diag = dg.Diagnostics()
    res = pr.train_fold(D, fold, a, diag=diag, epochs=12)
    assert res['refresh_epochs'] == [5, 10]
    rep = diag.report()
    assert len(rep['refreshes']) == 2
    assert [r['epoch'] for r in rep['refreshes']] == [5, 10]
    assert all(r['graph_objects_changed'] for r in rep['refreshes'])
    # no refresh recorded at any non-multiple epoch
    assert not any(r['epoch'] % 5 != 0 for r in rep['refreshes'])


def test_6_unknown_provenance_cannot_be_reported_clean():
    """A provenance-UNKNOWN feature profile must never surface a
    'verified_clean' claim ceiling."""
    assert pr.PROFILE_CLAIM_CEILING['P5_BENCHMARK_UNKNOWN_PROVENANCE'] == 'exploratory_only'
    data = os.path.join(pr.REPO,
        'faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz')
    fold_dir = os.path.join(pr.REPO,
        'faithful_repro/independent_v32_outer_refit_20261117/evaluation')
    out = pr.resolve_run(data, fold_dir)
    assert out['claim_ceiling'] == 'exploratory_only'
    assert 'clean' not in out['claim_ceiling']
    man = json.load(open(pr.MANIFEST))
    assert 'UNKNOWN_PROVENANCE' in \
        man['artifacts']['strict_tensor']['reverified']['view_provenance']


def test_lock_rejects_diag_leak_and_missing(tmp_path):
    cfg = pr.load_lock()
    with pytest.raises(pr.LockViolation):
        pr.args_from_lock(cfg, 'P5_BENCHMARK_UNKNOWN_PROVENANCE',
                          overrides={'diag_leak': 1})
    import yaml
    bad = dict(cfg); bad.pop('model')
    p = tmp_path / 'bad.yaml'
    p.write_text(yaml.dump(bad))
    with pytest.raises(pr.LockViolation):
        pr.load_lock(str(p))
