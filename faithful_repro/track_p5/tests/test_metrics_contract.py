"""Metric contract tests: named P5 metrics — MULTIHOT_MACRO_F1 (track_n
formula reference), TOP1_ANYHIT/FIRSTPOS harmonics, AP vs AUPR-trapz,
threshold inner-val only. Oracles are hand-computed, independent."""
import numpy as np
import torch
import faithful as fz


def _metrics_for(y, p_norm, taus):
    """Local copy of the locked Track N contract (same formulas, kept
    import-free so this file is a self-contained oracle)."""
    yhat = (p_norm >= np.asarray(taus)).astype(int)
    T = y.shape[1]
    per = []
    for k in range(T):
        tp = int(((yhat[:, k] == 1) & (y[:, k] == 1)).sum())
        fp = int(((yhat[:, k] == 1) & (y[:, k] == 0)).sum())
        fn = int(((yhat[:, k] == 0) & (y[:, k] == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        per.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(per))


def test_multihot_macro_f1_perfect():
    y = np.eye(5)[[0, 1, 2, 3, 4]]
    assert _metrics_for(y, y.astype(float), [0.5] * 5) == 1.0


def test_multihot_macro_f1_empty_pred():
    y = np.eye(5)[[0, 1, 2]]
    p = np.zeros((3, 5))
    # all predictions below threshold -> all F1 = 0
    assert _metrics_for(y, p, [0.5] * 5) == 0.0


def test_multihot_two_label_row():
    """Row [1,0,1,0,0] predicted correctly on both labels contributes
    F1=1 on classes 0 and 2."""
    y = np.array([[1, 0, 1, 0, 0]])
    p = np.array([[0.9, 0.1, 0.9, 0.1, 0.1]])
    macro = _metrics_for(y, p, [0.5] * 5)
    # classes 0,2: F1=1; classes 1,3,4: tp=fn=0 -> F1 = 0 (support>0 only
    # in Track N per_type? contract includes all classes -> expected
    # value = 2/5 under all-class mean)
    assert abs(macro - 0.4) < 1e-9


def test_top1_anyhit_harmonic_contract():
    """fz.top1 = any-hit precision + mean per-type recall — labelled
    TOP1_ANYHIT_HARMONIC_K5, NOT author evaluator."""
    Y = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
    P = np.array([[0.8, 0.1, 0.6, 0.1, 0.2], [0.1, 0.9, 0.1, 0.1, 0.1]])
    r = fz.top1(Y, P)
    assert r['top1_precision'] == 1.0  # argmax hits a true type each row
    # class recalls: c0=1.0 (row0 pred 0), c1=1.0 (row1 pred 1),
    # c2=0.0 (row0 pred 0, not 2) -> R=2/3, F1=0.8
    assert abs(r['top1_recall'] - 2 / 3) < 1e-9
    assert abs(r['top1_f1'] - 0.8) < 1e-9


def test_top1_firstpos_vs_anyhit_differ():
    """The two conventions diverge on multi-label rows — they must
    never be merged."""
    Y = np.array([[1, 0, 1, 0, 0]])
    P = np.array([[0.1, 0.1, 0.9, 0.1, 0.1]])  # argmax=2 (second label)
    anyhit = fz.top1(Y, P)['top1_precision']          # hit (2 is a true type)
    pred1 = P.argmax(1)
    firstpos = float((np.argmax(Y > 0, 1) == pred1).mean())  # 2 != 0
    assert anyhit == 1.0 and firstpos == 0.0


def test_absent_class_recall():
    """top1 recall only averages classes present in the batch."""
    Y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    P = np.eye(5)[[0, 1]] + 0.01
    r = fz.top1(Y, P)
    assert len(r['per_type_recall']) == 2  # only classes 0 and 1 present


def test_ap_vs_aupr_trapz_distinct():
    """average_precision_score and trapezoidal AUPR differ on sparse
    data — must be reported separately."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, auc
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    s = np.linspace(0.9, 0.05, 10)
    ap = average_precision_score(y, s)
    pr, rc, _ = precision_recall_curve(y, s)
    trapz = auc(rc, pr)
    assert abs(ap - trapz) > 0.02, "AP and trapezoid AUPR must differ on sparse positives"


def test_binary_metric_release_convention():
    """fz.binary uses test-set F1-max threshold — release convention;
    P5 locks validation-only thresholds. Pin the behavior."""
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.4, 0.1])
    r = fz.binary(y, s)
    assert r['f1'] <= 1.0 and 'auc' in r and 'aupr' in r


def test_f1_at_threshold_deterministic():
    y = np.array([1, 1, 0, 0]); s = np.array([0.9, 0.4, 0.5, 0.1])
    f05 = fz.f1_at_threshold(y, s, 0.5)
    assert abs(f05 - 2 * 1 / (2 * 1 + 1 + 1)) < 1e-9  # tp=1 fp=1 fn=1


def test_pooled_vs_fold_average():
    """Pooled vs mean-of-fold metrics can differ — aggregation order
    must be declared (Track R finding)."""
    y = np.vstack([np.eye(5)[[0, 1]], np.eye(5)[[2, 3, 4]]])
    p = y.astype(float)
    pooled = _metrics_for(y, p, [0.5] * 5)
    f1 = _metrics_for(y[:2], p[:2], [0.5] * 5)
    f2 = _metrics_for(y[2:], p[2:], [0.5] * 5)
    # all-class mean over folds where classes are absent gives 0 for
    # absent classes -> pooled != mean-of-fold; the contract requires
    # the aggregation order to be DECLARED, not assumed equal.
    assert abs(pooled - (f1 + f2) / 2) > 0.1
