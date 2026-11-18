# -*- coding: utf-8 -*-
"""Fixtures for metric_contracts.py. Run:
    .venv-bank1/bin/python test_metric_contracts.py
All checks assert-raise; prints PASS summary.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_contracts import (
    release_top1_harmonic_k4, generalized_top1_harmonic,
    per_class_macro_f1, micro_f1, any_hit_rate, confusion_and_support)

HERE = Path(__file__).resolve().parent
AUTHOR_CALC = (HERE.parent / 'independent_v20_release_20261116'
               / 'pinned_release' / 'Calculate_Metrics.py')


def _author_calc():
    spec = importlib.util.spec_from_file_location('author_calc',
                                                  AUTHOR_CALC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.OptimizedMetricCalculator()


def _multihot(rows, K):
    y = np.zeros((len(rows), K), dtype=np.int64)
    for i, r in enumerate(rows):
        y[i, r] = 1
    return y


def test_author_metric_loads():
    c = _author_calc()
    out = c.compute_top1_metrics(
        [[1], [2], [3], [4]],
        [[0, .9, .1, 0, 0], [0, .9, .1, 0, 0],
         [0, 0, 0, .9, .1], [0, 0, 0, 0, .9]])
    assert out['top1_f1'] > 0, out


def test_k4_equivalence_vs_pinned_release():
    """Our K4 reimplementation must match the author function on
    inputs the author function can score (scalar truths 1-4, len-5
    and len-4 predictions)."""
    rng = np.random.default_rng(7)
    reals = [int(rng.integers(1, 5)) for _ in range(300)]
    preds5 = rng.random((300, 5))
    preds4 = rng.random((300, 4))
    calc = _author_calc()
    a5 = calc.compute_top1_metrics(reals, preds5)
    a4 = calc.compute_top1_metrics(reals, preds4)
    o5 = release_top1_harmonic_k4(reals, preds5)
    o4 = release_top1_harmonic_k4(reals, preds4)
    for k in ('top1_precision', 'top1_recall', 'top1_f1',
              'valid_samples', 'correct_predictions',
              'total_predictions'):
        assert abs(a5[k] - o5[k]) < 1e-12, (k, a5[k], o5[k])
        assert abs(a4[k] - o4[k]) < 1e-12, (k, a4[k], o4[k])
    # multi-hot truth path (author takes first positive index)
    reals_mh = [[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]]
    am = calc.compute_top1_metrics(reals_mh, preds4[:4])
    om = release_top1_harmonic_k4(reals_mh, preds4[:4])
    for k in ('top1_precision', 'top1_recall', 'top1_f1'):
        assert abs(am[k] - om[k]) < 1e-12


def test_k4_release_drops_type5_and_len6():
    """Verified bug surface: author metric scores 0 valid samples on
    v3.2-shaped inputs (type-5 truth, 6-channel predictions)."""
    reals = [[5] for _ in range(50)]               # scalar type-5
    reals += [[0, 0, 0, 0, 1] for _ in range(50)]  # multi-hot type-5
    preds6 = np.random.default_rng(0).random((100, 6)).tolist()
    out = release_top1_harmonic_k4(reals, preds6)
    assert out['total_predictions'] == 0
    assert out['top1_f1'] == 0.0


def test_perfect_and_worst():
    K = 5
    rows = [[k] for k in range(K)] * 20
    y = _multihot(rows, K)
    good = np.zeros((100, K))
    good[np.arange(100), y.argmax(1)] = 1.0
    assert generalized_top1_harmonic(y, good)['f1_harmonic'] == 1.0
    assert per_class_macro_f1(y, good)['macro_f1'] == 1.0
    bad = np.zeros((100, K))
    bad[np.arange(100), (y.argmax(1) + 1) % K] = 1.0
    assert generalized_top1_harmonic(y, bad)['f1_harmonic'] == 0.0
    assert per_class_macro_f1(y, bad)['macro_f1'] == 0.0


def test_class_never_predicted_and_rare():
    K = 5
    y = _multihot([[0]] * 95 + [[4]] * 5, K)
    s = np.zeros((100, K)); s[:, 0] = 1.0  # only class 0 predicted
    m = per_class_macro_f1(y, s)
    assert m['per_class_f1'][4] == 0.0  # never predicted -> 0, not NaN
    # class0: P=95/100, R=1 -> F1=0.97436; macro = that/K
    assert abs(m['macro_f1'] - 0.9743589743 / K) < 1e-9
    g = generalized_top1_harmonic(y, s)
    # recall only over classes present: (1.0 + 0.0)/2
    assert abs(g['recall_macro'] - 0.5) < 1e-12
    assert abs(g['precision_top1'] - 0.95) < 1e-12


def test_multilabel_conventions_differ():
    """A multi-label pair exposes the convention difference: pred on
    the second positive channel counts as wrong under first_positive,
    correct under any_hit."""
    y = np.array([[1, 0, 1, 0, 0]])
    s = np.array([[0.1, 0.1, 0.9, 0.0, 0.0]])  # argmax -> channel 2
    fp = generalized_top1_harmonic(y, s, 'first_positive')
    ah = generalized_top1_harmonic(y, s, 'any_hit')
    assert fp['precision_top1'] == 0.0   # first positive is channel 0
    assert ah['precision_top1'] == 1.0   # channel 2 is positive


def test_type5_only_coverage():
    """type-5 (channel 4) truth+pred must flow through generalized
    metrics — never dropped."""
    y = _multihot([[4]] * 40, 5)
    s = np.zeros((40, 5)); s[:, 4] = 1.0
    assert generalized_top1_harmonic(y, s)['f1_harmonic'] == 1.0
    c = confusion_and_support(y, s)
    assert c['support_multihot'][4] == 40
    assert sum(c['support_first_positive']) == 40


def test_harmonic_agg_ne_macro_f1():
    """Counterexample: harmonic-of-aggregates != macro-F1."""
    y = _multihot([[0]] * 10 + [[1]] * 10 + [[2]] * 10 +
                  [[3]] * 10 + [[4]] * 60, 5)
    s = np.zeros((100, 5))
    s[:90, 0] = 1.0      # first 90 predicted class 0
    s[90:, 4] = 1.0      # last 10 of class-4 block predicted 4
    g = generalized_top1_harmonic(y, s)
    m = per_class_macro_f1(y, s)
    assert abs(g['f1_harmonic'] - m['macro_f1']) > 1e-3
    # P(acc)=20/100=0.2; R=(1+0+0+0+1/6)/5=7/30 -> F1=0.21538
    assert abs(g['f1_harmonic'] - 0.2153846) < 1e-4
    # class0: tp10/pred90 -> P=1/9,R=1,F1=0.2; class4: 2/7 -> macro=(0.2+0.28571)/5
    assert abs(m['macro_f1'] - 0.0971428) < 1e-4


def test_sklearn_agreement():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(3)
    y = (rng.random((500, 5)) < 0.3).astype(int)
    y[~y.any(axis=1)] = [1, 0, 0, 0, 0]
    s = rng.random((500, 5))
    oh = np.zeros_like(y); oh[np.arange(500), s.argmax(1)] = 1
    sk = f1_score(y, oh, average='macro', zero_division=0)
    ours = per_class_macro_f1(y, s)['macro_f1']
    assert abs(sk - ours) < 1e-12


def test_empty_and_inconsistent():
    try:
        generalized_top1_harmonic(np.zeros((0, 5)), np.zeros((0, 5)))
        assert False
    except ValueError:
        pass
    out = release_top1_harmonic_k4([], [])
    assert out['top1_f1'] == 0.0


def test_micro_and_anyhit():
    y = _multihot([[0], [1, 2], [4]], 5)
    s = np.array([[.9, 0, 0, 0, 0], [0, .5, .6, 0, 0],
                  [0, 0, 0, 0, .9]])
    assert any_hit_rate(y, s) == 1.0  # argmax ch2 hits {1,2}
    # tp: ch0=1, ch2=1, ch4=1; fp: ch2=1; fn: ch1=1
    assert abs(micro_f1(y, s) - 2 * 1.5 / (1.5 + 2)) < 1e-12


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_')]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print('ALL', len(fns), 'METRIC FIXTURES PASS')
