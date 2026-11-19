# -*- coding: utf-8 -*-
"""REV4.1 regression fixtures: must catch BOTH metric_bridge bugs.
Run: .venv-bank1/bin/python test_metric_bridge_regressions.py
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_bridge import conv_metrics


def test_multitype_anyhit_perfect():
    """Perfect predictor on multi-type pairs: anyhit must be 1.0.
    Buggy version compared class ids vs 0/1 matrix -> <1."""
    rng = np.random.default_rng(0)
    y = np.zeros((200, 5))
    # all multi-type rows
    for r in y:
        ks = rng.choice(5, 2, replace=False)
        r[ks] = 1
    s = np.zeros_like(y)
    # perfect top-1: predict a true type
    for i, r in enumerate(y):
        s[i, np.where(r > 0)[0][0]] = 1.0
    m = conv_metrics(y, s)
    assert m['n_multitype_pairs'] == 200
    assert m['multitype_anyhit'] == 1.0, m['multitype_anyhit']


def test_multitype_anyhit_wrong():
    """Predictor always wrong on multi-type: anyhit must be 0."""
    y = np.tile([1, 1, 0, 0, 0], (100, 1)).astype(float)
    s = np.tile([0, 0, 1, 0, 0], (100, 1)).astype(float)
    m = conv_metrics(y, s)
    assert m['multitype_anyhit'] == 0.0


def test_k4filtered_perfect():
    """4-class filtered convention, perfect predictor -> F1 = 1.
    Buggy version averaged recall over range(5) -> F1 < 1."""
    rng = np.random.default_rng(1)
    y = np.zeros((300, 5))
    ks = rng.integers(0, 4, 300)  # only types 0-3, never type-5
    y[np.arange(300), ks] = 1
    s = y.copy()
    m = conv_metrics(y, s)
    assert m['K4filtered_n'] == 300
    assert abs(m['K4filtered_F1'] - 1.0) < 1e-12, m['K4filtered_F1']


def test_k4filtered_drops_type5_carriers():
    y = np.array([[1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 1],  # carries type-5 -> dropped
                  [0, 0, 1, 0, 0]], dtype=float)
    s = y.copy()
    m = conv_metrics(y, s)
    assert m['K4filtered_n'] == 2
    assert m['K4filtered_F1'] == 1.0


def test_anyhit_rate_vs_f1_distinct():
    """anyhit_rate (accuracy of hitting any true type) and
    anyhit_F1 (harmonic with per-type recall) must both exist and
    not be conflated with multihot metrics."""
    y = np.array([[1, 1, 0, 0, 0]] * 10, dtype=float)
    s = np.array([[1, 0, 0, 0, 0]] * 10, dtype=float)
    m = conv_metrics(y, s)
    assert m['anyhit_rate'] == 1.0
    assert 0 < m['anyhit_F1'] <= 1.0
    assert m['multihot_macroF1'] < 1.0  # type-1 never predicted


if __name__ == '__main__':
    import traceback
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for f in fns:
        try:
            f()
            print('PASS', f.__name__)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    print('ALL', len(fns), 'BRIDGE REGRESSION FIXTURES PASS')
