# -*- coding: utf-8 -*-
"""Track N metric contract recheck — counterexamples for the
locked top1 firstpos-harmonic + multihot metrics."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'runs_g2c'))
from metrics_g2c import metrics_for  # noqa


def s(rows):
    return np.array(rows, dtype=float)


def t_perfect():
    # all 5 classes present -> perfect predictor must reach 1.0
    # under the locked formula (absent classes score 0 by design)
    y = np.eye(5).astype(int)
    p = np.eye(5) * 0.9
    m = metrics_for(y, p, [0.5] * 5)
    assert m['top1_firstpos_harmonic'] == 1.0, m
    assert m['macroF1'] == 1.0
    # partial support -> locked formula gives 0.75/0.8 (absent
    # class contributes 0 to per-class means) — contract locked
    y2 = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1]])
    p2 = np.zeros((3, 5))
    p2[0, 0] = p2[1, 1] = p2[1, 2] = p2[2, 4] = 0.9
    m2 = metrics_for(y2, p2, [0.5] * 5)
    assert np.isclose(m2['top1_firstpos_harmonic'], 0.75), m2


def t_wrong():
    y = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
    p = np.zeros((2, 5)); p[:, 4] = 0.9
    m = metrics_for(y, p, [0.5] * 5)
    assert m['top1_firstpos_harmonic'] == 0.0, m
    assert m['macroF1'] < 0.2


def t_multihot_firstpos_counts():
    # truth {1,3}: first-pos = 1. pred argmax = 3 (a real positive
    # but NOT first) -> firstpos miss even though multi-hot hit.
    y = np.array([[0, 1, 0, 1, 0]])
    p = s([[0, 0.4, 0, 0.9, 0]])
    m = metrics_for(y, p, [0.5] * 5)
    # multihot: pred {1,3} at tau .5 -> {3} (0.4<0.5) actually
    # argmax=3 -> top1 P=0,R partial
    assert m['top1_firstpos_harmonic'] == 0.0
    # and shows firstpos != multihot behaviour
    assert m['macroF1'] > 0


def t_first_vs_last():
    # same argmax prediction hits under last-pos convention but
    # misses under first-pos (distinguishes the two metrics)
    y = np.array([[0, 1, 0, 1, 0]])          # first=1, last=3
    p = s([[0, 0.1, 0, 0.9, 0]])             # argmax=3
    m = metrics_for(y, p, [0.5] * 5)
    assert m['top1_firstpos_harmonic'] == 0.0   # first-pos miss
    # truth first=1: only class-1 recall term -> both 0 -> 0


def t_support_and_cardinality():
    y = np.array([[1, 0, 0, 0, 0]] * 10)
    p = np.ones((10, 5)) * 0.6
    m = metrics_for(y, p, [0.5] * 5)
    assert abs(m['pred_cardinality_mean'] - 5.0) < 1e-9
    assert m['per_type'][0]['support'] == 10


if __name__ == '__main__':
    for f in (t_perfect, t_wrong, t_multihot_firstpos_counts,
              t_first_vs_last, t_support_and_cardinality):
        f(); print(f.__name__, 'PASS')
    print('METRIC CONTRACT RECHECK: ALL PASS')
