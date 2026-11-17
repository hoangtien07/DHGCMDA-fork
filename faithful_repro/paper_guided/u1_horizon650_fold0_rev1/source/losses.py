# -*- coding: utf-8 -*-
"""S1 loss composition — thin declared wrapper over Task1 pure modules.

S1 optimizes ONLY the normalized weighted type loss on logits
(Task1 `metrics.type_loss`) with train-only effective-count weights
(Task1 `contracts.class_weights_train`, beta=0.99999, sum=C).

No existence loss, no contrastive loss, no reconstruction loss —
nothing else may inject gradients in this stage.
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import contracts as _c   # noqa: E402
import metrics as _m     # noqa: E402


def imported_source_hashes():
    """sha256 of the effective imported pure sources (manifest gate)."""
    return {
        'contracts.py': hashlib.sha256(
            open(_c.__file__, 'rb').read()).hexdigest(),
        'metrics.py': hashlib.sha256(
            open(_m.__file__, 'rb').read()).hexdigest(),
    }


def class_weights_train(y_multi, beta=0.99999):
    return _c.class_weights_train(y_multi, beta=beta)


def type_loss(logits, multi_hot, class_weights):
    return _m.type_loss(logits, multi_hot, class_weights)


def type_metrics(truth, scores):
    return _m.type_metrics(truth, scores)
