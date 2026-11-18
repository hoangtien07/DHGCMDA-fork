# -*- coding: utf-8 -*-
"""Metric contracts for the DHGCMDA reproduction evidence program.

Pure functions, no training dependencies. Each metric has a distinct
name; nothing is aliased. Definitions:

- ``release_top1_harmonic_k4``: line-faithful reimplementation of the
  author-released ``OptimizedMetricCalculator.compute_top1_metrics``
  (pinned_release/Calculate_Metrics.py L263-378, main@06f9ba3).
  Hardcoded to 4 types by construction: scalar truths outside {1,2,3,4}
  and predictions not of length 4 or 5 are SKIPPED, which silently
  drops every v3.2 type-5 truth and every 6-channel prediction.
- ``generalized_top1_harmonic``: same formula family (harmonic mean of
  top-1 accuracy and mean per-class recall) generalized to any K and
  any declared multi-label truth convention. NOT proven author-
  identical for K!=4 or for conventions other than 'first_positive'.
- ``per_class_macro_f1``: internal U1 primary — argmax-onehot vs
  multi-hot, per-class F1 averaged over all K channels (matches
  sklearn f1_score(average='macro') on onehot-vs-multihot rows).
- ``micro_f1``, ``any_hit_rate``, ``confusion_and_support``: diagnostics.

Truth conventions for multi-hot rows (predeclared):
- 'first_positive': true type = index of the FIRST positive channel
  (author code: ``real_type_indices[0]``).
- 'any_hit': the row is correct iff the argmax channel is positive in
  truth; per-class recall uses class support as denominator.
"""
import numpy as np

# Author scalar->type-index map, K=4 (tissue/type-5 absent by design).
_AUTHOR_SCALAR_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


def release_top1_harmonic_k4(real_scores_list, predict_scores_list):
    """Author-released Top-1 metric, 4-type hardcode. Returns dict."""
    if len(real_scores_list) == 0 or len(predict_scores_list) == 0:
        return {'top1_precision': 0.0, 'top1_recall': 0.0,
                'top1_f1': 0.0, 'valid_samples': 0,
                'correct_predictions': 0, 'total_predictions': 0,
                'n_dropped': 0}
    correct = total = valid = dropped = 0
    type_correct = np.zeros(4)
    type_total = np.zeros(4)
    type_predicted = np.zeros(4)
    for real_score, pred_score in zip(real_scores_list,
                                      predict_scores_list):
        try:
            ra = np.asarray(real_score).ravel()
            pa = np.asarray(pred_score).ravel()
            if len(ra) == 0 or len(pa) == 0:
                dropped += 1
                continue
            rv = ra[0] if len(ra) == 1 else ra
            if isinstance(rv, np.ndarray):
                if rv.sum() == 0:
                    dropped += 1
                    continue
                idx = np.where(rv > 0)[0]
                if len(idx) == 0 or idx[0] > 3:
                    dropped += 1
                    continue
                true_type = int(idx[0])
            else:
                if rv == 0 or rv not in _AUTHOR_SCALAR_MAP:
                    dropped += 1
                    continue
                true_type = _AUTHOR_SCALAR_MAP[int(rv)]
            if len(pa) == 5:
                ts = pa[1:5]
            elif len(pa) == 4:
                ts = pa[:4]
            else:
                dropped += 1
                continue
            pred = int(np.argmax(ts))
            valid += 1
            total += 1
            type_total[true_type] += 1
            type_predicted[pred] += 1
            if pred == true_type:
                correct += 1
                type_correct[true_type] += 1
        except Exception:
            dropped += 1
            continue
    p = correct / total if total else 0.0
    recalls = [type_correct[k] / type_total[k]
               for k in range(4) if type_total[k] > 0]
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'top1_precision': float(p), 'top1_recall': float(r),
            'top1_f1': float(f1), 'valid_samples': valid,
            'correct_predictions': correct,
            'total_predictions': total, 'n_dropped': dropped,
            'type_correct': type_correct.tolist(),
            'type_total': type_total.tolist(),
            'type_predicted': type_predicted.tolist()}


def generalized_top1_harmonic(truth_multihot, scores,
                              truth_convention='first_positive'):
    """Harmonic-of-aggregates Top-1 for any K and declared convention.

    truth_multihot: (n,K) 0/1. scores: (n,K) finite.
    convention 'first_positive': true type = first positive channel
        (matches author scalar semantics when K=4).
    convention 'any_hit': correct iff argmax channel positive; class
        recall_k = #(pred==k & y_k=1) / support_k.
    """
    y = (np.asarray(truth_multihot) != 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 2 or y.shape != s.shape or y.shape[0] == 0:
        raise ValueError('shape mismatch or empty input')
    if not np.isfinite(s).all():
        raise ValueError('non-finite scores')
    n, K = y.shape
    pred = s.argmax(axis=1)
    if truth_convention == 'first_positive':
        true = y.argmax(axis=1)
        hit = (pred == true)
        totals = np.bincount(true, minlength=K).astype(np.float64)
        corrs = np.bincount(true[hit], minlength=K).astype(np.float64)
        recalls = [corrs[k] / totals[k] for k in range(K)
                   if totals[k] > 0]
    elif truth_convention == 'any_hit':
        hit = y[np.arange(n), pred] > 0
        totals = y.sum(axis=0).astype(np.float64)
        corrs = np.array([float(((pred == k) & (y[:, k] > 0)).sum())
                          for k in range(K)])
        recalls = [corrs[k] / totals[k] for k in range(K)
                   if totals[k] > 0]
    else:
        raise ValueError('unknown convention %r' % truth_convention)
    p = float(hit.mean())
    r = float(np.mean(recalls)) if recalls else 0.0
    return {'precision_top1': p, 'recall_macro': r,
            'f1_harmonic': float(2 * p * r / (p + r)) if (p + r)
            else 0.0,
            'n': n, 'K': K, 'convention': truth_convention,
            'per_class_recall': recalls,
            'type_predicted': np.bincount(pred, minlength=K).tolist()}


def per_class_macro_f1(truth_multihot, scores):
    """Internal U1 primary: argmax-onehot vs multi-hot macro F1."""
    y = (np.asarray(truth_multihot) != 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 2 or y.shape != s.shape or y.shape[0] == 0:
        raise ValueError('shape mismatch or empty input')
    n, K = y.shape
    pred = s.argmax(axis=1)
    oh = np.zeros_like(y)
    oh[np.arange(n), pred] = 1
    tp = (oh & y.astype(bool)).sum(axis=0).astype(np.float64)
    fp = (oh.astype(bool) & ~y.astype(bool)).sum(axis=0)
    fn = (~oh.astype(bool) & y.astype(bool)).sum(axis=0)
    prec = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(K), where=(tp + fn) > 0)
    f1c = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K),
                    where=(prec + rec) > 0)
    return {'macro_f1': float(f1c.mean()),
            'per_class_precision': prec.tolist(),
            'per_class_recall': rec.tolist(),
            'per_class_f1': f1c.tolist(),
            'support': y.sum(axis=0).tolist(),
            'predicted': np.bincount(pred, minlength=K).tolist()}


def micro_f1(truth_multihot, scores):
    y = (np.asarray(truth_multihot) != 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n, K = y.shape
    pred = s.argmax(axis=1)
    oh = np.zeros_like(y)
    oh[np.arange(n), pred] = 1
    tp = float((oh & y.astype(bool)).sum())
    fp = float((oh.astype(bool) & ~y.astype(bool)).sum())
    fn = float((~oh.astype(bool) & y.astype(bool)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def any_hit_rate(truth_multihot, scores):
    y = (np.asarray(truth_multihot) != 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pred = s.argmax(axis=1)
    return float((y[np.arange(y.shape[0]), pred] > 0).mean())


def confusion_and_support(truth_multihot, scores):
    """Full K x K confusion (argmax pred x first-positive truth) +
    support per channel — guarantees no type is silently dropped."""
    y = (np.asarray(truth_multihot) != 0).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n, K = y.shape
    pred = s.argmax(axis=1)
    true = y.argmax(axis=1)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return {'confusion_first_positive': cm.tolist(),
            'support_multihot': y.sum(axis=0).tolist(),
            'support_first_positive': np.bincount(
                true, minlength=K).tolist(),
            'n': n, 'K': K}
