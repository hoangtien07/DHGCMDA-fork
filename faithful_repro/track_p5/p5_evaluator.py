"""P5 primary evaluator — MULTIHOT_MACRO_F1 + auxiliary metrics.

Contract (locked in P5_CONFIG_LOCK.yaml):
- thresholds fitted on INNER-VALIDATION predictions only
- checkpoint selected by primary metric, never top1
- outer test evaluated exactly once after freeze
- zero-support policy: classes with no positives in the eval set are
  EXCLUDED from the macro mean and reported explicitly
- pooled and mean-of-fold aggregation reported separately
- limitation: softmax+multi-hot CE is an interpretation of the type
  head, NOT independent-Bernoulli multilabel modeling
"""
import numpy as np

PRIMARY = "MULTIHOT_MACRO_F1"
ZERO_SUPPORT_POLICY = "exclude_and_report"  # declared, not assumed


def per_class_thresholds(p_val, y_val, grid=None):
    """Fit one threshold per class maximizing that class's F1 on
    VALIDATION predictions only. Returns (taus, val_f1_per_class)."""
    p_val = np.asarray(p_val); y_val = np.asarray(y_val)
    C = y_val.shape[1]
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    taus = np.zeros(C); f1s = np.zeros(C)
    for k in range(C):
        best_t, best_f = grid[0], -1.0
        col = y_val[:, k]
        if col.sum() == 0:  # zero val support -> keep mid, flag
            taus[k] = 0.5; f1s[k] = np.nan; continue
        for t in grid:
            pred = p_val[:, k] >= t
            tp = int((pred & (col == 1)).sum()); fp = int((pred & (col == 0)).sum())
            fn = int((~pred & (col == 1)).sum())
            f = 2 * tp / max(2 * tp + fp + fn, 1)
            if f > best_f:
                best_f, best_t = f, t
        taus[k] = best_t; f1s[k] = best_f
    return taus, f1s


def multihot_macro_f1(y, p, taus, support_policy=ZERO_SUPPORT_POLICY):
    """Primary metric. y,p: (n,C); taus: (C,).
    support_policy 'exclude_and_report': classes with support 0 in y are
    excluded from the macro mean and listed in 'excluded_classes'."""
    y = np.asarray(y); p = np.asarray(p); taus = np.asarray(taus)
    yhat = (p >= taus).astype(int)
    per, support = [], []
    excluded = []
    for k in range(y.shape[1]):
        sup = int(y[:, k].sum())
        support.append(sup)
        if sup == 0 and support_policy == "exclude_and_report":
            excluded.append(k); per.append(np.nan); continue
        tp = int(((yhat[:, k] == 1) & (y[:, k] == 1)).sum())
        fp = int(((yhat[:, k] == 1) & (y[:, k] == 0)).sum())
        fn = int(((yhat[:, k] == 0) & (y[:, k] == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        per.append(2 * prec * rec / max(prec + rec, 1e-12))
    valid = [v for v, s in zip(per, support) if s > 0 or support_policy != "exclude_and_report"]
    macro = float(np.mean([v for v in per if not np.isnan(v)])) if any(
        not np.isnan(v) for v in per) else 0.0
    card = float(yhat.sum(1).mean())
    return {
        "metric": PRIMARY, "macro_f1": macro,
        "per_class_f1": [None if np.isnan(v) else float(v) for v in per],
        "per_class_support": support,
        "per_class_tp": tpfpfn(y, yhat, 'tp'),
        "per_class_fp": tpfpfn(y, yhat, 'fp'),
        "per_class_fn": tpfpfn(y, yhat, 'fn'),
        "excluded_classes": excluded,
        "zero_support_policy": support_policy,
        "prediction_cardinality_mean": card,
        "thresholds": [float(t) for t in taus],
    }


def tpfpfn(y, yhat, kind):
    y = np.asarray(y); yhat = np.asarray(yhat)
    if kind == 'tp':
        c = (yhat == 1) & (y == 1)
    elif kind == 'fp':
        c = (yhat == 1) & (y == 0)
    else:
        c = (yhat == 0) & (y == 1)
    return [int(v) for v in c.sum(0)]


def top1_anyhit_harmonic(Y, P):
    """Auxiliary TOP1_ANYHIT_HARMONIC_K5 (released top1 convention)."""
    import faithful as fz
    return fz.top1(np.asarray(Y), np.asarray(P))


def top1_firstpos_harmonic(Y, P):
    """Auxiliary TOP1_SCALAR_FIRSTPOS_HARMONIC_K5:
    scalar correctness = argmax P == first-positive type.
    P = frac first-pos correct; R = mean per-class (pred==k | true k)."""
    Y = np.asarray(Y); P = np.asarray(P)
    pred = P.argmax(1)
    firstpos = Y.argmax(1) if (Y.sum(1) > 0).all() else np.array(
        [np.argmax(r) if r.sum() else -1 for r in Y])
    Pp = float((pred == firstpos).mean())
    recs = []
    for k in range(Y.shape[1]):
        m = Y[:, k] > 0
        if m.any():
            recs.append(float((pred[m] == k).mean()))
    R = float(np.mean(recs))
    F1 = 2 * Pp * R / (Pp + R) if Pp + R > 0 else 0.0
    return {"top1_firstpos_precision": Pp, "top1_firstpos_recall": R,
            "top1_firstpos_f1": float(F1), "n": int(len(pred))}


def aggregate_fold_results(fold_metrics):
    """Report pooled and mean-of-fold aggregation SEPARATELY.

    Pooled macro-F1 is computed CORRECTLY: TP/FP/FN are summed across
    folds per class, then per-class F1 is computed from the pooled
    counts, then averaged over supported classes (zero pooled support
    excluded, per declared policy). Mean-of-fold is the plain average
    of per-fold macro-F1 — a different number, kept alongside."""
    primary = np.array([f["macro_f1"] for f in fold_metrics])
    C = len(fold_metrics[0]["per_class_tp"])
    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C); sup = np.zeros(C)
    for f in fold_metrics:
        tp += np.array(f["per_class_tp"]); fp += np.array(f["per_class_fp"])
        fn += np.array(f["per_class_fn"]); sup += np.array(f["per_class_support"])
    pooled_f1 = []
    for k in range(C):
        if sup[k] == 0:
            pooled_f1.append(np.nan); continue
        prec = tp[k] / max(tp[k] + fp[k], 1); rec = tp[k] / max(tp[k] + fn[k], 1)
        pooled_f1.append(2 * prec * rec / max(prec + rec, 1e-12))
    pooled_valid = [v for v in pooled_f1 if not np.isnan(v)]
    return {
        "mean_of_fold_macro_f1": float(primary.mean()),
        "std_fold_macro_f1": float(primary.std()),
        "pooled_macro_f1": float(np.mean(pooled_valid)) if pooled_valid else 0.0,
        "pooled_per_class_f1": [None if np.isnan(v) else float(v) for v in pooled_f1],
        "pooled_counts": {"tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(),
                          "support": sup.tolist()},
        "aggregation_declared": {
            "primary_reported": "mean_of_fold_macro_f1",
            "pooled_variant": "TP/FP/FN summed across folds -> class F1 -> macro mean",
            "pooled_also_reported": True},
    }


INTERPRETATION_LIMITATION = (
    "Type head uses softmax(logits/T) with multi-hot weighted CE — a "
    "P5 interpretation of the released SimplifiedTypePredictor. It is "
    "NOT independent-Bernoulli multilabel modeling (sigmoid+BCE branch "
    "is documented but not executed).")
