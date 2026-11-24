"""Run the four Stage 4 baselines against the same frozen nested pair splits."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from clean_protocol import DATASETS, RUNS, fit_views, load_frozen_fold, load_module
from prepare_v32 import ROOT, sha
from score_predictions import type_metrics

METHODS = ("prior", "logistic", "tdrc", "spld")


def take(array, pairs):
    return array[tuple(pairs.T)]


def f1_threshold(labels, scores):
    if not np.asarray(labels).any():
        return float(np.nextafter(scores.max(), np.inf))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def fit_tensor_baseline(method, views, split, max_iter=None):
    import tensorly as tl

    original = tl.tenalg.khatri_rao

    def array_khatri_rao(matrices, *args, **kwargs):
        return original([np.asarray(matrix) for matrix in matrices], *args, **kwargs)

    tl.tenalg.khatri_rao = array_khatri_rao
    try:
        labels = views["masked_labels"].astype(np.float64)
        mi = np.asmatrix(views["mirna_functional"])
        disease = np.asmatrix(views["disease_semantic"])
        if method == "tdrc":
            source = ROOT / "baselines" / "TDRC" / "method.py"
            model = load_module(source, "tdrc_model").Model()
            return model()(labels, disease, mi, max_iter=max_iter or 500)

        source = ROOT / "forensics" / "spld_recovered" / "SPLDHyperAWNTF" / "method"
        old_modules = {name: sys.modules.get(name) for name in ("ConstructHW", "hypergraph_utils")}
        try:
            utils = load_module(source / "hypergraph_utils.py", "spld_hypergraph")
            # NumPy 1.26 rejects comparing an incidence matrix to an empty list.
            concat = inspect.getsource(utils.hyperedge_concat).replace(
                "h != []", "(not isinstance(h, list) or len(h) > 0)")
            exec(compile(concat, str(source / "hypergraph_utils.py"), "exec"), vars(utils))
            sys.modules["hypergraph_utils"] = utils
            sys.modules["ConstructHW"] = load_module(source / "ConstructHW.py", "spld_construct")
            model = load_module(source / "model.py", "spld_model").Hyper_Model()
        finally:
            for name, original_module in old_modules.items():
                if original_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original_module
        weights = np.zeros_like(labels)
        fit_pairs = np.concatenate((split["fit_pos"], split["fit_neg"]))
        weights[tuple(fit_pairs.T)] = 1
        return model()(labels, mi, disease, weights, max_iter=max_iter or 2000)
    finally:
        tl.tenalg.khatri_rao = original


def predict(method, views, split, pairs, seed):
    if method in {"tdrc", "spld"}:
        return take(fit_tensor_baseline(method, views, split), pairs)
    track = ROOT / "faithful_repro" / "track_n" / "baselines.py"
    old_import_path = sys.path[:]
    try:
        baseline = load_module(track, "track_n_baselines")
    finally:
        sys.path[:] = old_import_path
    masked = views["masked_labels"]
    if method == "prior":
        fit_pos = split["fit_pos"]
        return baseline.prior_scores(fit_pos, take(masked, fit_pos), pairs,
                                     *masked.shape[:2])["product_laplace1"]

    fit = np.concatenate((split["fit_pos"], split["fit_neg"]))
    truth = take(masked, fit).astype(np.int8)
    counts = baseline.fit_counts(masked)
    sims = (views["mirna_functional"], views["disease_semantic"])
    fit_features = baseline.pair_features(*counts, fit, *sims, y_own=truth)
    predictions = baseline.pair_features(*counts, pairs, *sims)
    return baseline.logistic_scores(truth, fit_features, predictions, T=masked.shape[-1], seed=seed)


def evaluate(labels, split, scores, existence=None, threshold_on_test=False):
    val = np.concatenate((split["val_pos"], split["val_neg"]))
    test = np.concatenate((split["test_pos"], split["test_neg"]))
    val_scores, test_scores = np.split(scores, [len(val)])
    val_truth, test_truth = take(labels, val), take(labels, test)
    val_exists, test_exists = val_truth.any(axis=1), test_truth.any(axis=1)
    binary = scores.max(axis=1) if existence is None else existence
    val_binary, test_binary = np.split(binary, [len(val)])
    binary_threshold = (f1_threshold(test_exists, test_binary) if threshold_on_test
                        else f1_threshold(val_exists, val_binary))
    type_thresholds = [f1_threshold(val_truth[:, k], val_scores[:, k]) for k in range(labels.shape[-1])]
    metrics = type_metrics(test_truth[test_exists], test_scores[test_exists])
    val_metrics = type_metrics(val_truth[val_exists], val_scores[val_exists])
    metrics.update({
        "binary_auc": float(roc_auc_score(test_exists, test_binary)),
        "binary_aupr": float(average_precision_score(test_exists, test_binary)),
        "binary_f1": float(f1_score(test_exists, test_binary >= binary_threshold)),
        "multilabel_macro_f1": float(
            f1_score(test_truth, test_scores >= type_thresholds, average="macro", zero_division=0)),
    })
    return {"inner_validation": val_metrics, "outer_test": metrics, "binary_threshold": binary_threshold,
            "type_thresholds": type_thresholds, "binary_threshold_source": "test" if threshold_on_test else "inner_validation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    options = parser.parse_args()
    labels, semantic, split, split_hash = load_frozen_fold(options.dataset, options.seed, options.fold)
    views = fit_views(labels, split["fit_pos"], semantic)
    ordered_pairs = np.concatenate([split[name] for name in ("val_pos", "val_neg", "test_pos", "test_neg")])
    scores = predict(options.method, views, split, ordered_pairs, options.seed)
    if not np.isfinite(scores).all():
        raise RuntimeError("Baseline produced non-finite scores")
    result = evaluate(labels, split, scores)
    output = RUNS / "clean_baselines" / f"{options.dataset}-{options.seed}-fold{options.fold}-{options.method}"
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scores.npz", pairs=ordered_pairs, scores=scores)
    record = {"run": vars(options), "split_sha256": split_hash, "scores_sha256": sha(output / "scores.npz"), **result}
    (output / "completion.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
