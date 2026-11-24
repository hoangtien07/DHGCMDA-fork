"""Rescore saved predictions per fold using the thesis strategy's conventions."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from prepare_v32 import ASSOCIATIONS, HERE, ROOT, RELEASE, RELEASE_TYPES, SOURCE_TYPES
from v32_runs import top1_five_types


def load_labels(dataset):
    if dataset == "v2.0_495m383D":
        rows = np.loadtxt(RELEASE / dataset / ASSOCIATIONS, delimiter=",", dtype=int) - 1
        labels = np.zeros((495, 383, 4), dtype=bool)
        labels[tuple(rows.T)] = True
        return labels
    if dataset == "v3.2_spld_paper":
        labels = np.load(ROOT / dataset / "Y_multilabel.npy", allow_pickle=False)
        indices = [SOURCE_TYPES.index(name) for name in RELEASE_TYPES]
        return labels[..., indices].astype(bool)
    raise ValueError(f"Unsupported dataset: {dataset}")


def type_metrics(truth, score):
    if truth.ndim != 2 or score.shape != truth.shape or not truth.any(axis=1).all():
        raise ValueError("Expected matching type columns for positive pairs")
    prediction = np.eye(truth.shape[1], dtype=bool)[np.argmax(score, axis=1)]
    hits = int((prediction & truth).sum())
    precision = hits / len(truth)
    recall = hits / int(truth.sum())
    f1 = 2 * hits / (len(truth) + int(truth.sum()))
    return {"tdrc_top1_precision": precision, "tdrc_top1_recall": recall, "tdrc_top1_f1": f1,
            "multilabel_macro_f1_top1_decoding": float(
                f1_score(truth, prediction, average="macro", zero_division=0))}


def score_run(run, dataset, completion):
    labels = load_labels(dataset)
    folds = []
    for fold in range(1, completion["folds"] + 1):
        with np.load(run / f"fold_{fold}.npz", allow_pickle=False) as data:
            pairs = data["test_pos"]
            truth = labels[pairs[:, 0], pairs[:, 1]]
            score = data["score_pos"]
            if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
                raise ValueError(f"Invalid probability columns in {run.name} fold {fold}")
            metrics = type_metrics(truth, score[:, 1:])
            # Use the uniform class probability as a fixed cutoff for the
            # archived mutually normalized scores. No validation scores exist
            # for the author-protocol runs.
            decoded = score[:, 1:] >= 1 / labels.shape[2]
            metrics["multilabel_macro_f1_uniform_cutoff"] = float(
                f1_score(truth, decoded, average="macro", zero_division=0))
            if labels.shape[2] == 5:
                metrics.update(top1_five_types(data["truth_pos"], score))
            folds.append(metrics)
    return {
        "folds": folds,
        "mean": {name: float(np.mean([row[name] for row in folds])) for name in folds[0]},
        "binary_author_evaluation": completion["binary"],
        "top1_author_evaluation": completion["top1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    options = parser.parse_args()
    output = {}
    prefixes = ("c1-", "c2-", "c3-", "v32-")
    for path in sorted(options.runs.glob("*/completion.json")):
        run = path.parent
        if not run.name.startswith(prefixes) or not run.name.endswith(f"-{options.device}"):
            continue
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        completion = json.loads(path.read_text(encoding="utf-8"))
        output[run.name] = score_run(run, config["args"]["dataset"], completion)
    if not output:
        raise RuntimeError("No completed calibration or v3.2 runs")
    destination = options.runs / f"rescored-{options.device}.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
