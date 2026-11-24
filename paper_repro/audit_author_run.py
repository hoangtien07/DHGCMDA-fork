"""Verify logged folds and replay the published evaluator from saved v2.0 predictions."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Replay only the evaluator on CPU so a training run can keep using the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from prepare_v32 import RELEASE
from score_predictions import load_labels, type_metrics

NUMBERS = ("auc", "aupr", "f1", "accuracy", "recall", "specificity", "precision")
FOLD_START = re.compile(r"^=== Cross Validation Fold (\d+)/(\d+) ===$", re.MULTILINE)
FOLD_DONE = re.compile(r"^Fold (\d+) completed - AUC: ([.\d]+), Top-1 F1: ([.\d]+)$", re.MULTILINE)
EPOCH = re.compile(r"^Epoch (\d+), Total Loss:", re.MULTILINE)


def logged_folds(log, planned_folds, planned_epochs):
    starts = list(FOLD_START.finditer(log))
    verified = []
    for index, start in enumerate(starts):
        fold, total = map(int, start.groups())
        if fold != index + 1 or total != planned_folds or fold > total:
            raise ValueError(f"Unexpected fold header {fold}/{total}")
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log)
        section = log[start.end():end]
        complete = FOLD_DONE.search(section)
        if complete is None and index == len(starts) - 1:
            break
        if complete is None or int(complete[1]) != fold:
            raise ValueError(f"Fold {fold} lacks a matching completion line")
        reported_epochs = [int(value) for value in EPOCH.findall(section)]
        # The pinned code logs epoch 1 and multiples of 50. Its 650-epoch
        # runs expose the terminal epoch directly; short smoke runs do not.
        expected_epochs = [1, *range(50, planned_epochs + 1, 50)]
        if reported_epochs != expected_epochs or "NaN detected" in section:
            raise ValueError(f"Fold {fold} has incomplete or non-finite epoch evidence")
        verified.append((fold, reported_epochs[-1], complete[2], complete[3]))
    return verified


def verify(run, log_text):
    with (run / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)["args"]
    if config["dataset"] != "v2.0_495m383D":
        raise ValueError("This audit replays the v2.0 author evaluator")
    sys.path.insert(0, str(RELEASE))
    from main_experiments_hetero1 import evaluate_optimized_with_comprehensive_metrics

    labels = load_labels(config["dataset"])
    evidence = {}
    seen = set()
    binary_rows, cv_type_rows, top1_rows = [], [], []
    for fold, last_epoch, auc_logged, top1_logged in logged_folds(
        log_text, config["validation"], config["epoch"]
    ):
        with np.load(run / f"fold_{fold}.npz", allow_pickle=False) as data:
            pos, neg = data["test_pos"], data["test_neg"]
            pair_set = set(map(tuple, np.concatenate((pos, neg))))
            truth = labels[tuple(pos.T)]
            if (len(pair_set) != len(pos) + len(neg) or seen.intersection(pair_set)
                    or not truth.any(axis=1).all() or labels[tuple(neg.T)].any()
                    or not np.array_equal(data["truth_pos"], (truth * np.arange(1, 5)).max(axis=1))
                    or np.any(data["truth_neg"] != 0)):
                raise ValueError(f"Invalid labels or overlapping test pairs in fold {fold}")
            seen.update(pair_set)
            binary, cv_type, top1, _ = evaluate_optimized_with_comprehensive_metrics(
                *(torch.from_numpy(data[name]) for name in ("truth_pos", "truth_neg", "score_pos", "score_neg"))
            )
            if (f"{binary[0, 0]:.4f}" != auc_logged or f"{top1['top1_f1']:.4f}" != top1_logged
                    or not np.allclose(binary.ravel(), cv_type[2:])):
                raise ValueError(f"Replayed scores disagree with the recorded scores for fold {fold}")
            binary_rows.append(binary.ravel())
            cv_type_rows.append(cv_type[2:])
            top1_rows.append(top1)
            evidence[str(fold)] = {
                "last_logged_epoch": last_epoch, "auc": float(binary[0, 0]),
                "aupr": float(binary[0, 1]), "binary_f1": float(binary[0, 2]),
                "top1_f1": float(top1["top1_f1"]), "positive_pairs": len(pos),
                "tdrc_top1_f1": type_metrics(truth, data["score_pos"][:, 1:])["tdrc_top1_f1"],
            }

    completion = run / "completion.json"
    aggregate = None
    if completion.is_file():
        result = json.loads(completion.read_text(encoding="utf-8"))
        if len(evidence) != config["validation"] or result["folds"] != len(evidence):
            raise ValueError("Completion record does not have all logged and replayed folds")
        aggregate = {
            "binary": dict(zip(NUMBERS, np.mean(binary_rows, axis=0).tolist())),
            "cv_type": dict(zip(NUMBERS, np.mean(cv_type_rows, axis=0).tolist())),
            "top1": {key: float(np.mean([row[key] for row in top1_rows])) for key in top1_rows[0]},
        }
        for family, metrics in aggregate.items():
            if not all(np.isclose(value, result[family][key], atol=1e-10, rtol=0)
                       for key, value in metrics.items()):
                raise ValueError(f"Replayed {family} means disagree with the completion record")
    return {"run": run.name, "folds_verified": len(evidence), "folds_planned": config["validation"],
            "fold_evidence": evidence, "verified_aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    data = verify(options.run, options.log.read_text(encoding="utf-8"))
    payload = json.dumps(data, indent=2) + "\n"
    options.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
