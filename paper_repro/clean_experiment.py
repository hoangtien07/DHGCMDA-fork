"""Execute Stages 4–5 and summarize paired outcomes under the frozen protocol."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from clean_baselines import METHODS
from clean_protocol import DATASETS, RUNS, SEEDS
from prepare_v32 import DATASET, HERE
from train_controls import CANDIDATES, CONTROLS

METRIC = "tdrc_top1_f1"
SECONDARY = ("multilabel_macro_f1", "binary_auc", "binary_aupr")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def invoke(script, *arguments):
    argv = [sys.executable, "-B", "-u", str(HERE / script), *map(str, arguments)]
    print(" ".join(argv), flush=True)
    subprocess.run(argv, cwd=HERE.parent, check=True)


def baseline(dataset, seed, fold, method):
    path = RUNS / "clean_baselines" / f"{dataset}-{seed}-fold{fold}-{method}" / "completion.json"
    if not path.is_file():
        invoke("clean_baselines.py", method, "--dataset", dataset, "--seed", seed, "--fold", fold)
    return read_json(path)


def model(dataset, seed, fold, variant, device, confirm):
    path = RUNS / "clean_models" / f"{dataset}-{seed}-fold{fold}-{variant}-{device}"
    args = ("--dataset", dataset, "--seed", seed, "--fold", fold, "--device", device)
    ready = path / "ready.json"
    if not ready.is_file():
        invoke("train_controls.py", "fit", variant, *args)
    if not confirm:
        return read_json(ready)
    result = path / "completion.json"
    if not result.is_file():
        invoke("train_controls.py", "confirm", variant, *args)
    return read_json(result)


def score(result, holdout="outer_test"):
    return result[holdout][METRIC]


def statistics(values):
    return {"mean": float(np.mean(values)), "std_ddof_1": float(np.std(values, ddof=1))}


def compare(a, b):
    delta = np.asarray(a) - np.asarray(b)
    signs = delta[delta != 0]
    return {**statistics(delta), "positive": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
            "wilcoxon_p_descriptive": float(wilcoxon(delta).pvalue) if len(signs) else 1.0,
            "sign_p_descriptive": float(binomtest(int((signs > 0).sum()), len(signs)).pvalue)
            if len(signs) else 1.0}


def controls(device):
    output = {}
    for dataset in DATASETS:
        variants = [name for name in CONTROLS if name != "multilabel-only" or dataset == DATASET]
        variants.insert(0, "clean")
        rows = {name: [] for name in (*METHODS, *variants)}
        for fold in range(1, 6):
            for name in variants:
                rows[name].append(model(dataset, 1234, fold, name, device, confirm=True))
            for name in METHODS:
                rows[name].append(baseline(dataset, 1234, fold, name))
        scores = {name: [score(value) for value in records] for name, records in rows.items()}
        output[dataset] = {
            "scores": {name: statistics(values) for name, values in scores.items()},
            "mean_metrics": {name: {
                metric: float(np.mean([value["outer_test"][metric] for value in records]))
                for metric in records[0]["outer_test"]} for name, records in rows.items()},
            "controlled_difference_vs_legacy": {
                name: compare(scores[name], scores["legacy"]) for name in variants if name != "legacy"},
        }
    return output


def improvements(device):
    output = {}
    for dataset in DATASETS:
        rows = []
        for seed in SEEDS:
            for fold in range(1, 6):
                validations = {name: model(dataset, seed, fold, name, device, confirm=False)
                               for name in CANDIDATES}
                winner = max(CANDIDATES, key=lambda name: score(validations[name], "inner_validation"))
                chosen = model(dataset, seed, fold, winner, device, confirm=True)
                clean = model(dataset, seed, fold, "clean", device, confirm=True)
                baselines = {name: baseline(dataset, seed, fold, name) for name in METHODS}
                strongest = max(METHODS, key=lambda name: score(baselines[name], "inner_validation"))
                row = {
                    "seed": seed, "fold": fold, "candidate_selected_in_validation": winner,
                    "baseline_selected_in_validation": strongest, "selected": score(chosen),
                    "clean": score(clean), "strongest_baseline": score(baselines[strongest]),
                    **{name: score(value) for name, value in baselines.items()},
                    "secondary": {
                        name: {metric: record["outer_test"][metric] for metric in SECONDARY}
                        for name, record in (("selected", chosen), ("clean", clean),
                                             ("strongest_baseline", baselines[strongest]))
                    },
                }
                if dataset == DATASET:
                    row["multilabel_exploratory"] = score(
                        model(dataset, seed, fold, "clean-multilabel", device, confirm=True))
                rows.append(row)
        selected = [row["selected"] for row in rows]
        comparison = {name: compare(selected, [row[name] for row in rows])
                      for name in ("clean", "strongest_baseline", *METHODS)}
        output[dataset] = {
            "fold_seed_results": rows, "selected": statistics(selected), "paired_differences": comparison,
            "candidate_counts": dict(Counter(row["candidate_selected_in_validation"] for row in rows)),
            "secondary": {
                name: {metric: statistics([row["secondary"][name][metric] for row in rows])
                       for metric in SECONDARY}
                for name in ("selected", "clean", "strongest_baseline")
            },
            "beats_prior_and_logistic": all(comparison[name]["mean"] > 0 for name in ("prior", "logistic")),
            "beats_clean_and_strongest_baseline": all(
                comparison[name]["mean"] > 0 for name in ("clean", "strongest_baseline")
            ),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("controls", "improvements", "all"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    options = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    g3_result = RUNS / "g3.json"
    rescore = RUNS / f"rescored-{options.device}.json"
    if not g3_result.is_file() or not rescore.is_file():
        raise RuntimeError("Complete the v3.2 G3 workflow and rescore its predictions first")
    invoke("clean_protocol.py")
    control_result = RUNS / f"controls-{options.device}.json"
    if options.stage in {"controls", "all"}:
        report = controls(options.device)
        control_result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(control_result, flush=True)
    if options.stage in {"improvements", "all"}:
        if not control_result.is_file():
            raise RuntimeError("Stage 4 comparisons are required before Stage 5")
        report = improvements(options.device)
        report["inference_scope"] = ("Paired fold × seed results share data. P-values are descriptive, "
                                     "and do not establish independent external generalization.")
        report["negative_result_on_v32"] = not report[DATASET]["beats_prior_and_logistic"]
        report["improvement_on_both_datasets"] = all(
            report[dataset]["beats_clean_and_strongest_baseline"] for dataset in DATASETS
        )
        result = RUNS / f"improvements-{options.device}.json"
        result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(result, flush=True)


if __name__ == "__main__":
    main()
