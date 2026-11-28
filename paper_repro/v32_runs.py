"""Enforce the calibration lock and G3 rules for the v3.2 reproduction."""

import argparse
import json
from pathlib import Path

import numpy as np

from calibration import RUNS, calibration_run, read_json
from prepare_v32 import DATA, DATASET, HERE, PORT, sha

TABLE4 = {"auc": (0.9271, 0.02), "aupr": (0.9181, 0.02), "f1": (0.8674, 0.02)}
TABLE3 = {"precision": (0.7915, 0.03), "recall": (0.9421, 0.03), "f1": (0.8600, 0.02)}


def run_name(profile, seed, device, diagnostic=None):
    variant = f"-{diagnostic}" if diagnostic else ""
    return f"v32-{profile}-{seed}{variant}-{device}"


def assess(metrics, target):
    deltas = {name: metrics[name] - paper for name, (paper, _) in target.items()}
    distance = max(abs(deltas[name]) / tolerance for name, (_, tolerance) in target.items())
    return {"delta": deltas, "distance_in_tolerances": distance, "matched": distance <= 1}


def decide(k_star, t_star):
    k_result, t_result = assess(k_star, TABLE4), assess(t_star, TABLE3)
    better = min(("K_star", "T_star"), key=lambda profile:
                 {"K_star": k_result, "T_star": t_result}[profile]["distance_in_tolerances"])
    if k_result["matched"] and t_result["matched"]:
        status = "matched"
    elif max(k_star["auc"], t_star["auc"]) < 0.89:
        status = "far"
    elif t_star["auc"] >= 0.90 and t_star["recall"] < 0.91:
        status = "near"
    else:
        status = "inconclusive"
    return {"status": status, "table4_K_star": k_result,
            "table3_T_star": t_result, "closer_profile": better}


def g3(runs, device):
    k = read_json(runs / run_name("K_star", 1234, device) / "completion.json")
    t = read_json(runs / run_name("T_star", 1234, device) / "completion.json")
    return decide(k["binary"], t["binary"])


def resolve(options):
    lock_path = RUNS / "calibration_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("v3.2 training requires the completed v2.0 calibration lock")
    lock = read_json(lock_path)
    if lock["device"] != options.device or not lock["g2"]["passed"]:
        raise RuntimeError("v3.2 requires the device and passed G2 from the calibration lock")
    for mode, expected in lock["completion_sha256"].items():
        config, _, content_hash = calibration_run(RUNS, mode, options.device)
        if content_hash != expected or config["release"] != lock["release"]:
            raise RuntimeError(f"Calibration evidence changed for {mode}")
    if options.diagnostic or options.seed != 1234:
        decision = g3(RUNS, options.device)
        if options.diagnostic:
            if decision["status"] != "near" or options.profile != "T_star" or options.seed != 1234:
                raise RuntimeError("G3 permits diagnostics only for the near T_star case")
        elif decision["status"] != "matched" or options.profile != decision["closer_profile"]:
            raise RuntimeError("Extra seeds require a G3 match on the closer configuration")

    manifest_path = DATA / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["release"] != lock["release"]:
        raise RuntimeError("Port and calibration use different release commits")
    variant = "diagnostic-disease-gip" if options.diagnostic == "disease-gip" else "standard"
    for relative_path, expected in manifest["code_sha256"].items():
        if sha(PORT / relative_path) != expected:
            raise RuntimeError(f"Port source changed: {relative_path}")
    for filename, expected in manifest["data"]["variants"][variant].items():
        if sha(DATA / variant / DATASET / filename) != expected:
            raise RuntimeError(f"Port input changed: {filename}")

    profile = lock[options.profile]
    heads = 4 if options.diagnostic == "heads" else profile["n_head"]
    counts = manifest["data"]["collapsed_counts"]
    inverse = 1 / np.array(counts, dtype=float)
    opts = ["--dataset", DATASET, "--mi_num", "411", "--dis_num", "271",
            "--num_association_types", "5", "--n_head", str(heads), "--class_weights",
            *map(str, inverse / inverse.mean())]
    return {"name": run_name(options.profile, options.seed, options.device, options.diagnostic),
            "code": PORT, "data": DATA / variant, "argv": opts,
            "exist_weight": profile["exist_weight"],
            "input_manifest": {"dataset_sha256": sha(manifest_path), "lock_sha256": sha(lock_path),
                               "profile": options.profile, "variant": variant,
                               "diagnostic": options.diagnostic}}


def top1_five_types(truth, scores):
    truth = np.asarray(truth).ravel()
    scores = np.asarray(scores)
    if scores.shape != (len(truth), 6) or len(truth) == 0 or not np.isin(truth, range(1, 6)).all():
        raise ValueError("Expected one label 1–5 and six prediction columns per positive pair")
    prediction = np.argmax(scores[:, 1:], axis=1) + 1
    correct = truth == prediction
    precision = float(correct.mean())
    recall = float(np.mean([correct[truth == channel].mean() for channel in np.unique(truth)]))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"top1_precision": precision, "top1_recall": recall, "top1_f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--runs", type=Path, default=RUNS)
    options = parser.parse_args()
    result = g3(options.runs, options.device)
    output = json.dumps(result, indent=2) + "\n"
    (options.runs / "g3.json").write_text(output, encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
