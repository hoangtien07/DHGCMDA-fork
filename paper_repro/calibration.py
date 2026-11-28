"""Validate the v2.0 calibration and freeze the exact v3.2 inputs to training."""

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("DHGCMDA_RUN_ROOT", HERE / "runs")).resolve()
TABLE3 = {"top1_precision": 0.5842, "top1_recall": 0.6341, "top1_f1": 0.5970}
TABLE4 = {"auc": 0.9738, "aupr": 0.9669, "f1": 0.9278}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def differences(result, paper):
    return {metric: result[metric] - target for metric, target in paper.items()}


def calibration_run(runs, mode, device):
    path = runs / f"{mode}-{device}"
    completion = path / "completion.json"
    result = read_json(completion)
    config = read_json(path / "config.json")
    expected_seed = 42 if mode == "c3" else 1234
    expected_weight = 0.1 if mode == "c2" else 0.3
    if (result["folds"], result["epochs_per_fold"]) != (5, 650):
        raise ValueError(f"{mode}: invalid fold or epoch count")
    # C1 was started before the optimizer-step evidence was added.
    if (mode != "c1" or "optimizer_steps_by_fold" in result) and result.get("optimizer_steps_by_fold") != [650] * 5:
        raise ValueError(f"{mode}: optimizer-step evidence does not cover all five folds")
    args = config["args"]
    if (args["seed"], args["n_head"], args["device"]) != (expected_seed, 8, device):
        raise ValueError(f"{mode}: invalid seed, attention heads or device")
    if (config.get("exist_weight", 0.3), config.get("type_weight", 0.7)) != (expected_weight, 0.7):
        raise ValueError(f"{mode}: invalid loss weights")
    if any(not (path / f"fold_{fold}.npz").is_file() for fold in range(1, 6)):
        raise ValueError(f"{mode}: incomplete fold predictions")
    return config, result, hashlib.sha256(completion.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("g2", "freeze"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--runs", type=Path, default=RUNS)
    options = parser.parse_args()
    c1 = calibration_run(options.runs, "c1", options.device)
    delta4 = differences(c1[1]["binary"], TABLE4)
    passed = all(abs(delta) <= 0.01 for delta in delta4.values())
    gate = {"passed": passed, "paper": TABLE4, "delta": delta4, "c1_sha256": c1[2]}
    (options.runs / "g2.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)
    if not passed:
        raise SystemExit("G2 failed. Investigate the C1 environment and cache before continuing.")
    if options.stage == "g2":
        return

    c2 = calibration_run(options.runs, "c2", options.device)
    c3 = calibration_run(options.runs, "c3", options.device)
    if len({run[0]["release"] for run in (c1, c2, c3)}) != 1:
        raise ValueError("Calibration runs do not share the same release")
    contract = {
        "release": c1[0]["release"], "device": options.device, "g2": gate,
        "K_star": {"source": "c1", "exist_weight": 0.3, "type_weight": 0.7, "n_head": 8},
        "T_star": {"source": "c2", "exist_weight": 0.1, "type_weight": 0.7, "n_head": 8},
        "c2_table3_delta": differences(c2[1]["top1"], TABLE3),
        "c3_minus_c1_binary": differences(c3[1]["binary"], c1[1]["binary"]),
        "c3_minus_c1_top1": differences(c3[1]["top1"], c1[1]["top1"]),
        "completion_sha256": {name: run[2] for name, run in zip(("c1", "c2", "c3"), (c1, c2, c3))},
    }
    path = options.runs / "calibration_lock.json"
    content = json.dumps(contract, indent=2) + "\n"
    if path.exists() and read_json(path) != contract:
        raise RuntimeError(f"Existing calibration lock differs: {path}")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
