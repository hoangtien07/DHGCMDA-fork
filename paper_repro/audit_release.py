"""Prepare the pinned release and record the strategy's pre-training checks."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE = HERE / "release"
REVISION = "06f9ba3eed92e334d9a2b8003fc06d83596e5614"
DATASET = "v2.0_495m383D"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def prepare_release():
    entries = []
    tree = git("ls-tree", "-rz", REVISION)
    for entry in filter(None, tree.split(b"\0")):
        metadata, name = entry.split(b"\t", 1)
        path = Path(name.decode("utf-8"))
        if path.parts[0] != DATASET and (path.parent != Path(".") or path.suffix != ".py"):
            continue
        blob = metadata.split()[2].decode("ascii")
        content = git("cat-file", "blob", blob)
        target = RELEASE / path
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"Pinned release file has local changes: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        entries.append({"path": path.as_posix(), "git_blob": blob,
                        "sha256": hashlib.sha256(content).hexdigest()})
    if not (RELEASE / "main_experiments_hetero1.py").is_file():
        raise RuntimeError("Release trainer is missing")
    return {"commit": REVISION, "files": entries}


def gip_audit():
    data = RELEASE / DATASET
    expected = np.loadtxt(data / "M_GSM.txt")
    cases = {
        "full_binary": "all_mirna_disease_pairs.csv",
        "typed_pairs": "multi_all_mirna_disease_pairs_without_negative.csv",
    }
    results = {}
    for name, filename in cases.items():
        rows = np.loadtxt(data / filename, delimiter=",", dtype=int)
        associations = np.zeros((495, 383), dtype=np.float64)
        associations[rows[:, 0] - 1, rows[:, 1] - 1] = rows[:, 2] > 0
        degree = associations.sum(axis=1)
        distance = degree[:, None] + degree[None, :] - 2 * (associations @ associations.T)
        gamma = len(degree) / degree.sum()
        results[name] = {
            "pairs": int(degree.sum()), "gamma": float(gamma),
            "max_abs_error": float(np.max(np.abs(np.exp(-gamma * distance) - expected))),
        }
    results["passes_g1"] = results["full_binary"]["max_abs_error"] < 1e-12
    return results


def view_audit():
    data = RELEASE / DATASET
    v20_one = np.loadtxt(data / "D_SSM1.txt")
    v20_two = np.loadtxt(data / "D_SSM2.txt")
    v32 = ROOT / "v3.2_spld_paper"
    inputs = ROOT / "HMDD_data" / "MDAv3.2-3"
    comparisons = {}
    for matrix, csv_name in (("D_MESH", "DSSM3.2_3.csv"),
                             ("M_MISIM", "mi_fun_sim_3.2_3.csv")):
        expected = np.load(v32 / f"{matrix}.npy")
        columns = range(1, expected.shape[1] + 1)
        actual = np.loadtxt(inputs / csv_name, delimiter=",", skiprows=1, usecols=columns)
        comparisons[matrix] = {"shape": list(expected.shape),
                               "max_abs_csv_difference": float(np.max(np.abs(expected - actual)))}
    return {"v20_semantic_max_abs_difference": float(np.max(np.abs(v20_one - v20_two))),
            "v32_bundle": comparisons}


def binary_feasibility():
    # Released splitting puts nine chunks of floor(n/10) pairs into CV.
    # The paper reports aggregates. Check each fold size and pooled counts,
    # allowing the four-decimal rounding used in the paper.
    target = (0.7915, 0.9421, 0.8600)
    fold_size = 9 * (8735 // 10) // 5
    sizes = (fold_size, 5 * fold_size)
    results = {}
    for positives in sizes:
        matches = []
        for tp in range(1, positives + 1):
            if abs(tp / positives - target[1]) > 0.00005:
                continue
            center = round(tp / target[0] - tp)
            for fp in range(max(0, center - 2), min(positives, center + 2) + 1):
                scores = (tp / (tp + fp), tp / positives, 2 * tp / (positives + tp + fp))
                if all(abs(score - expected) <= 0.00005 for score, expected in zip(scores, target)):
                    matches.append({"tp": tp, "fp": fp, "precision_recall_f1": scores})
        results[str(positives)] = matches
    # One integer witness for metrics averaged per fold.
    counts = [(1558, 426), (1405, 429), (1432, 352), (1495, 371), (1510, 372)]
    fold_metrics = [(tp / (tp + fp), tp / fold_size, 2 * tp / (fold_size + tp + fp))
                    for tp, fp in counts]
    mean = np.mean(fold_metrics, axis=0).tolist()
    if any(abs(actual - expected) > 0.00005 for actual, expected in zip(mean, target)):
        raise RuntimeError("The five-fold witness no longer matches the target")
    return {"single_or_pooled_matches": results,
            "five_fold_witness": {"tp_fp_by_fold": counts, "mean_precision_recall_f1": mean}}


def evaluator_audit():
    sys.path.insert(0, str(RELEASE))
    import torch
    from Calculate_Metrics import Metric_fun
    from main_experiments_hetero1 import evaluate_optimized_with_comprehensive_metrics

    types = torch.arange(1, 6)
    negatives = torch.zeros_like(types)
    positive_scores = torch.cat((torch.full((5, 1), 0.9), torch.eye(5)), dim=1)
    negative_scores = torch.full_like(positive_scores, 0.1)
    metric = Metric_fun()
    control = metric.compute_top1_metrics(types[:4].tolist(), positive_scores[:4, :5].tolist())
    binary, cv_type, top1, _ = evaluate_optimized_with_comprehensive_metrics(
        types, negatives, positive_scores, negative_scores
    )
    if control["top1_f1"] != 1 or any(top1.values()) or binary[0, 0] < 0.99:
        raise RuntimeError("Released evaluator behaved differently from the expected controls")
    if not np.allclose(binary.ravel(), cv_type[2:]):
        raise RuntimeError("The seven printed CV_type fields differ from the binary fields")
    names = ("top1_precision", "top1_recall", "top1_f1")
    return {"four_type_control": {name: float(control[name]) for name in names},
            "six_column_top1": top1,
            "binary": binary.ravel().tolist(), "cv_type_printed_fields": cv_type[2:].tolist()}


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    manifest = prepare_release()
    (HERE / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {"gip": gip_audit(), "views": view_audit(), "binary_feasibility": binary_feasibility()}
    if "--evaluator" in sys.argv[2:]:
        result["evaluator"] = evaluator_audit()
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["gip"]["passes_g1"]:
        raise SystemExit("G1 failed")
