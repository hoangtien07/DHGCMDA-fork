"""Freeze shared nested pair splits and construct train-only similarity inputs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from calibration import RUNS
from prepare_v32 import DATA, DATASET, HERE, ROOT, RELEASE, gip, sha
from score_predictions import load_labels

SEEDS = (1234, 42, 2025)
DATASETS = ("v2.0_495m383D", DATASET)


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nested_splits(labels, seed):
    known = labels.any(axis=-1)
    positives = np.argwhere(known)
    negatives = np.argwhere(~known)
    rng = np.random.default_rng(seed)
    negatives = negatives[rng.choice(len(negatives), len(positives), replace=False)]
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (outer_train, test) in enumerate(outer.split(positives), 1):
        shuffled = rng.permutation(outer_train)
        size = len(shuffled) // 5
        val, fit = shuffled[:size], shuffled[size:]
        rows = {}
        for split, indices in (("fit", fit), ("val", val), ("test", test)):
            rows[f"{split}_pos"] = positives[indices]
            rows[f"{split}_neg"] = negatives[indices]
        all_pairs = np.concatenate(list(rows.values()), axis=0)
        if len(np.unique(all_pairs, axis=0)) != len(all_pairs):
            raise ValueError(f"Pair overlap at seed {seed}, fold {fold}")
        yield rows


def disease_semantic(dataset):
    if dataset == DATASET:
        path = DATA / "standard" / DATASET / "D_SSM1.txt"
    elif dataset == "v2.0_495m383D":
        path = RELEASE / dataset / "D_SSM1.txt"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return path, np.loadtxt(path)


def fit_views(labels, fit_pos, semantic):
    masked = np.zeros_like(labels, dtype=bool)
    i, j = fit_pos.T
    masked[i, j] = labels[i, j]
    if not masked[i, j].any(axis=1).all():
        raise ValueError("Positive pairs must have a label in the fit mask")
    binary = masked.any(axis=2).astype(np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    if semantic.shape != (labels.shape[1], labels.shape[1]) or not np.isfinite(semantic).all():
        raise ValueError("Invalid disease semantic input")
    # Some matrix exports differ in the last digit by direction.
    semantic = (semantic + semantic.T) * 0.5
    functional_module = load_module(
        ROOT / "faithful_repro" / "independent_v20_trainfunc_20261117" / "functional.py",
        "fold_functional_similarity")
    return {"masked_labels": masked, "mirna_gip": gip(binary), "disease_gip": gip(binary.T),
            "mirna_functional": functional_module.functional_similarity(binary, semantic),
            "disease_semantic": semantic}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "seeds": list(SEEDS), "folds_per_seed": 5, "inner_validation_fraction": 0.2,
        "pair_strata": "positive pairs and a preselected 1:1 negative sample",
        "primary": "TDRC Top-1 F1",
        "selection": "inner validation only, shared by all methods",
        "functional_view": "fit-only disease-set similarity",
        "disease_views": ["fit-only disease GIP", "bundled disease semantic with declared provenance"],
        "mirna_views": ["fit-only miRNA GIP", "fit-only disease-set similarity"],
        "datasets": {},
    }
    for dataset in DATASETS:
        labels = load_labels(dataset)
        semantic_path, _ = disease_semantic(dataset)
        records = {}
        for seed in SEEDS:
            for index, split in enumerate(nested_splits(labels, seed), 1):
                path = output / f"{dataset}-{seed}-fold{index}.npz"
                np.savez_compressed(path, **split)
                records[path.name] = sha(path)
        result["datasets"][dataset] = {
            "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "disease_semantic_sha256": sha(semantic_path), "split_sha256": records}
    (output / "protocol.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_frozen_fold(dataset, seed, fold, runs=RUNS):
    frozen = runs / "clean_protocol"
    contract = json.loads((frozen / "protocol.json").read_text(encoding="utf-8"))
    evidence = contract["datasets"][dataset]
    path = frozen / f"{dataset}-{seed}-fold{fold}.npz"
    labels = load_labels(dataset)
    semantic_path, semantic = disease_semantic(dataset)
    split_hash = sha(path)
    if (split_hash != evidence["split_sha256"][path.name]
            or hashlib.sha256(labels.tobytes()).hexdigest() != evidence["label_sha256"]
            or sha(semantic_path) != evidence["disease_semantic_sha256"]):
        raise RuntimeError("Benchmark inputs differ from the frozen protocol")
    with np.load(path, allow_pickle=False) as source:
        return labels, semantic, dict(source), split_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    opts = parser.parse_args()
    if not (opts.runs / "g3.json").is_file():
        raise RuntimeError("The clean experiment starts after the main v3.2 runs have a G3 decision")
    frozen = opts.runs / "clean_protocol"
    if (frozen / "protocol.json").is_file():
        print(frozen / "protocol.json")
        return
    result = freeze(frozen)
    print(f"Frozen {sum(len(row['split_sha256']) for row in result['datasets'].values())} shared splits")


if __name__ == "__main__":
    main()
