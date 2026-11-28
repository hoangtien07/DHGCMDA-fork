"""Run the pinned release or its five-type port and save fold predictions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np
import torch
from calibration import RUNS

HERE = Path(__file__).resolve().parent
RELEASE = HERE / "release"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train_full_fold(train, model, data, optimizer, args):
    steps = 0

    def count_step(_optimizer, _args, _kwargs):
        nonlocal steps
        steps += 1

    handle = optimizer.register_step_post_hook(count_step)
    try:
        predictions = train(model, data, optimizer, args)
    finally:
        handle.remove()
    if steps != args.epoch:
        raise RuntimeError(f"Fold stopped at {steps} optimizer steps; expected {args.epoch}")
    return predictions, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "c1", "c2", "c3", "v32"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profile", choices=("K_star", "T_star"))
    parser.add_argument("--seed", type=int, choices=(42, 1234, 2025), default=1234)
    parser.add_argument("--diagnostic", choices=("heads", "disease-gip"))
    options = parser.parse_args()
    if (options.mode == "v32") != bool(options.profile):
        parser.error("--profile is required for v32 and unavailable for v2.0")
    if options.mode != "v32" and (options.seed != 1234 or options.diagnostic):
        parser.error("--seed and --diagnostic are only available in v32 mode")

    if options.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch environment")
        sample = torch.eye(16, device="cuda", requires_grad=True)
        (sample @ sample).sum().backward()
        torch.cuda.synchronize()
    elif torch.cuda.is_available():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 before requesting CPU in the pinned release")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = json.loads((HERE / "release_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = RELEASE / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Pinned file has changed: {path}")

    port = None
    if options.mode == "v32":
        from v32_runs import resolve
        port = resolve(options)
    sys.path.insert(0, str(port["code"] if port else RELEASE))
    import main_experiments_hetero1 as author
    import torch_geometric
    from param import parameter_parser

    name = port["name"] if port else f"{options.mode}-{options.device}"
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=False)
    os.chdir(run)
    sys.argv = [sys.argv[0], "--data_path", str(port["data"] if port else RELEASE), "--device", options.device]
    if options.mode == "smoke":
        sys.argv.extend(["--epoch", "3", "--validation", "2"])
    if port:
        sys.argv.extend(port["argv"])
    args = parameter_parser()
    if port or options.mode == "c3":
        # The released module seeds at import time and does not consume args.seed.
        args.seed = options.seed if port else 42
        author.seed_torch(args.seed)
    exist_weight = port["exist_weight"] if port else (0.1 if options.mode == "c2" else 0.3)
    if exist_weight != 0.3:
        loss_type = author.SimplifiedMultiTypeAssociationLoss
        original_init = loss_type.__init__

        def init_with_exist_weight(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.exist_weight = exist_weight

        # The pinned constructor resolves its class name in an explicit super().
        loss_type.__init__ = init_with_exist_weight
    write_json(run / "config.json", {
        "release": manifest["commit"], "args": vars(args),
        "exist_weight": exist_weight, "type_weight": 0.7,
        "input_manifest": port["input_manifest"] if port else None,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "pyg": torch_geometric.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name() if options.device == "cuda" else None,
    })

    original_epoch = author.train_epoch_optimized
    fold_steps = []
    corrected_top1 = []

    def capture(model, fold_data, opt, arguments):
        predictions, steps = train_full_fold(original_epoch, model, fold_data, opt, arguments)
        truth_pos, truth_neg, score_pos, score_neg = predictions
        test_pos, test_neg = fold_data[3]
        items = {"test_pos": test_pos, "test_neg": test_neg, "truth_pos": truth_pos,
                 "truth_neg": truth_neg, "score_pos": score_pos, "score_neg": score_neg}
        arrays = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in items.items()}
        if not np.isfinite(arrays["score_pos"]).all() or not np.isfinite(arrays["score_neg"]).all():
            raise RuntimeError("Fold produced non-finite predictions")
        expected = args.num_association_types + 1
        for key in ("pos", "neg"):
            count = len(arrays[f"test_{key}"])
            if arrays[f"score_{key}"].shape != (count, expected) or len(arrays[f"truth_{key}"]) != count:
                raise RuntimeError(f"Unexpected {key} prediction or truth shape for {count} pairs")
        if port:
            from v32_runs import top1_five_types
            corrected_top1.append(top1_five_types(arrays["truth_pos"], arrays["score_pos"]))
        fold_steps.append(steps)
        temporary = run / f"fold_{len(fold_steps)}.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(run / f"fold_{len(fold_steps)}.npz")
        return predictions

    author.train_epoch_optimized = capture
    binary, cv_type, top1 = author.main_optimized(args)
    fold = len(fold_steps)
    if fold != args.validation or not np.isfinite(binary).all() or binary[0][0] <= 0:
        raise RuntimeError("Released trainer did not produce valid metrics for every fold")

    names = ("auc", "aupr", "f1", "accuracy", "recall", "specificity", "precision")
    result = {"folds": fold, "epochs_per_fold": args.epoch, "optimizer_steps_by_fold": fold_steps,
              "binary": dict(zip(names, np.asarray(binary).ravel().tolist())),
              "cv_type": dict(zip(names, np.asarray(cv_type)[2:].tolist())), "top1": top1}
    if options.device == "cuda":
        result["peak_cuda_reserved_mib"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2))
    if port:
        result["top1_five_types"] = {
            metric: float(np.mean([row[metric] for row in corrected_top1]))
            for metric in corrected_top1[0]
        }
    if options.mode == "c1":
        reference = {"auc": 0.9738, "aupr": 0.9669, "f1": 0.9278}
        deltas = {metric: result["binary"][metric] - value for metric, value in reference.items()}
        result["g2"] = {"paper_table4_with_auc_aupr_swapped": reference,
                        "delta": deltas, "passed": all(abs(value) <= 0.01 for value in deltas.values())}
    write_json(run / "completion.tmp", result)
    (run / "completion.tmp").replace(run / "completion.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
