"""Train the last v3.2 control folds while the main batch handles earlier folds."""

import argparse
import os
from pathlib import Path
import re
import shutil
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    options = parser.parse_args()
    primary = options.runs.resolve()
    staging = primary / "parallel-controls"
    staging.mkdir(exist_ok=False)
    shutil.copytree(primary / "clean_protocol", staging / "clean_protocol")
    os.environ.update(DHGCMDA_RUN_ROOT=str(staging), CUDA_VISIBLE_DEVICES="-1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

    from clean_experiment import CONTROLS, DATASET, METHODS, baseline, model
    from prepare_v32 import sha

    pipeline = psutil.Process(int((primary / "pipeline.pid").read_text()))
    if not any("continue_cpu.ps1" in arg for arg in pipeline.cmdline()):
        raise RuntimeError("Main CPU pipeline is not running")
    fold_pattern = re.compile(r"--dataset v3\.2_spld_paper --seed 1234 --fold (\d)")

    def primary_fold():
        if not pipeline.is_running() or (primary / "controls-cpu.json").exists():
            return 5
        matches = fold_pattern.findall((primary / "pipeline.log").read_text(encoding="utf-8", errors="replace"))
        return max(map(int, matches), default=0)

    for fold in (5, 4, 3):
        split = f"clean_protocol/{DATASET}-1234-fold{fold}.npz"
        for area, variants in (("clean_models", ("clean", *CONTROLS)), ("clean_baselines", METHODS)):
            for name in variants:
                if primary_fold() >= fold - 1:
                    print(f"Stopping because the main batch is approaching fold {fold}", flush=True)
                    return
                suffix = "-cpu" if area == "clean_models" else ""
                relative = Path(area) / f"{DATASET}-1234-fold{fold}-{name}{suffix}"
                target = primary / relative
                if target.exists():
                    continue
                while psutil.virtual_memory().available < 4 * 1024**3:
                    if primary_fold() >= fold - 1:
                        return
                    print("Waiting for at least 4 GiB of available RAM", flush=True)
                    time.sleep(30)
                if area == "clean_models":
                    model(DATASET, 1234, fold, name, "cpu", confirm=True)
                else:
                    baseline(DATASET, 1234, fold, name)
                if sha(staging / split) != sha(primary / split):
                    raise RuntimeError(f"Frozen split changed for fold {fold}")
                if primary_fold() >= fold - 1 or target.exists():
                    print(f"Keeping {relative} in staging because the main batch is approaching", flush=True)
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                (staging / relative).rename(target)
                print(f"Published {relative}", flush=True)


if __name__ == "__main__":
    main()
