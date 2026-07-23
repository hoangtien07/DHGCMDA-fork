#!/usr/bin/env python3
"""Cross-platform, provenance-safe runner for the active DHGCMDA experiment queue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "docs" / "status" / "registry.json"
READY_ARMS = {"p6", "p2", "p1", "p5"}
ALL_ARMS = READY_ARMS | {"p3"}
SEEDS = (0, 42, 1234)
CANONICAL = [
    "--device", "cpu",
    "--K_neigs", "2",
    "--cv_scheme", "full",
    "--leakage_free",
    "--deterministic",
    "--loss_mode", "two_head",
    "--exist_weight", "0.3",
    "--epoch", "650",
    "--validation", "5",
]


def default_python() -> Path:
    return ROOT / ("venv/Scripts/python.exe" if platform.system() == "Windows" else "venv/bin/python")


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def arm_variants(arm: str) -> list[tuple[str, list[str]]]:
    if arm == "p6":
        return [("diag", ["--predictor_mode", "diag", "--fusion_mode", "fixed"])]
    if arm == "p2":
        return [("real_sequence", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed", "--mirna_seq_sim_path", "v2.0_495m383D/M_SEQ.txt"])]
    if arm == "p1":
        return [("gate", ["--predictor_mode", "full_bilinear", "--fusion_mode", "gate"])]
    if arm == "p5":
        return [
            ("ce", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed", "--type_loss", "ce", "--dump_scores"]),
            ("ldam", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed", "--type_loss", "ldam", "--dump_scores"]),
        ]
    raise ValueError(arm)


def build_jobs(arm: str, python: Path, run_id: str) -> list[dict]:
    jobs: list[dict] = []
    for variant, variant_args in arm_variants(arm):
        for seed in SEEDS:
            job_id = f"{variant}_s{seed}"
            log_dir = ROOT / "logs" / "plan_n" / arm / run_id
            result_dir = ROOT / "results" / "plan_n" / arm / run_id
            log_path = log_dir / f"{job_id}.log"
            result_path = result_dir / f"{job_id}.json"
            args = list(variant_args)
            if arm == "p5":
                dump_dir = result_dir / "dumps" / job_id
                args[args.index("--dump_scores") + 1:args.index("--dump_scores") + 1] = [str(dump_dir.relative_to(ROOT))]
            command = [str(python), "main_experiments_hetero1.py", *CANONICAL, *args, "--seed", str(seed)]
            jobs.append({"id": job_id, "log": log_path, "result": result_path, "command": command})
    return jobs


def print_status() -> int:
    registry = load_registry()
    print("Active queue:")
    for record in sorted(registry["records"], key=lambda item: (item.get("queue_order", 999), item["id"])):
        if record.get("kind") in {"experiment", "analysis"}:
            suffix = f" — {record.get('reason', record.get('comparison', ''))}"
            print(f"  {record['id']}: {record['status'].upper()}{suffix}")
    return 0


def check_environment(python: Path) -> int:
    checks = {
        "registry": REGISTRY_PATH,
        "status document": ROOT / "docs/status/PROJECT_STATUS.md",
        "validator": ROOT / "tools/validate_tracking.py",
        "training entrypoint": ROOT / "main_experiments_hetero1.py",
        "sequence matrix": ROOT / "v2.0_495m383D/M_SEQ.txt",
        "python": python,
    }
    failed = False
    for label, path in checks.items():
        ok = path.exists()
        print(f"[{ 'OK' if ok else 'MISSING' }] {label}: {path}")
        failed |= not ok
    if failed:
        return 1
    validator = subprocess.run([str(python), "tools/validate_tracking.py", "--root", str(ROOT)], cwd=ROOT)
    return 1 if failed or validator.returncode else 0


def write_index(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def execute_job(job: dict, run_index: dict) -> bool:
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    job["result"].parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(job["command"]))
    with job["log"].open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(job["command"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        code = process.wait()
    job["exit_code"] = code
    if code == 0:
        parse = subprocess.run([job["command"][0], "parse_metrics.py", str(job["log"].relative_to(ROOT)), str(job["result"].relative_to(ROOT))], cwd=ROOT)
        job["parse_exit_code"] = parse.returncode
        return parse.returncode == 0
    return False


def run_arm(arm: str, python: Path, run_id: str, dry_run: bool) -> int:
    if arm == "p3":
        print("P3 is BLOCKED: softmax type outputs are incompatible with genuine multi-label BCE. Use a dedicated logic-fix branch first.", file=sys.stderr)
        return 2
    if arm not in READY_ARMS:
        print(f"Unknown arm {arm}", file=sys.stderr)
        return 2
    jobs = build_jobs(arm, python, run_id)
    result_root = ROOT / "results" / "plan_n" / arm / run_id
    index_path = result_root / "run_index.json"
    if result_root.exists():
        print(f"Refusing to overwrite existing run directory: {result_root}", file=sys.stderr)
        return 2
    if dry_run:
        for job in jobs:
            print("$", " ".join(job["command"]))
            print(f"  log: {job['log'].relative_to(ROOT)}")
            print(f"  result: {job['result'].relative_to(ROOT)}")
        return 0

    run_index = {
        "arm": arm,
        "run_id": run_id,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "jobs": jobs,
    }
    write_index(index_path, run_index)
    succeeded = True
    for job in jobs:
        if not execute_job(job, run_index):
            succeeded = False
        write_index(index_path, run_index)
        if not succeeded:
            break
    run_index["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    run_index["status"] = "complete" if succeeded else "failed"
    write_index(index_path, run_index)
    return 0 if succeeded else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=default_python())
    parser.add_argument("command", choices=["status", "check", "dry-run", "run"])
    parser.add_argument("arm", nargs="?", choices=sorted(ALL_ARMS))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    python = args.python.resolve()

    if args.command == "status":
        return print_status()
    if args.command == "check":
        return check_environment(python)
    if args.arm is None:
        parser.error("dry-run and run require an arm")
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_arm(args.arm, python, run_id, dry_run=args.command == "dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
