#!/usr/bin/env python3
"""Cross-platform, provenance-safe runner for the active DHGCMDA experiment queue."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "docs" / "status" / "registry.json"
READY_ARMS = {"p6", "p2", "p1", "p5"}
REFERENCE_ARMS = {"baseline"}
RUNNABLE_ARMS = READY_ARMS | REFERENCE_ARMS
ALL_ARMS = RUNNABLE_ARMS | {"p3"}
SEEDS = (0, 42, 1234)
MIN_FREE_DISK = 10 * 1024**3
MIN_AVAILABLE_RAM = 3 * 1024**3
RECOMMENDED_AVAILABLE_RAM = 6 * 1024**3
MIN_TOTAL_RAM = 12 * 1024**3
INDEX_SCHEMA_VERSION = 2
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
CODE_SUFFIXES = {".py", ".ps1", ".sh"}
_GATE_RE = re.compile(r"gate .*?=([0-9]*\.?[0-9]+)", re.IGNORECASE)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_python() -> Path:
    return ROOT / ("venv/Scripts/python.exe" if platform.system() == "Windows" else "venv/bin/python")


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
    ).strip()


def git_state() -> dict:
    try:
        sha = git_output("rev-parse", "HEAD")
        tracked_dirty = bool(git_output("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        tracked_dirty = True
    return {"git_sha": sha, "tracked_dirty": tracked_dirty}


def code_hash() -> str:
    """Hash tracked executable source while excluding results and documentation-only changes."""
    digest = hashlib.sha256()
    try:
        paths = git_output("ls-files").splitlines()
    except (OSError, subprocess.CalledProcessError):
        paths = []
    selected = []
    for relative in paths:
        path = Path(relative)
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        if path.parts and path.parts[0] in {"logs", "results", "docs"}:
            continue
        selected.append(relative)
    for relative in sorted(selected):
        candidate = ROOT / relative
        if not candidate.is_file():
            continue
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash() -> tuple[str, str]:
    relative = Path("v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv")
    candidate = ROOT / relative
    if not candidate.is_file():
        return str(relative), "missing"
    return str(relative).replace("\\", "/"), sha256_file(candidate)


def environment_metadata() -> dict:
    dataset_path, dataset_sha = dataset_hash()
    return {
        **git_state(),
        "code_sha256": code_hash(),
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha,
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": package_version("torch"),
        "torch_geometric": package_version("torch-geometric"),
        "numpy": package_version("numpy"),
        "platform": platform.platform(),
        "threads": os.environ.get("DHGCMDA_N_THREADS", str(os.cpu_count() or 1)),
        "requested_device": "cpu",
    }


def canonical_args(smoke: bool = False) -> list[str]:
    args = list(CANONICAL)
    if smoke:
        args[args.index("--epoch") + 1] = "3"
        args[args.index("--validation") + 1] = "2"
    return args


def arm_variants(arm: str) -> list[tuple[str, list[str]]]:
    if arm == "baseline":
        return [("baseline", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed"])]
    if arm == "p6":
        return [("diag", ["--predictor_mode", "diag", "--fusion_mode", "fixed"])]
    if arm == "p2":
        return [(
            "real_sequence",
            [
                "--predictor_mode", "full_bilinear",
                "--fusion_mode", "fixed",
                "--mirna_seq_sim_path", "v2.0_495m383D/M_SEQ.txt",
            ],
        )]
    if arm == "p1":
        return [("gate", ["--predictor_mode", "full_bilinear", "--fusion_mode", "gate"])]
    if arm == "p5":
        return [
            ("ce", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed", "--type_loss", "ce"]),
            ("ldam", ["--predictor_mode", "full_bilinear", "--fusion_mode", "fixed", "--type_loss", "ldam"]),
        ]
    raise ValueError(arm)


def output_roots(arm: str, run_id: str, smoke: bool) -> tuple[Path, Path]:
    if smoke:
        return Path("logs/smoke/local") / run_id, Path("results/smoke/local") / run_id
    return Path("logs/plan_n") / arm / run_id, Path("results/plan_n") / arm / run_id


def build_jobs(
    arm: str,
    python: Path,
    run_id: str,
    seeds: Iterable[int] = SEEDS,
    smoke: bool = False,
) -> list[dict]:
    jobs: list[dict] = []
    log_root, result_root = output_roots(arm, run_id, smoke)
    for variant, variant_args in arm_variants(arm):
        for seed in seeds:
            job_id = f"{variant}_s{seed}"
            args = [*canonical_args(smoke), *variant_args]
            if arm == "p5":
                dump_dir = result_root / "dumps" / job_id
                args.extend(["--dump_scores", str(dump_dir).replace("\\", "/")])
            command = [
                str(python),
                "main_experiments_hetero1.py",
                *args,
                "--seed",
                str(seed),
            ]
            jobs.append({
                "id": job_id,
                "variant": variant,
                "seed": seed,
                "command": command,
                "result": str(result_root / f"{job_id}.json").replace("\\", "/"),
                "attempts_root": str(log_root / "attempts").replace("\\", "/"),
                "status": "pending",
                "attempts": [],
            })
    return jobs


def fingerprint_payload(arm: str, seeds: Iterable[int], smoke: bool, jobs: list[dict]) -> dict:
    environment = environment_metadata()
    common_args = canonical_args(smoke)
    return {
        "arm": arm,
        "selected_seeds": list(seeds),
        "evidence_class": "screening_only" if smoke else "canonical",
        "common_args": common_args,
        "commands": [{"id": job["id"], "command": job["command"][1:]} for job in jobs],
        "environment": environment,
        "compatibility": {
            "code_sha256": environment["code_sha256"],
            "dataset_sha256": environment["dataset_sha256"],
            "python": environment["python"],
            "torch": environment["torch"],
            "torch_geometric": environment["torch_geometric"],
            "numpy": environment["numpy"],
            "threads": environment["threads"],
            "requested_device": "cpu",
            "common_args": common_args,
            "selected_seeds": list(seeds),
        },
    }


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def print_status() -> int:
    registry = load_registry()
    print("Active queue:")
    for record in sorted(
        registry["records"],
        key=lambda item: (item.get("queue_order", 999), item["id"]),
    ):
        if record.get("kind") in {"experiment", "analysis"}:
            suffix = f" - {record.get('reason', record.get('comparison', ''))}"
            print(f"  {record['id']}: {record['status'].upper()}{suffix}")
    local = next(
        (r for r in registry["records"] if r.get("id") == "local-cpu-execution-readiness"),
        None,
    )
    if local:
        progress = local.get("progress", {})
        print(
            f"Local CPU readiness: {local['status'].upper()} "
            f"{progress.get('completed', 0)}/{progress.get('total', 0)}, "
            f"next={progress.get('current', '-')}"
        )
    return 0


def check_environment(python: Path) -> int:
    checks = {
        "registry": REGISTRY_PATH,
        "status document": ROOT / "docs/status/PROJECT_STATUS.md",
        "local plan": ROOT / "docs/status/LOCAL_CPU_EXECUTION_PLAN.md",
        "validator": ROOT / "tools/validate_tracking.py",
        "training entrypoint": ROOT / "main_experiments_hetero1.py",
        "sequence matrix": ROOT / "v2.0_495m383D/M_SEQ.txt",
        "dataset": ROOT / "v2.0_495m383D/multi_all_mirna_disease_pairs_without_negative.csv",
        "python": python,
    }
    failed = False
    for label, path in checks.items():
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
        failed |= not ok
    if failed:
        return 1

    if sys.version_info[:2] != (3, 12):
        print(f"[FAIL] Python 3.12 required; running {platform.python_version()}")
        failed = True
    else:
        print(f"[OK] Python: {platform.python_version()}")

    try:
        import psutil
        import torch
        import torch_geometric

        memory = psutil.virtual_memory()
        available_ram = memory.available
        free_disk = shutil.disk_usage(ROOT).free
        print(f"[OK] torch={torch.__version__}, pyg={torch_geometric.__version__}")
        print(f"[INFO] CUDA visible to runner={torch.cuda.is_available()} (training is forced to CPU)")
        print(f"[INFO] available RAM={available_ram / 1024**3:.1f} GB")
        print(f"[INFO] free disk={free_disk / 1024**3:.1f} GB")
        if memory.total < MIN_TOTAL_RAM:
            print("[FAIL] Less than 12 GB total RAM is installed")
            failed = True
        if available_ram < MIN_AVAILABLE_RAM:
            print("[FAIL] Less than 3 GB RAM is currently available")
            failed = True
        elif available_ram < RECOMMENDED_AVAILABLE_RAM:
            print(
                "[WARN] Less than 6 GB RAM is currently available; "
                "close memory-heavy applications before a canonical run"
            )
        if free_disk < MIN_FREE_DISK:
            print("[FAIL] Less than 10 GB disk space is free")
            failed = True
    except Exception as exc:
        print(f"[FAIL] dependency/resource preflight: {exc}")
        failed = True

    print(f"[INFO] DHGCMDA_N_THREADS={os.environ.get('DHGCMDA_N_THREADS', 'unset')}")
    print(f"[INFO] PYTHONUTF8={os.environ.get('PYTHONUTF8', 'unset')}")
    validator = subprocess.run(
        [str(python), "tools/validate_tracking.py", "--root", str(ROOT)],
        cwd=ROOT,
    )
    return 1 if failed or validator.returncode else 0


def completed_job_valid(job: dict) -> tuple[bool, str]:
    result = ROOT / job["result"]
    expected = job.get("result_sha256")
    if not result.is_file():
        return False, f"missing result {result}"
    if not expected:
        return False, "missing result checksum"
    actual = sha256_file(result)
    if actual != expected:
        return False, f"result checksum mismatch for {result}"
    successful_log = job.get("successful_log")
    if not successful_log or not (ROOT / successful_log).is_file():
        return False, "missing successful log"
    return True, ""


def parse_attempt(job: dict, attempt: dict, index: dict, index_path: Path) -> bool:
    log_path = ROOT / attempt["log"]
    if not log_path.is_file():
        attempt["status"] = "failed"
        attempt["parse_error"] = "training log is missing"
        job["status"] = "failed"
        atomic_write_json(index_path, index)
        return False

    result = ROOT / job["result"]
    result.parent.mkdir(parents=True, exist_ok=True)
    temporary = result.with_name(f".{result.name}.attempt{attempt['number']:02d}.tmp")
    if temporary.exists():
        temporary.unlink()
    parse = subprocess.run(
        [
            job["command"][0],
            "parse_metrics.py",
            str(log_path.relative_to(ROOT)),
            str(temporary.relative_to(ROOT)),
        ],
        cwd=ROOT,
    )
    attempt["parse_exit_code"] = parse.returncode
    if parse.returncode != 0 or not temporary.is_file():
        attempt["status"] = "parse_pending"
        job["status"] = "parse_pending"
        atomic_write_json(index_path, index)
        return False

    os.replace(temporary, result)
    job["result_sha256"] = sha256_file(result)
    job["successful_log"] = attempt["log"]
    job["status"] = "complete"
    attempt["status"] = "complete"
    attempt["finished_at"] = utc_now()
    atomic_write_json(index_path, index)
    return True


def execute_job(job: dict, index: dict, index_path: Path) -> bool:
    attempt_number = len(job.get("attempts", [])) + 1
    attempts_root = ROOT / job["attempts_root"]
    attempts_root.mkdir(parents=True, exist_ok=True)
    log_path = attempts_root / f"{job['id']}_attempt{attempt_number:02d}.log"
    if log_path.exists():
        raise RuntimeError(f"Refusing to overwrite attempt log: {log_path}")

    attempt = {
        "number": attempt_number,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "started_at": utc_now(),
        "status": "running",
    }
    job.setdefault("attempts", []).append(attempt)
    job["status"] = "running"
    index["status"] = "running"
    index["updated_at"] = utc_now()
    atomic_write_json(index_path, index)

    print("$", " ".join(job["command"]))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with log_path.open("x", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                job["command"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                attempt["status"] = "interrupted"
                attempt["finished_at"] = utc_now()
                job["status"] = "interrupted"
                index["status"] = "interrupted"
                index["updated_at"] = utc_now()
                atomic_write_json(index_path, index)
                return False
    except OSError as exc:
        attempt["status"] = "failed"
        attempt["error"] = str(exc)
        attempt["finished_at"] = utc_now()
        job["status"] = "failed"
        atomic_write_json(index_path, index)
        return False

    attempt["training_exit_code"] = code
    attempt["finished_at"] = utc_now()
    if code != 0:
        attempt["status"] = "failed"
        job["status"] = "failed"
        atomic_write_json(index_path, index)
        return False

    attempt["status"] = "parse_pending"
    job["status"] = "parse_pending"
    atomic_write_json(index_path, index)
    return parse_attempt(job, attempt, index, index_path)


def resume_parse_if_possible(job: dict, index: dict, index_path: Path) -> bool:
    if job.get("status") != "parse_pending":
        return False
    attempts = job.get("attempts", [])
    if not attempts:
        return False
    attempt = attempts[-1]
    if attempt.get("training_exit_code") != 0:
        return False
    print(f"[resume] Retrying metrics parse without retraining: {job['id']}")
    return parse_attempt(job, attempt, index, index_path)


def create_run_index(
    arm: str,
    run_id: str,
    seeds: tuple[int, ...],
    smoke: bool,
    jobs: list[dict],
    payload: dict,
) -> dict:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "arm": arm,
        "run_id": run_id,
        "evidence_class": "screening_only" if smoke else "canonical",
        "protocol": "screening_only" if smoke else "honest_v2_fullcv",
        "selected_seeds": list(seeds),
        "fingerprint": fingerprint(payload),
        "fingerprint_payload": payload,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "jobs": jobs,
    }


def run_arm(
    arm: str,
    python: Path,
    run_id: str,
    dry_run: bool,
    seeds: tuple[int, ...],
    resume: bool,
    smoke: bool = False,
) -> int:
    if arm == "p3":
        print(
            "P3 is BLOCKED: softmax type outputs are incompatible with genuine multi-label BCE.",
            file=sys.stderr,
        )
        return 2
    if arm not in RUNNABLE_ARMS:
        print(f"Unknown arm {arm}", file=sys.stderr)
        return 2
    if not smoke and not dry_run and git_state()["tracked_dirty"]:
        print(
            "Canonical runs require a clean tracked worktree. Commit or stash code/document changes first.",
            file=sys.stderr,
        )
        return 2

    jobs = build_jobs(arm, python, run_id, seeds=seeds, smoke=smoke)
    payload = fingerprint_payload(arm, seeds, smoke, jobs)
    log_root, result_root = output_roots(arm, run_id, smoke)
    index_path = ROOT / result_root / "run_index.json"

    if dry_run:
        for job in jobs:
            print("$", " ".join(job["command"]))
            print(f"  attempts: {job['attempts_root']}/{job['id']}_attemptNN.log")
            print(f"  result: {job['result']}")
        return 0

    if index_path.exists():
        if not resume:
            print(
                f"Refusing existing run directory without --resume: {ROOT / result_root}",
                file=sys.stderr,
            )
            return 2
        index = json.loads(index_path.read_text(encoding="utf-8"))
        current_fingerprint = fingerprint(payload)
        if index.get("fingerprint") != current_fingerprint:
            print("Resume fingerprint mismatch; refusing to mix incompatible work.", file=sys.stderr)
            return 2
        print(f"[resume] Loaded {index_path.relative_to(ROOT)}")
    else:
        if resume:
            print(f"--resume requires an existing run index: {index_path}", file=sys.stderr)
            return 2
        (ROOT / log_root).mkdir(parents=True, exist_ok=True)
        (ROOT / result_root).mkdir(parents=True, exist_ok=True)
        index = create_run_index(arm, run_id, seeds, smoke, jobs, payload)
        atomic_write_json(index_path, index)

    succeeded = True
    for job in index["jobs"]:
        if job.get("status") == "complete":
            valid, reason = completed_job_valid(job)
            if not valid:
                print(f"Completed job is corrupt: {job['id']}: {reason}", file=sys.stderr)
                succeeded = False
                break
            print(f"[resume] Skipping completed job: {job['id']}")
            continue
        if resume_parse_if_possible(job, index, index_path):
            continue
        if not execute_job(job, index, index_path):
            succeeded = False
            break

    index["finished_at"] = utc_now()
    index["updated_at"] = utc_now()
    index["status"] = "complete" if succeeded else "failed"
    atomic_write_json(index_path, index)
    return 0 if succeeded else 1


def load_complete_index(arm: str, run_id: str) -> tuple[dict, Path]:
    index_path = ROOT / "results" / "plan_n" / arm / run_id / "run_index.json"
    if not index_path.is_file():
        raise ValueError(f"Run index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "complete":
        raise ValueError(f"Run is not complete: {arm}/{run_id} ({index.get('status')})")
    if tuple(index.get("selected_seeds", [])) != SEEDS:
        raise ValueError(f"Summary requires seeds {SEEDS}; got {index.get('selected_seeds')}")
    for job in index.get("jobs", []):
        valid, reason = completed_job_valid(job)
        if job.get("status") != "complete" or not valid:
            raise ValueError(f"Invalid completed job {job.get('id')}: {reason}")
    return index, index_path


def compatible(control: dict, candidate: dict) -> tuple[bool, list[str]]:
    left = control.get("fingerprint_payload", {}).get("compatibility", {})
    right = candidate.get("fingerprint_payload", {}).get("compatibility", {})
    mismatches = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            mismatches.append(key)
    return not mismatches, mismatches


def collect_condition(index: dict, variant: str) -> dict:
    from parse_metrics import _read_log
    from summarize_stats import parse_log_perfold

    auc: list[float] = []
    top1: list[float] = []
    aupr: list[float] = []
    logs: list[str] = []
    jobs = sorted(
        (job for job in index["jobs"] if job["variant"] == variant),
        key=lambda job: SEEDS.index(int(job["seed"])),
    )
    if len(jobs) != len(SEEDS):
        raise ValueError(f"{variant}: expected {len(SEEDS)} jobs, found {len(jobs)}")
    for job in jobs:
        log = ROOT / job["successful_log"]
        values = parse_log_perfold(log)
        if len(values["auc"]) != 5 or len(values["top1_f1"]) != 5:
            raise ValueError(f"{job['id']}: expected five completed folds")
        auc.extend(values["auc"])
        top1.extend(values["top1_f1"])
        metrics = json.loads((ROOT / job["result"]).read_text(encoding="utf-8"))
        if metrics.get("AUPR") is None:
            raise ValueError(f"{job['id']}: missing AUPR")
        aupr.append(float(metrics["AUPR"]))
        logs.append(str(log.relative_to(ROOT)).replace("\\", "/"))
        if "FINAL COMPREHENSIVE RESULTS" not in _read_log(log):
            raise ValueError(f"{job['id']}: final results marker is missing")
    return {"auc": auc, "top1_f1": top1, "aupr_seed": aupr, "logs": logs}


def p1_fusion_summary(index: dict) -> dict:
    from parse_metrics import _read_log

    modalities = {"mirna": [], "disease": []}
    for job in sorted(index["jobs"], key=lambda item: SEEDS.index(int(item["seed"]))):
        text = _read_log(ROOT / job["successful_log"])
        for line in text.splitlines():
            if "[P1 fusion]" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            left = _GATE_RE.search(parts[0])
            right = _GATE_RE.search(parts[1])
            if left:
                modalities["mirna"].append(float(left.group(1)))
            if right:
                modalities["disease"].append(float(right.group(1)))
    boundary = {
        key: sum(value <= 0.05 or value >= 0.95 for value in values)
        for key, values in modalities.items()
    }
    collapsed = any(count >= 8 for count in boundary.values())
    return {"weights": modalities, "boundary_counts": boundary, "collapsed": collapsed}


def p5_recall_summary(index: dict) -> dict:
    import numpy as np

    output: dict[str, dict] = {}
    for variant in ("ce", "ldam"):
        true_all = []
        pred_all = []
        for job in index["jobs"]:
            if job["variant"] != variant:
                continue
            dump_dir = ROOT / Path(job["result"]).parent / "dumps" / job["id"]
            files = sorted(dump_dir.glob("fold*.npz"))
            if len(files) != 5:
                raise ValueError(f"{job['id']}: expected five score dumps, found {len(files)}")
            for file in files:
                with np.load(file) as data:
                    true = data["true_type"].astype(int)
                    probs = data["type_probs"]
                    pred = probs.argmax(axis=1) + 1
                    true_all.append(true)
                    pred_all.append(pred)
        y_true = np.concatenate(true_all)
        y_pred = np.concatenate(pred_all)
        recalls = {}
        for class_id in range(1, 5):
            mask = y_true == class_id
            recalls[str(class_id)] = float((y_pred[mask] == class_id).mean()) if mask.any() else None
        minority_mean = float((recalls["2"] + recalls["3"]) / 2.0)
        output[variant] = {
            "per_class_recall": recalls,
            "minority_mean_recall": minority_mean,
            "n": int(len(y_true)),
        }
    output["minority_mean_delta_ldam_minus_ce"] = (
        output["ldam"]["minority_mean_recall"] - output["ce"]["minority_mean_recall"]
    )
    return output


def summarize_arm(arm: str, run_id: str, baseline_run_id: str | None) -> int:
    from summarize_stats import describe, holm_correct, paired_tests

    if arm not in RUNNABLE_ARMS:
        print("summarize supports baseline, p6, p2, p1 and p5", file=sys.stderr)
        return 2
    candidate, candidate_index_path = load_complete_index(arm, run_id)
    if arm == "baseline":
        baseline = collect_condition(candidate, "baseline")
        auc = describe(baseline["auc"])
        top1 = describe(baseline["top1_f1"])
        auc_delta = float(auc["mean"]) - 0.9361
        top1_delta = float(top1["mean"]) - 0.6151
        accepted = abs(auc_delta) <= 0.01 and abs(top1_delta) <= 0.03
        report = {
            "arm": arm,
            "run_id": run_id,
            "created_at": utc_now(),
            "sources": {
                "candidate_index": str(
                    candidate_index_path.relative_to(ROOT)
                ).replace("\\", "/"),
                "verified_reference": "v2-honest-k2-full-bilinear",
            },
            "conditions": {
                "baseline": {
                    "auc": auc,
                    "top1_f1": top1,
                    "aupr_seed": describe(baseline["aupr_seed"]),
                    "logs": baseline["logs"],
                }
            },
            "reference": {
                "auc_fold_mean": 0.9361,
                "top1_f1_fold_mean": 0.6151,
                "auc_mean_delta": auc_delta,
                "top1_f1_mean_delta": top1_delta,
                "acceptance": {
                    "max_abs_auc_delta": 0.01,
                    "max_abs_top1_f1_delta": 0.03,
                },
            },
            "gate": {
                "decision": "continue" if accepted else "pause_for_anchor_audit"
            },
        }
        summary_path = candidate_index_path.parent / "summary.json"
        atomic_write_json(summary_path, report)
        print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
        print(f"Saved summary: {summary_path.relative_to(ROOT)}")
        return 0

    labels: tuple[str, str]
    conditions: dict[str, dict]
    if arm == "p5":
        labels = ("ce", "ldam")
        conditions = {
            "ce": collect_condition(candidate, "ce"),
            "ldam": collect_condition(candidate, "ldam"),
        }
        control_index_path = None
    else:
        if not baseline_run_id:
            print(f"{arm} requires --baseline-run-id", file=sys.stderr)
            return 2
        control, control_index_path = load_complete_index("baseline", baseline_run_id)
        is_compatible, mismatches = compatible(control, candidate)
        if not is_compatible:
            print(
                f"Baseline/candidate compatibility mismatch: {', '.join(mismatches)}",
                file=sys.stderr,
            )
            return 2
        variant = {"p6": "diag", "p2": "real_sequence", "p1": "gate"}[arm]
        labels = ("baseline", variant)
        conditions = {
            "baseline": collect_condition(control, "baseline"),
            variant: collect_condition(candidate, variant),
        }

    report = {
        "arm": arm,
        "run_id": run_id,
        "baseline_run_id": baseline_run_id,
        "created_at": utc_now(),
        "sources": {
            "candidate_index": str(candidate_index_path.relative_to(ROOT)).replace("\\", "/"),
            "control_index": (
                str(control_index_path.relative_to(ROOT)).replace("\\", "/")
                if control_index_path else None
            ),
        },
        "conditions": {},
        "comparisons": {},
    }
    for label, data in conditions.items():
        report["conditions"][label] = {
            "auc": describe(data["auc"]),
            "top1_f1": describe(data["top1_f1"]),
            "aupr_seed": describe(data["aupr_seed"]),
            "logs": data["logs"],
        }

    p_values: list[float] = []
    p_keys: list[str] = []
    candidate_label = labels[1]
    control_label = labels[0]
    for metric in ("auc", "top1_f1", "aupr_seed"):
        result = paired_tests(
            conditions[candidate_label][metric],
            conditions[control_label][metric],
        )
        key = f"{candidate_label}_minus_{control_label}::{metric}"
        report["comparisons"][key] = result
        paired_t = result.get("paired_t", {})
        if "p" in paired_t:
            p_values.append(float(paired_t["p"]))
            p_keys.append(key)
    if p_values:
        for key, adjusted in zip(p_keys, holm_correct(p_values)):
            report["comparisons"][key]["paired_t"]["p_holm"] = float(adjusted)

    top1_key = f"{candidate_label}_minus_{control_label}::top1_f1"
    auc_key = f"{candidate_label}_minus_{control_label}::auc"
    top1 = report["comparisons"][top1_key]
    auc = report["comparisons"][auc_key]
    top1_delta = float(top1["mean_diff"])
    auc_delta = float(auc["mean_diff"])
    top1_holm = float(top1.get("paired_t", {}).get("p_holm", 1.0))
    auc_holm = float(auc.get("paired_t", {}).get("p_holm", 1.0))

    if arm == "p6":
        pause = (
            (top1_delta >= 0.01 and top1_holm < 0.05)
            or (auc_delta >= 0.005 and auc_holm < 0.05)
        )
        report["gate"] = {
            "decision": "pause_downstream_and_review_baseline" if pause else "continue",
            "top1_delta_diag_minus_baseline": top1_delta,
            "auc_delta_diag_minus_baseline": auc_delta,
        }
    elif arm == "p2":
        adopt = top1_delta > 0 and top1_holm < 0.05 and auc_delta >= -0.005
        report["gate"] = {"decision": "adopt" if adopt else "do_not_adopt"}
    elif arm == "p1":
        fusion = p1_fusion_summary(candidate)
        adopt = (
            top1_delta >= 0.01
            and top1_holm < 0.05
            and auc_delta >= -0.005
            and not fusion["collapsed"]
        )
        report["fusion"] = fusion
        report["gate"] = {
            "decision": "open_attention_gate" if adopt else "keep_attention_on_hold"
        }
    elif arm == "p5":
        recall = p5_recall_summary(candidate)
        adopt = (
            recall["minority_mean_delta_ldam_minus_ce"] >= 0.02
            and top1_delta >= -0.01
            and auc_delta >= -0.005
        )
        report["per_class"] = recall
        report["gate"] = {"decision": "prefer_ldam" if adopt else "keep_ce"}

    summary_path = candidate_index_path.parent / "summary.json"
    atomic_write_json(summary_path, report)
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=default_python())
    parser.add_argument("command", choices=["status", "check", "dry-run", "smoke", "run", "summarize"])
    parser.add_argument("arm", nargs="?", choices=sorted(ALL_ARMS))
    parser.add_argument("--run-id")
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    python = args.python.resolve()

    if args.command == "status":
        return print_status()
    if args.command == "check":
        return check_environment(python)
    if args.arm is None:
        parser.error("dry-run, smoke, run and summarize require an arm")

    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seeds = tuple(dict.fromkeys(args.seeds or SEEDS))
    if args.command == "summarize":
        return summarize_arm(args.arm, run_id, args.baseline_run_id)
    if args.command == "smoke":
        if args.arm != "baseline":
            parser.error("smoke currently supports only the baseline arm")
        return run_arm(
            args.arm,
            python,
            run_id,
            dry_run=False,
            seeds=(0,),
            resume=args.resume,
            smoke=True,
        )
    return run_arm(
        args.arm,
        python,
        run_id,
        dry_run=args.command == "dry-run",
        seeds=seeds,
        resume=args.resume,
        smoke=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
