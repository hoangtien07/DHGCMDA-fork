from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_next


class FakeFailedProcess:
    def __init__(self):
        self.stdout = io.StringIO("training failed\n")

    def wait(self, timeout=None):
        return 1

    def terminate(self):
        return None

    def kill(self):
        return None


class RunNextTests(unittest.TestCase):
    def test_baseline_seed_filter(self):
        jobs = run_next.build_jobs(
            "baseline",
            Path("python"),
            "run-id",
            seeds=(42,),
        )
        self.assertEqual([job["id"] for job in jobs], ["baseline_s42"])
        command = jobs[0]["command"]
        self.assertIn("full_bilinear", command)
        self.assertEqual(command[-2:], ["--seed", "42"])

    def test_p5_default_builds_six_jobs_and_score_dumps(self):
        jobs = run_next.build_jobs("p5", Path("python"), "run-id")
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job["variant"] for job in jobs}, {"ce", "ldam"})
        for job in jobs:
            self.assertIn("--dump_scores", job["command"])

    def test_smoke_overrides_epoch_and_validation(self):
        jobs = run_next.build_jobs(
            "baseline",
            Path("python"),
            "smoke-id",
            seeds=(0,),
            smoke=True,
        )
        command = jobs[0]["command"]
        self.assertEqual(command[command.index("--epoch") + 1], "3")
        self.assertEqual(command[command.index("--validation") + 1], "2")
        self.assertTrue(jobs[0]["result"].startswith("results/smoke/local/"))

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "index.json"
            run_next.atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

    def test_completed_job_checksum_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result.json"
            log = root / "attempt.log"
            result.write_text('{"AUPR": 0.5}', encoding="utf-8")
            log.write_text("FINAL COMPREHENSIVE RESULTS", encoding="utf-8")
            digest = hashlib.sha256(result.read_bytes()).hexdigest()
            job = {
                "result": "result.json",
                "result_sha256": digest,
                "successful_log": "attempt.log",
            }
            with mock.patch.object(run_next, "ROOT", root):
                self.assertEqual(run_next.completed_job_valid(job), (True, ""))
                job["result_sha256"] = "bad"
                valid, reason = run_next.completed_job_valid(job)
                self.assertFalse(valid)
                self.assertIn("checksum", reason)

    def test_failed_attempts_get_unique_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "results" / "run_index.json"
            job = {
                "id": "baseline_s0",
                "variant": "baseline",
                "seed": 0,
                "command": [sys.executable, "fake.py"],
                "result": "results/baseline_s0.json",
                "attempts_root": "logs/attempts",
                "status": "pending",
                "attempts": [],
            }
            index = {"status": "running", "jobs": [job]}
            with (
                mock.patch.object(run_next, "ROOT", root),
                mock.patch.object(
                    run_next.subprocess,
                    "Popen",
                    side_effect=[FakeFailedProcess(), FakeFailedProcess()],
                ),
            ):
                self.assertFalse(run_next.execute_job(job, index, index_path))
                self.assertFalse(run_next.execute_job(job, index, index_path))
            self.assertEqual(len(job["attempts"]), 2)
            logs = [root / attempt["log"] for attempt in job["attempts"]]
            self.assertNotEqual(logs[0], logs[1])
            self.assertTrue(all(path.is_file() for path in logs))

    def test_compatibility_uses_scientific_environment_not_git_commit(self):
        common = {
            "code_sha256": "code",
            "dataset_sha256": "data",
            "python": "3.12.10",
            "torch": "2.5.1+cpu",
            "torch_geometric": "2.7.0",
            "numpy": "2.4.3",
            "threads": "10",
            "requested_device": "cpu",
            "common_args": ["--device", "cpu"],
            "selected_seeds": [0, 42, 1234],
        }
        left = {"fingerprint_payload": {"compatibility": common}}
        right = {"fingerprint_payload": {"compatibility": dict(common)}}
        self.assertEqual(run_next.compatible(left, right), (True, []))
        right["fingerprint_payload"]["compatibility"]["threads"] = "12"
        ok, mismatches = run_next.compatible(left, right)
        self.assertFalse(ok)
        self.assertEqual(mismatches, ["threads"])

    def test_collect_condition_requires_three_complete_seed_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for seed in run_next.SEEDS:
                log = root / f"s{seed}.log"
                lines = [
                    f"Fold {fold} completed - AUC: 0.93, Top-1 F1: 0.61"
                    for fold in range(1, 6)
                ]
                lines.append("FINAL COMPREHENSIVE RESULTS")
                log.write_text("\n".join(lines), encoding="utf-8")
                result = root / f"s{seed}.json"
                result.write_text('{"AUPR": 0.92}', encoding="utf-8")
                jobs.append({
                    "id": f"baseline_s{seed}",
                    "variant": "baseline",
                    "seed": seed,
                    "successful_log": log.name,
                    "result": result.name,
                })
            with mock.patch.object(run_next, "ROOT", root):
                condition = run_next.collect_condition({"jobs": jobs}, "baseline")
            self.assertEqual(len(condition["auc"]), 15)
            self.assertEqual(len(condition["top1_f1"]), 15)
            self.assertEqual(len(condition["aupr_seed"]), 3)


if __name__ == "__main__":
    unittest.main()
