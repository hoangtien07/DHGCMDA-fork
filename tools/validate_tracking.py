#!/usr/bin/env python3
"""Validate the lightweight project-status registry without third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ALLOWED = {"verified", "provisional", "superseded", "ready", "blocked", "hold", "legacy", "legacy_reference"}
REQUIRED = {"schema_version", "source_of_truth", "protocols", "records", "reports"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    registry_path = root / "docs" / "status" / "registry.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    missing = REQUIRED - set(data)
    if missing:
        errors.append(f"registry missing keys: {sorted(missing)}")
    if data.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if not (root / data.get("source_of_truth", "")).is_file():
        errors.append("source_of_truth path is missing")

    ids: set[str] = set()
    ready_arms: set[str] = set()
    for record in data.get("records", []):
        record_id = record.get("id")
        if not record_id or record_id in ids:
            errors.append(f"duplicate or missing record id: {record_id!r}")
        ids.add(record_id)
        status = record.get("status")
        if status not in ALLOWED:
            errors.append(f"{record_id}: invalid status {status!r}")
        protocol = record.get("protocol")
        if protocol and protocol not in data.get("protocols", {}) and protocol != "v32_correct_metric":
            errors.append(f"{record_id}: unknown protocol {protocol!r}")
        for artifact in record.get("artifacts", []):
            if not (root / artifact).exists():
                errors.append(f"{record_id}: missing artifact {artifact}")
        progress = record.get("progress")
        if progress is not None:
            milestones = progress.get("milestones", [])
            milestone_ids = [item.get("id") for item in milestones]
            if not milestones:
                errors.append(f"{record_id}: progress has no milestones")
            if len(milestone_ids) != len(set(milestone_ids)):
                errors.append(f"{record_id}: duplicate progress milestone id")
            allowed_milestone_status = {"complete", "pending", "hold", "blocked"}
            for milestone in milestones:
                if milestone.get("status") not in allowed_milestone_status:
                    errors.append(
                        f"{record_id}: invalid milestone status "
                        f"{milestone.get('id')}={milestone.get('status')!r}"
                    )
            completed = sum(item.get("status") == "complete" for item in milestones)
            if progress.get("completed") != completed:
                errors.append(
                    f"{record_id}: progress completed={progress.get('completed')} "
                    f"but milestone count is {completed}"
                )
            if progress.get("total") != len(milestones):
                errors.append(
                    f"{record_id}: progress total={progress.get('total')} "
                    f"but milestone count is {len(milestones)}"
                )
            current = progress.get("current")
            if current and current not in milestone_ids:
                errors.append(f"{record_id}: current milestone {current!r} is missing")
        if status == "ready":
            arm = record.get("runner_arm")
            if not arm:
                errors.append(f"{record_id}: ready record has no runner_arm")
            else:
                ready_arms.add(arm)

    for report in data.get("reports", []):
        if report.get("status") not in ALLOWED:
            errors.append(f"report has invalid status: {report}")
        if not (root / report.get("path", "")).exists():
            errors.append(f"missing report artifact: {report.get('path')}")

    manifest = root / "docs/archive/reports/2026-07-07/MANIFEST.sha256"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            candidate = manifest.parent / relative.strip()
            if not candidate.is_file():
                errors.append(f"report bundle manifest missing file: {relative}")
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"report bundle checksum mismatch: {relative}")

    runner_path = root / "run_next.py"
    if ready_arms and not runner_path.is_file():
        errors.append("ready arms exist but run_next.py is missing")

    if errors:
        print("Tracking registry validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Tracking registry valid: {len(ids)} records, ready arms={','.join(sorted(ready_arms))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
