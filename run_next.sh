#!/usr/bin/env bash
# Canonical Linux entrypoint for the active experiment queue.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT/venv/bin/python"
exec "$PYTHON_BIN" "$ROOT/run_next.py" --python "$PYTHON_BIN" "$@"
