#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "❌ $VENV_DIR not found. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ruff format .
ruff check .
.venv/bin/python -m pytest -q || rc=$?
if [ "${rc:-0}" -eq 5 ]; then
  echo "⚠️  pytest: no tests collected (ok for now)"
  rc=0
fi
mypy .
exit "${rc:-0}"
