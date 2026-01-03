#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Install project + dev + viz extras (viz is optional, but harmless to include)
python -m pip install -e ".[dev,viz]"

echo "✅ Environment ready. Activate with: source $VENV_DIR/bin/activate"
