#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source .venv/bin/activate

python -m vervaeke_agent "$@"
