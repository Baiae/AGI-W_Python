"""Smoke tests for the CLI entrypoint."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vervaeke_agent", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
