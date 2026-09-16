"""The ported WRS solvers, run under Chista's one-command convention.

Run:  .venv/bin/python -m tests.test_secretary

`secretary/` carries the routing and workforce solvers ported from
ChistaWRS (issue #14): `oxa_solver` is the RUNTIME one — pure Python,
no third-party imports — and `cpsat_solver` is the OFFLINE exact
oracle, which imports `ortools` and therefore must never enter the
submission's import closure. `tests/test_layering.py` enforces that.

Their own suite is pytest-shaped and stays that way (it is a faithful
port, not a rewrite — see `docs/F052_wrs-solvers-ported.md`). This
module is the adapter so `tests/` remains one command per module: it
drives pytest and re-reports in the PASS/FAIL form every other module
here uses.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "tests" / "wrs"


def main() -> int:
    if not SUITE.is_dir():
        print(f"FAIL secretary suite: {SUITE} missing")
        return 1
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(SUITE), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
    summary = tail[0] if tail else "(no pytest output)"
    if proc.returncode == 0:
        print(f"PASS secretary solvers: {summary}")
        return 0
    print(f"FAIL secretary solvers: {summary}")
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            print(f"  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
