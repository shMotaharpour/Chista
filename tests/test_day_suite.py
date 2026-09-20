"""The day layer, run under Chista's one-command convention.

Run:  .venv/bin/python -m tests.test_day

`wsr/` turns the planner's chains into the ops a worker-day is made of: a chain
becomes a task array, a beam search decides which worker does what and when, and
the compiler writes the ops. The suite in `tests/day_layer/` replays whole days
against the real harness, because the engine refuses a bad op in silence.

That suite is pytest-shaped. This module is the adapter so `tests/` remains one
command per module: it drives pytest and re-reports in the PASS/FAIL form every
other module here uses.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "tests" / "day_layer"


def main() -> int:
    if not SUITE.is_dir():
        print(f"FAIL day suite: {SUITE} missing")
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
        print(f"PASS day solvers: {summary}")
        return 0
    print(f"FAIL day solvers: {summary}")
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            print(f"  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
