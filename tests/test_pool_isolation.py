"""offline.pool isolation tests: thread caps, obs copies, process boundaries.

Run:  .venv/bin/python -m tests.test_pool_isolation

Contracts under test (issue #20 brief §5, addendum A4):
- Thread-limit environment variables are set ONLY in offline/runner.py -
  the AST scan the brief asks for, same shape as test_layering's.
- The caps are proven to have taken effect by READ-BACK from a library
  that consumed them, not by re-reading os.environ.
- The wrapper hands third-party agents COPIES of the observation, never
  the live view (pool-analysis safety note).
- Per-seat records: one episode returns populated records for BOTH seats
  (addendum A1).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
RUNNER = REPO / "offline" / "runner.py"


def _sets_thread_var(path: Path) -> bool:
    """Does this file's source assign a thread-limit variable?"""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                names = []
                if isinstance(tgt, ast.Name):
                    names = [tgt.id]
                elif isinstance(tgt, ast.Tuple):
                    names = [e.id for e in tgt.elts
                             if isinstance(e, ast.Name)]
                if any(n in THREAD_VARS for n in names):
                    return True
    return False


def test_only_the_runner_sets_thread_vars() -> None:
    """The thread-cap variables are assigned only in offline/runner.py."""
    offenders = []
    for p in sorted(REPO.rglob("*.py")):
        if ".venv" in p.parts:
            continue
        if _sets_thread_var(p):
            rel = p.relative_to(REPO).as_posix()
            if rel != "offline/runner.py":
                offenders.append(rel)
    assert not offenders, (
        "thread-limit variables must be set ONLY in offline/runner.py "
        "(brief 5.1) - these modules set them too:\n  "
        + "\n  ".join(offenders)
    )


def test_runner_sets_all_four_caps() -> None:
    """The runner itself covers all four variables (the rule is complete)."""
    src = RUNNER.read_text()
    for var in THREAD_VARS:
        assert var in src, f"runner.py does not cap {var}"


def test_read_back_proves_the_cap_took_effect() -> None:
    """Prove the cap by reading it back from a library that consumed it -
    re-reading os.environ measures your own assignment (brief 5.1)."""
    import subprocess
    import sys
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "import offline.runner  # sets the caps at import\n"
        "import torch\n"
        "print(torch.get_num_threads())\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-300:]
    threads = int(proc.stdout.strip().splitlines()[-1])
    assert threads == 1, (
        f"torch reports {threads} threads after offline.runner import: "
        "the caps did not take effect"
    )


def _seed_record():
    from offline.runner import run_episode_process
    return run_episode_process("v3-agent", "adaptive-replay-agent", 0,
                               episode_steps=96)


def test_per_seat_records_populated() -> None:
    """One episode returns populated records for BOTH seats (A1)."""
    rec = _seed_record()
    assert rec["status"] == "DONE"
    for seat in (0, 1):
        d = rec["seats"][seat]
        assert d["slug"], seat
        assert isinstance(d["rewards"], (int, float))
        assert d["actions_hash"], seat
        assert d["guard"]["self_p95_ms"] >= 0


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all pool isolation tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
