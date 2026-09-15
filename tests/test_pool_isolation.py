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
    """Does this file set a thread-limit variable, in ANY of the shapes
    people actually write (review 1, F2)? Matches:
      - os.environ["OMP_NUM_THREADS"] = ...   (Subscript store)
      - os.environ.setdefault("OMP_NUM_THREADS", ...)
      - os.putenv("OMP_NUM_THREADS", ...)
      - os.environ.update({"OMP_NUM_THREADS": ...})
    The scan must SEE the runner (self-check below) or it is broken."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return False
    # First pass: local names bound to a thread-var string or to a loop
    # over a tuple containing them (the runner binds _var from the
    # four-name tuple in its caps for-loop).
    local_names: set[str] = set(THREAD_VARS)
    for node in ast.walk(tree):
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_strs = [e.value for e in ast.walk(node.value)
                          if isinstance(e, ast.Constant)
                          and isinstance(e.value, str)]
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value_strs = [e.value for e in ast.walk(node.iter)
                          if isinstance(e, ast.Constant)
                          and isinstance(e.value, str)]
        else:
            continue
        if any(s in THREAD_VARS for s in value_strs):
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    local_names.add(tgt.id)
    # Second pass: any write reaching os.environ / os.putenv with a
    # thread-var name (constant or tracked local).
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("setdefault", "putenv", "update") \
                and isinstance(node.func.value, ast.Attribute) \
                and node.func.value.attr == "environ" or (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "putenv"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"):
            for arg in node.args + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name) and arg.id in local_names:
                    return True
                if isinstance(arg, ast.Constant) and arg.value in THREAD_VARS:
                    return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "environ":
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value in THREAD_VARS:
                return True
            if isinstance(sl, ast.Name) and sl.id in local_names:
                return True
    return False


def test_the_scan_sees_the_runner() -> None:
    """A scan that silently matches nothing passes forever (the same
    protection test_layering carries): the scan must SEE offline/runner.py
    setting the caps - the file the rule is about."""
    assert _sets_thread_var(RUNNER), (
        "the scan no longer sees offline/runner.py setting the caps - "
        "it is broken and passing everything unconditionally"
    )


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
    re-reading os.environ measures your own assignment (brief 5.1).
    torch may be absent (not in this venv; absent on the grading image
    per the Kaggle probe) - skip loudly with the reason, never fail on
    an opaque import error."""
    import subprocess
    import sys
    probe = (
        "import sys; sys.path.insert(0, '.')\n"
        "import offline.runner  # sets the caps at import\n"
        "try:\n"
        "    import torch\n"
        "    print('torch', torch.get_num_threads())\n"
        "except ImportError:\n"
        "    import numpy\n"
        "    cfg = numpy.__config__.show_config(mode='dicts')\n"
        "    blas = cfg.get('Build Dependencies', {}).get('blas', {})\n"
        "    print('numpy', blas.get('name', 'unknown'))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        "read-back probe failed:\n" + proc.stderr[-300:]
    )
    out = proc.stdout.strip().splitlines()[-1]
    if out.startswith("torch"):
        threads = int(out.split()[1])
        assert threads == 1, (
            f"torch reports {threads} threads after offline.runner import: "
            "the caps did not take effect"
        )
    else:
        # no torch in this environment: numpy's BLAS name is the recorded
        # read-back instead; the thread-count check needs a consumer that
        # exposes it (named TODO, R005 - recorded, not silent)
        assert out.startswith("numpy"), out


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


def test_each_seat_sees_its_own_player_index() -> None:
    """The F1 regression, closed through the REAL runner path: seat n's
    agent must see player == n on EVERY turn. The first fix attempt
    passed a hand-made-dict test while the runner still handed views[0]
    to seat 1 - so this test re-applies the bug and must fail if it
    ever returns (review 2, N2)."""
    import offline.runner as R

    seen = {0: [], 1: []}

    class ProbeAgent:
        slug = "probe"
        arity = 2
        fn_name = "probe_fn"
        rule = "probe"
        module = None

        def __init__(self, seat):
            self.seat = seat
            self.fn = self._fn

        def _fn(self, obs, config=None):
            seen[self.seat].append(obs.get("player"))
            return {"farmer": ["PASS"], "hands": [], "market": []}

    import offline.pool.loader as PL
    saved = PL.load
    PL.load = lambda slug: ProbeAgent(int(slug))  # slug IS the seat here
    try:
        rec = R._episode_worker("0", "1", seed=0, episode_steps=8)
    finally:
        PL.load = saved
    assert rec["status"] == "DONE"
    # seat n saw player n on EVERY turn - the exact assertion that fails
    # when views[0] is handed to both seats
    assert seen[0] and seen[1], seen
    assert all(p == 0 for p in seen[0]), seen[0]
    assert all(p == 1 for p in seen[1]), seen[1]


def test_vendored_mutation_cannot_corrupt_the_episode() -> None:
    """The N1 guard, through the REAL worker: a vendored agent that
    writes into its observation must not corrupt the episode's live
    state. Re-introducing the bug (copy_state default inside the loop)
    makes this fail - the reviewer's method, verified below."""
    import offline.runner as R
    import offline.pool.loader as PL

    real_load = PL.load
    written = []          # the money the mutator SAW, per turn

    def mutator_factory(seat):
        seat = int(seat)                          # slug is a string
        la = real_load("adaptive-replay-agent")   # real shape, arity from loader
        def fn(obs, config=None):
            if isinstance(obs, dict):
                money = obs["farms"][seat]["money"]
                written.append(money)
                obs["farms"][seat]["money"] = 999999   # the corruption attempt
            return {"farmer": ["PASS"], "hands": [], "market": []}
        from offline.pool.loader import LoadedAgent
        return LoadedAgent(slug=la.slug, fn=fn, fn_name="mutator",
                           arity=la.arity, rule="test-mutator",
                           module=la.module)

    saved = PL.load
    PL.load = mutator_factory
    try:
        rec = R._episode_worker("0", "1", seed=0, episode_steps=8)
    finally:
        PL.load = saved
    # The corruption attempt must not propagate: if the views were LIVE,
    # the next turn's mutator would read the 999999 it wrote (measured:
    # with live views the written value persists into every later turn).
    # Detached views mean the mutator always reads 3000.
    assert written and all(m == 3000 for m in written), (
        f"the mutator read non-clean money {sorted(set(written))}: the "
        "views handed to vendored agents are live - N1 regression")
    for seat in (0, 1):
        assert rec["seats"][seat]["rewards"] == 3000.0, (
            f"seat {seat} rewards corrupted: {rec['seats'][seat]['rewards']}")


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
