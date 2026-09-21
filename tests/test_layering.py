"""The submission never imports the competitors.

Run:  .venv/bin/python -m tests.test_layering

`opponents/` is third-party code vendored for evaluation. It is input to the
arena, never to the agent: a submission that imports it would ship someone
else's published work, and on Kaggle it would not be there to import anyway.

The rule this file states (issue #20 review round): **nothing the submission
loads may import `opponents`** — stated as the transitive in-repo import
closure of the submission entry point (`agent/main.py`), not as a per-file
scan with an exemption list. The closure is strictly stronger: it catches an
indirect import through a helper, and it needs no list that grows until the
guard means nothing. `offline/` (the arena, issue #20) imports `opponents`
by design and is outside the submission's closure by construction.

Written before `agent/` exists, deliberately. A guard added after the first
violation is a cleanup; added before, it is a contract — and the cost of
discovering this one late is a submission that fails to load.
"""
from __future__ import annotations

import ast
import contextlib
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "agent" / "main.py"
FORBIDDEN_ROOT = "opponents"
# Third-party packages the submission may never import. `ortools` is what an
# offline exact oracle pulls; the day layer runs a beam search on numpy alone,
# so the closure must never reach it.
FORBIDDEN_PACKAGES = ("ortools",)


def _imported_modules(path: Path, strict: bool = False) -> set[str]:
    """Full dotted names imported by one file (import x / from x import y).

    `strict` is for the CLOSURE WALK only. There, a relative import
    (`from .greedy import ...`, node.level > 0) must raise: the walk cannot
    resolve it (level means "up N packages", which needs the importing
    package's position at runtime), and an unresolved import silently
    shrinks the closure — a weaker guard that still reports success.

    Everywhere else relative imports are skipped, not refused. The repo-wide
    scan only asks which top-level package a file reaches, and a relative
    import cannot reach a sibling top-level package. Raising there made a
    perfectly ordinary `from .tile_state import ...` in a package __init__
    fail the layering suite, with a message calling it part of the
    submission tree when it was not in the closure at all (review round 2).
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if strict:
                    raise ValueError(
                        f"{path}: relative import (level {node.level}) inside "
                        "the submission's import closure — the closure walk "
                        "cannot see it, so it would silently shrink the "
                        "closure; use an absolute import here")
                continue
            if node.module:
                names.add(node.module)
    return names


def _module_file(name: str, repo: Path = REPO) -> Path | None:
    """The in-repo file a dotted name maps to (first segment = top package)."""
    top = name.split(".")[0]
    if not (repo / top).is_dir():
        return None
    parts = name.split(".")
    candidate = repo.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = repo.joinpath(*parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    return None


def submission_closure(entry: Path = ENTRY,
                       repo: Path = REPO) -> tuple[set[Path], set[str]]:
    """The transitive in-repo import closure of the submission entry point.

    Returns (files in the closure, imported root names seen) — the roots are
    for the self-check below.

    `repo` is a parameter so the walk can be pointed at a synthetic tree: the
    regression tests below need a repository containing a violation, and
    building one in a temp directory is the alternative to writing the
    violation into this repository's own source (review round 2). The arena
    (#20) will want the same handle to walk a different entry point.
    """
    seen: set[Path] = {entry}
    queue = [entry]
    roots: set[str] = set()
    while queue:
        path = queue.pop()
        for name in _imported_modules(path, strict=True):
            roots.add(name.split(".")[0])
            target = _module_file(name, repo)
            if target is not None and target not in seen:
                seen.add(target)
                queue.append(target)
    return seen, roots


def _imported_roots(path: Path) -> set[str]:
    """First segments of the dotted names a file imports.

    The question "which package does this file reach" is about ROOTS:
    `import opponents.extract` reaches the opponents tree exactly as much
    as `import opponents` does - membership of the bare root is what both
    violation checks test (a dotted name like "opponents.extract" would
    walk through a root-membership test untouched; review round 1 caught
    exactly that regression).
    """
    return {name.split(".")[0] for name in _imported_modules(path)}


def test_submission_closure_excludes_opponents() -> None:
    """Nothing the submission loads may import `opponents` — transitively."""
    closure, _ = submission_closure()
    offenders = sorted(
        str(p.relative_to(REPO)) for p in closure
        if FORBIDDEN_ROOT in _imported_roots(p)
    )
    assert not offenders, (
        f"the submission's import closure reaches `{FORBIDDEN_ROOT}`, which "
        "ships only in this repository and must never reach a submission:"
        "\n  " + "\n  ".join(offenders)
    )


def test_submission_closure_excludes_offline_only_packages() -> None:
    """The submission may not import `ortools`.

    An exact solver is an offline instrument: it takes seconds where the turn
    budget is milliseconds, and it is not shipped with a submission. The day
    layer runs a beam search on numpy alone, so the closure must never reach it.

    This asserts on the ROOTS the closure walk collects, so it catches
    `import ortools.sat.python` exactly as it catches `import ortools`.

    Scipy is deliberately NOT on this list. The first version of this guard
    included it and went red on its first run: the closure already reaches
    scipy through `planner/master.py` (#12), whose import is guarded exactly
    so a missing one cannot take the submission down at load, and whose
    presence on Kaggle is confirmed. A guard that forbids something the
    project decided to ship is a wrong guard, not a finding.
    """
    _closure, roots = submission_closure()
    reached = sorted(set(FORBIDDEN_PACKAGES) & roots)
    assert not reached, (
        "the submission's import closure reaches offline-only packages "
        f"{reached}: the runtime path must stay on the day layer's own search, "
        "never an offline exact solver"
    )


def test_the_submission_is_the_agent_folder() -> None:
    """Everything the entry point loads lives inside `agent/`.

    The submission is the `agent/` folder: on Kaggle there is no `offline/`, no
    `tests/`, no `opponents/`, and no builder. So the entry point's import closure
    may not leave the folder — not for a definition, not for an artifact's code, not
    for a helper.

    Written BEFORE the move (AGENTS.md, "The submission is `agent/`"), so it is red
    until the layers are inside `agent/` and green from then on. A guard added after
    the first violation is a cleanup; added before, it is a contract.
    """
    closure, _ = submission_closure()
    outside = sorted(
        str(p.relative_to(REPO)) for p in closure
        if not str(p.relative_to(REPO)).startswith("agent/")
    )
    assert not outside, (
        "the submission's import closure leaves `agent/`, so a submission built from "
        "this folder would not load on Kaggle:\n  " + "\n  ".join(outside)
    )


def test_the_closure_is_real_and_covers_the_agent() -> None:
    """A closure that silently matches nothing passes forever.

    The entry must resolve, and the closure must contain the modules the spine
    is known to load. It used to require `agent/greedy`: `main -> runtime ->
    {greedy, dispatch}` was the M1 spine, and greedy was the policy the runtime
    fell back to. The ladder is retired (#79) — `agent/runtime.py` has one
    policy now, the manager — so greedy is required to be ABSENT here (its own
    guard in `test_agent_runtime.py` lists it with the rest of the retired
    path), and the manager chain is what must be present.
    """
    assert ENTRY.is_file(), f"submission entry point missing: {ENTRY}"
    closure, roots = submission_closure()
    names = {p.relative_to(REPO).with_suffix("").as_posix() for p in closure}
    for required in ("agent/runtime", "agent/dispatch", "agent/manager/core",
                     "agent/planner/master", "agent/planner/colgen"):
        assert required in names, (
            f"closure misses {required}: the walk is broken ({sorted(names)})"
        )
    assert "agent/greedy" not in names, (
        "the retired greedy policy is back in the submission's closure"
    )
    assert "opponents" not in roots, (
        "the submission already imports opponents — the guard is allowing "
        "the thing it exists to forbid"
    )


@contextlib.contextmanager
def _fake_repo(runtime_body: str):
    """A three-file repository in a temp directory, for the tests below.

    The first version of these tests wrote the violation into this repo's
    OWN `agent/greedy.py` and restored it in a `finally`. A clean run put it
    back, but a crash between the two left `import opponents` in a shipped
    file, and two concurrent runs poisoned the shared backup and restored
    the mutated copy permanently — the exact accident this whole file exists
    to prevent, caused by its own test (review round 2). So the violation
    goes in a repository we build and throw away.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "agent").mkdir()
        (root / "opponents").mkdir()
        (root / "opponents" / "__init__.py").write_text("AGENT_COUNT = 0\n")
        (root / "opponents" / "extract.py").write_text("def audit():\n    pass\n")
        (root / "agent" / "main.py").write_text(
            "from agent.runtime import RUNTIME\n\n\ndef agent(obs):\n"
            "    return RUNTIME\n")
        (root / "agent" / "greedy.py").write_text("GREEDY = 1\n")
        (root / "agent" / "runtime.py").write_text(runtime_body)
        yield root, root / "agent" / "main.py"


def test_guard_catches_submodule_imports() -> None:
    """The regression from review round 1: a dotted submodule import
    (`import opponents.extract`) walked through root-membership tests that
    had regressed to testing dotted names. The three shapes that slipped
    through are asserted here, so this can regress only with a failing
    suite."""
    for stmt in ("import opponents",
                 "from opponents.extract import audit",
                 "import opponents.extract"):
        body = f"from agent.greedy import GREEDY\n{stmt}\n\nRUNTIME = 1\n"
        with _fake_repo(body) as (root, entry):
            closure, _ = submission_closure(entry, root)
            reached = any(FORBIDDEN_ROOT in _imported_roots(p) for p in closure)
            assert reached, f"closure missed the violation: {stmt!r}"


def test_relative_imports_are_refused_in_the_closure() -> None:
    """The walk cannot resolve relative imports; an unresolved import
    silently shrinks the closure. They are refused, not ignored."""
    body = "from .greedy import GREEDY\n\nRUNTIME = 1\n"
    with _fake_repo(body) as (root, entry):
        try:
            submission_closure(entry, root)
        except ValueError as exc:
            assert "relative import" in str(exc), f"wrong error: {exc}"
        else:
            raise AssertionError(
                "a relative import inside the submission closure must raise")


def test_relative_imports_are_tolerated_outside_the_closure() -> None:
    """...and only there. `from .x import y` in a package __init__ is
    ordinary Python; the repo-wide scan asks which top-level package a file
    reaches, which a relative import cannot change. Refusing it repo-wide
    made a legal idiom a layering failure (review round 2)."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "pkg_init.py"
        f.write_text("from .tile_state import TileState\nimport numpy as np\n")
        assert _imported_roots(f) == {"numpy"}, (
            "the relative import should be skipped, not raised, outside the "
            f"closure walk — got {_imported_roots(f)}"
        )


def test_the_tests_do_not_touch_the_repository() -> None:
    """These regression tests build their violations in a temp directory.
    If one ever writes into the real tree again, the file it would target is
    the one the submission loads — so this asserts the tree is untouched."""
    for name in ("greedy.py", "runtime.py", "main.py", "dispatch.py"):
        src = REPO / "agent" / name
        if src.is_file():
            assert FORBIDDEN_ROOT not in _imported_roots(src), (
                f"agent/{name} imports `{FORBIDDEN_ROOT}` — a test that "
                "mutates real source was interrupted, or the guard is being "
                "violated for real"
            )
    strays = sorted(p.name for p in (REPO / "agent").glob("*.bak*"))
    assert not strays, f"backup files left in agent/: {strays}"


def test_layering_holds_repo_wide_outside_the_arena() -> None:
    """The old rule, kept where it still applies: no module OUTSIDE the
    arena tree (`offline/`, which imports opponents by design) may import
    it either — the belt to the closure's braces."""
    exempt = {"opponents", "tests", "offline"}
    violations = []
    for p in sorted(REPO.rglob("*.py")):
        top = p.relative_to(REPO).parts[0]
        if top.startswith(".") or top in exempt:
            continue
        if FORBIDDEN_ROOT in _imported_roots(p):
            violations.append(str(p.relative_to(REPO)))
    assert not violations, (
        f"these modules import `{FORBIDDEN_ROOT}` outside the arena:\n  "
        + "\n  ".join(violations)
    )


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
    closure, _ = submission_closure()
    print(f"layering holds: submission closure = {len(closure)} modules, "
          "opponents excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
