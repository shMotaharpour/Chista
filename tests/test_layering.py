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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "agent" / "main.py"
FORBIDDEN_ROOT = "opponents"


def _imported_modules(path: Path) -> set[str]:
    """Full dotted names imported by one file (import x / from x import y).

    Relative imports (`from .greedy import ...`, node.level > 0) RAISE
    here: the closure walk cannot resolve them (level means "up N
    packages", which needs the importing package's position at runtime),
    and an unresolved import silently shrinks the closure - a weaker
    guard that still reports success (review round 1, finding 2). The
    submission tree uses absolute imports; the assertion below keeps
    that true by construction instead of by cleverness.
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
                raise ValueError(
                    f"{path}: relative import (level {node.level}) inside "
                    "the submission tree - the closure walk cannot see it; "
                    "use absolute imports so the layering guard can")
            if node.module:
                names.add(node.module)
    return names


def _module_file(name: str) -> Path | None:
    """The in-repo file a dotted name maps to (first segment = top package)."""
    top = name.split(".")[0]
    pkg_dir = REPO / top
    if not pkg_dir.is_dir():
        return None
    parts = name.split(".")
    candidate = REPO.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = REPO.joinpath(*parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    return None


def submission_closure(entry: Path = ENTRY) -> tuple[set[Path], set[str]]:
    """The transitive in-repo import closure of the submission entry point.

    Returns (files in the closure, imported root names seen) — the roots are
    for the self-check below.
    """
    seen: set[Path] = {entry}
    queue = [entry]
    roots: set[str] = set()
    while queue:
        path = queue.pop()
        for name in _imported_modules(path):
            roots.add(name.split(".")[0])
            target = _module_file(name)
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


def test_the_closure_is_real_and_covers_the_agent() -> None:
    """A closure that silently matches nothing passes forever.

    The entry must resolve, and the closure must contain the modules the
    M1 spine is known to load (main -> runtime -> {greedy, dispatch});
    agent/obs and world/fast_sim join when #11's replanner wires them in
    - this assertion is updated with the wiring, per review round 1.
    """
    assert ENTRY.is_file(), f"submission entry point missing: {ENTRY}"
    closure, roots = submission_closure()
    names = {p.relative_to(REPO).with_suffix("").as_posix() for p in closure}
    # M1 spine: main -> runtime -> {greedy, dispatch}. agent.obs and
    # world.fast_sim join the closure when #11's replanner wires them in.
    for required in ("agent/runtime", "agent/greedy", "agent/dispatch"):
        assert required in names, (
            f"closure misses {required}: the walk is broken ({sorted(names)})"
        )
    assert "opponents" not in roots, (
        "the submission already imports opponents — the guard is allowing "
        "the thing it exists to forbid"
    )


def test_guard_catches_submodule_imports() -> None:
    """The regression from review round 1: a dotted submodule import
    (`import opponents.extract`) walked through root-membership tests
    that had regressed to testing dotted names. The exact shapes that
    slipped through are asserted here, so this can regress only with a
    failing suite."""
    import shutil

    target = REPO / "agent" / "greedy.py"
    backup = target.with_suffix(".py.bak_guard")
    shutil.copy(target, backup)
    try:
        text = target.read_text()
        for stmt in ("import opponents\n",
                     "from opponents.extract import audit\n",
                     "import opponents.extract\n"):
            target.write_text(text + stmt)
            closure, _ = submission_closure()
            reached = any(FORBIDDEN_ROOT in _imported_roots(p) for p in closure)
            assert reached, f"closure missed the violation: {stmt!r}"
    finally:
        shutil.copy(backup, target)
        backup.unlink()


def test_relative_imports_are_refused_in_the_closure() -> None:
    """The walk cannot resolve relative imports; an unresolved import
    silently shrinks the closure. They are refused, not ignored."""
    import shutil

    target = REPO / "agent" / "runtime.py"
    backup = target.with_suffix(".py.bak_guard")
    shutil.copy(target, backup)
    try:
        text = target.read_text()
        target.write_text(text.replace(
            "from agent.greedy import greedy_action",
            "from .greedy import greedy_action"))
        raised = False
        try:
            submission_closure()
        except ValueError:
            raised = True
        assert raised, "a relative import inside the submission must raise"
    finally:
        shutil.copy(backup, target)
        backup.unlink()


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
