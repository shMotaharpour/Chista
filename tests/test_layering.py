"""The submission never imports the competitors.

Run:  .venv/bin/python -m tests.test_layering

`opponents/` is third-party code vendored for evaluation. It is input to the
arena, never to the agent: a submission that imports it would ship someone
else's published work, and on Kaggle it would not be there to import anyway.

Written before `agent/` exists, deliberately. A guard added after the first
violation is a cleanup; added before, it is a contract — and the cost of
discovering this one late is a submission that fails to load.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Where the rule does not apply: the vendored tree itself, and the tests that
# exist to check it.
EXEMPT = {"opponents", "tests"}
FORBIDDEN_ROOT = "opponents"


def source_files() -> list[Path]:
    return sorted(p for p in REPO.rglob("*.py")
                  if not p.relative_to(REPO).parts[0].startswith(".")
                  and p.relative_to(REPO).parts[0] not in EXEMPT)


def imported_roots(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_module_outside_the_arena_imports_opponents() -> None:
    violations = [str(p.relative_to(REPO)) for p in source_files()
                  if FORBIDDEN_ROOT in imported_roots(p)]
    assert not violations, (
        f"these modules import `{FORBIDDEN_ROOT}`, which ships only in this "
        "repository and must never reach a submission:\n  " + "\n  ".join(violations)
    )


def test_the_guard_can_see_the_files_it_guards() -> None:
    """A scan that silently matches nothing passes forever."""
    scanned = source_files()
    assert scanned, "no source files scanned: the walk or the exemptions are wrong"
    roots = {p.relative_to(REPO).parts[0] for p in scanned}
    assert {"world", "tile_dp"} <= roots, (
        f"expected world/ and tile_dp/ in the scan, saw {sorted(roots)}"
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
    print(f"layering holds across {len(source_files())} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
