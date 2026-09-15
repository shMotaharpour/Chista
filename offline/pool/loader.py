"""Pool loader: import the 19 vendored competitors uniformly.

Issue #20, implementation brief §2. Three facts this loader is built
around (all verified against the pool, brief §2.1):

- the entry points use THREE different names and TWO arities (9 take
  one argument, 9 take an optional second, 1 - adaptive-public-state-
  multi-route's kaggle_agent - REQUIRES two);
- Kaggle takes the LAST top-level def in the file as the entry point,
  not a name. The loader resolves by source order and records which
  rule fired;
- each slug imports as its OWN module object (spec_from_file_location,
  unique name) - never sys.path mutation, never a shared namespace,
  because vendored agents keep module-level state.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

POOL_DIR = Path(__file__).resolve().parents[2] / "opponents"


def slugs() -> list[str]:
    """The vendored slugs, sorted (19 of them; test_opponents pins count)."""
    return sorted(p.name for p in POOL_DIR.iterdir()
                  if p.is_dir() and (p / "agent.py").is_file())


@dataclass(frozen=True)
class LoadedAgent:
    """One resolved competitor: callable + how it was resolved."""

    slug: str
    fn: object                  # the resolved entry-point callable
    fn_name: str                # resolved function name
    arity: int                  # 1 or 2 (obs) / (obs, configuration)
    rule: str                   # which resolution rule fired
    module: object              # the loaded module (state lives here)


def _last_top_level_def(path: Path) -> str | None:
    """The LAST top-level def in the file, by source order (Kaggle's rule)."""
    import ast
    tree = ast.parse(path.read_text(errors="replace"))
    last = None
    for node in tree.body:                       # body = top level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            last = node.name
    return last


def load(slug: str) -> LoadedAgent:
    """Import one competitor and resolve its entry point (never raises here)."""
    agent_path = POOL_DIR / slug / "agent.py"
    if not agent_path.is_file():
        raise FileNotFoundError(f"no agent.py for slug {slug!r}")
    module_name = f"chista_pool_{slug.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses.resolve_requires registrations look the module up in
    # sys.modules DURING exec - it must be registered before exec_module
    # (the documented pattern; without this, slotted dataclass agents
    # crash on import).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Rule 1 (Kaggle's own): the last top-level def in the file.
    rule = "last-top-level-def"
    fn_name = _last_top_level_def(agent_path)
    fn = getattr(module, fn_name, None) if fn_name else None
    if not callable(fn):
        # Rule 2 (fallback): the known-name table, logged.
        rule = "fallback-name"
        for name in ("agent", "_kaggle_submission_entrypoint", "kaggle_agent"):
            fn = getattr(module, name, None)
            if callable(fn):
                fn_name = name
                break
    if not callable(fn):
        raise TypeError(f"{slug}: no callable entry point found")

    params = list(inspect.signature(fn).parameters.values())
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    arity = 2 if len(params) >= 2 else 1
    if len(required) > 2:
        raise TypeError(f"{slug}: entry point takes {len(required)} required "
                        f"args, the harness supplies at most 2")
    return LoadedAgent(slug=slug, fn=fn, fn_name=fn_name, arity=arity,
                       rule=rule, module=module)


def call(loaded: LoadedAgent, obs: dict, configuration=None) -> dict:
    """Call a resolved agent in its own convention (arity by inspection).

    Never raises here - the guard's job; this only adapts the call.
    """
    if loaded.arity == 1:
        return loaded.fn(obs)
    return loaded.fn(obs, configuration)


def load_all() -> dict[str, LoadedAgent]:
    """All slugs, resolved. A slug that fails raises - the guard handles
    per-agent failures at call time; a slug that cannot even be imported
    is a packaging error this issue must hear about loudly."""
    return {slug: load(slug) for slug in slugs()}
