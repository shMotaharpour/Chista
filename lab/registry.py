"""Agent registry: built-ins from the environment + local file agents."""
from __future__ import annotations

import os

AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")

BUILTINS = {"pass", "random", "starter"}


def list_local_agents() -> dict[str, str]:
    """Scan agent/ for .py files containing an `agent` entry point."""
    found = {}
    if os.path.isdir(AGENT_DIR):
        for fn in sorted(os.listdir(AGENT_DIR)):
            if fn.endswith(".py") and fn != "__init__.py":
                name = fn[:-3]
                found[name] = os.path.join(AGENT_DIR, fn)
    return found


def resolve(name: str) -> str:
    """Return something env.run() accepts: builtin name or path to .py file."""
    if name in BUILTINS:
        return name
    local = list_local_agents()
    if name in local:
        return local[name]
    raise KeyError(f"Unknown agent '{name}'. Built-ins: {sorted(BUILTINS)}. Local: {sorted(local)}")
