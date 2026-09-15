"""Chista agent entry point.

The harness calls a bare function per turn; all state hangs off a
module-level singleton (agent.runtime.RUNTIME). The signature follows
docs/player_agent.md (`def agent(obs)`); `config` is accepted and ignored
until P1 settles whether the harness passes it. This module never raises:
the runtime's fallback ladder turns any internal error into the safest
legal action dict.
"""

from __future__ import annotations

from agent.runtime import RUNTIME


def agent(obs, config=None) -> dict:
    """Harness entry point: one legal action dict per turn, never raises."""
    return RUNTIME.act(obs, config)
