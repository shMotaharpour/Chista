"""Chista agent entry point.

The harness calls a bare function per turn; all state hangs off a
module-level singleton (agent.runtime.RUNTIME). P1 (settled on the
competition grader, 2026-09-15): the second argument ARRIVES and carries
the run configuration (episodeSteps, actTimeout, turnsPerDay, ...) - the
signature below is the settled one. This module never raises: the
runtime's fallback ladder turns any internal error into the safest legal
action dict.
"""

from __future__ import annotations

from agent.runtime import RUNTIME


def agent(obs, config=None) -> dict:
    """Harness entry point: one legal action dict per turn, never raises."""
    return RUNTIME.act(obs, config)
