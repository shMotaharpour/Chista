"""Chista agent entry point.

The harness calls a bare function per turn; all state hangs off a
module-level singleton (agent.runtime.RUNTIME).

Signature `agent(obs, config=None)` - deliberately tolerant of both
calling conventions (owner note 2026-09-15):
- the kaggle-environments LIBRARY (local `make()` runs, and P1 verified
  there) truncates args to the agent's argcount, so a two-arg agent gets
  `(obs, config)` with the run configuration;
- the Kaggle competition grader wrapper is NOT verified to pass a second
  argument - it may call `agent(obs)` only, in which case `config=None`
  and nothing in the runtime may depend on it. Every value the runtime
  needs (turnsPerDay, shedCapacity, ...) is either pinned by an engine
  probe (tests/test_agent_obs.py) or carried in the observation.

This module never raises: the runtime's fallback ladder turns any
internal error into the safest legal action dict.
"""

from __future__ import annotations

from agent.runtime import RUNTIME


def agent(obs, config=None) -> dict:
    """Harness entry point: one legal action dict per turn, never raises."""
    return RUNTIME.act(obs, config)
