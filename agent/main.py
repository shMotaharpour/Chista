"""Chista agent entry point.

The harness calls a bare function per turn; all state hangs off a
module-level singleton (agent.runtime.RUNTIME), which owns the turn's clock,
the manager, and the never-raise promise.

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

This module never raises while `Config.never_raise` is ON (the submission's
setting): the spine records the failure and returns the safest legal action dict
(all-PASS). It does not play a different policy instead - see
docs/ARCHITECTURE.md section 5. With the switch OFF the failure is recorded and
then re-raised with its traceback: a diagnostic arm for a hunt, never a
submission setting.
"""

from __future__ import annotations

from agent.runtime import RUNTIME


def agent(obs, config=None) -> dict:
    """Harness entry point: one legal action dict per turn, never raises."""
    return RUNTIME.act(obs, config)
