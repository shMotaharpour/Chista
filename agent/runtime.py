"""Agent runtime: per-turn loop, deadline guard, fallback ladder, timing log.

The spine issue (#9): turn the harness's observation into one legal action
dict, never raise, never draw from the 60 s overage bank (F046: 1 free
second per turn, unbankable; the harness bills ~35 ms more than measured,
so the working budget is 0.965 s).

Per turn:
  with the deadline enforced, decode the observation (stub in M1 - #10),
  replan at hour 0 (stub in M1), dispatch the committed plan.
On exception or deadline expiry the fallback ladder applies, in order:
  1. the committed plan, dispatched greedily
  2. the previous day's plan, repaired
  3. the greedy tile policy (agent/greedy.py)
  4. all-PASS
Every rung returns a shape-valid dict; the top level catches everything.

Timing log (one flat, parseable line per record): a compact line per day
always, a per-turn line only when anomalous (over target or the bank
moved), and the full per-turn stream under CHISTA_TRACE=1.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from agent.greedy import greedy_action
from agent.dispatch import PASS_ACTION, dispatch_plan

# F046 numbers (engine finding, not assumptions): 1.0 s free per turn,
# ~35 ms harness billing overhead, 60 s episode bank.
FREE_SECONDS = 1.0
BILLING_MARGIN_S = 0.035
WORKING_BUDGET_S = FREE_SECONDS - BILLING_MARGIN_S
EPISODE_BANK_S = 60.0
# Turn-length anomaly threshold for the timing log. Named TODO (owner's
# rule: no invented numbers): the M1 distribution (bench_turn_budget)
# sets the real value when it lands; 0.4 s is the issue's acceptance max.
ANOMALY_S = 0.4


class Deadline:
    """The turn's wall clock, read from the observation like the harness does."""

    def __init__(self, remaining_overage_s: float):
        self.remaining_overage_s = remaining_overage_s
        self._start = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    def expired(self) -> bool:
        return self.elapsed_s() >= WORKING_BUDGET_S

    def remaining_ms(self) -> int:
        return max(0, int((WORKING_BUDGET_S - self.elapsed_s()) * 1000))


def _overage_of(obs) -> float:
    """The authoritative remaining bank, from the observation itself."""
    if isinstance(obs, dict):
        return float(obs.get("remainingOverageTime", EPISODE_BANK_S))
    return float(getattr(obs, "remainingOverageTime", EPISODE_BANK_S))


def _hour_of(obs) -> int:
    try:
        if isinstance(obs, dict):
            return int(obs.get("hour", 0))
        return int(getattr(obs, "hour", 0))
    except (TypeError, ValueError):
        return 0


class Runtime:
    """Per-turn state singleton: plans, timing log, fallback ladder."""

    def __init__(self) -> None:
        self.plan: Any = None                # committed day plan (stub: None)
        self.prev_plan: Any = None           # previous day's plan, for rung 2
        self.last_return_t: float | None = None
        self.day_logged = -1
        self.turn_index = 0
        self.worst_overrun_s = 0.0
        self.trace = os.environ.get("CHISTA_TRACE") == "1"

    # ---- fallback ladder rungs (each returns a shape-valid dict) ----
    def _rung_plan(self, obs) -> dict | None:
        if self.plan is not None:
            try:
                return dispatch_plan(self.plan, obs)
            except Exception:
                return None
        return None

    def _rung_prev_plan(self, obs) -> dict | None:
        if self.prev_plan is not None:
            try:
                return dispatch_plan(self.prev_plan, obs)
            except Exception:
                return None
        return None

    def _rung_greedy(self, obs) -> dict:
        return greedy_action(obs)

    def _rung_pass(self, _obs) -> dict:
        return dict(PASS_ACTION)

    # ---- the per-turn entry ----
    def act(self, obs, config=None) -> dict:
        t_entry = time.perf_counter()
        gap_ms = 0.0
        if self.last_return_t is not None:
            gap_ms = (t_entry - self.last_return_t) * 1000.0
        deadline = Deadline(_overage_of(obs))
        error = None
        try:
            # M1: decode (#10) and replan are stubs - the greedy policy is
            # the brain until #11 lands. At hour 0 a committed plan archives
            # (rung 2: the previous day's plan, dispatched again); with no
            # replanner, plan stays None and greedy drives the day.
            hour = _hour_of(obs)
            if hour == 0 and self.plan is not None:
                self.prev_plan, self.plan = self.plan, None
            action = (self._rung_plan(obs)
                      or self._rung_prev_plan(obs)
                      or self._rung_greedy(obs)
                      or self._rung_pass(obs))
        except Exception as exc:                     # noqa: BLE001 - never raise
            error = exc
            try:
                action = self._rung_greedy(obs)
            except Exception:                        # noqa: BLE001
                action = dict(PASS_ACTION)
        # timing bookkeeping (outside the try: logging must not change behaviour)
        self_ms = time.perf_counter() - t_entry
        overrun_s = max(0.0, self_ms - WORKING_BUDGET_S)
        self.worst_overrun_s = max(self.worst_overrun_s, overrun_s)
        self._log_turn(obs, deadline, self_ms, gap_ms, error)
        self.last_return_t = time.perf_counter()
        self.turn_index += 1
        return action

    def _log_turn(self, obs, deadline: Deadline, self_ms: float,
                  gap_ms: float, error: Exception | None) -> None:
        day = obs.get("day", -1) if isinstance(obs, dict) else getattr(obs, "day", -1)
        hour = _hour_of(obs)
        anomalous = (self_ms > ANOMALY_S or error is not None)
        if self.trace:
            kind = "ERR" if error else "ok"
            print(f"T {self.turn_index} day={day} hour={hour} "
                  f"self_ms={self_ms * 1000:.1f} gap_ms={gap_ms:.1f} "
                  f"overage={deadline.remaining_overage_s:.3f} {kind}",
                  flush=True)
        if day != self.day_logged and hour == 0:
            # one compact line per day, always
            print(f"D {day} overage={deadline.remaining_overage_s:.3f} "
                  f"worst_overrun_ms={self.worst_overrun_s * 1000:.1f}",
                  flush=True)
            self.day_logged = day
        elif anomalous:
            reason = "deadline" if error is None else f"error={error!r}"
            print(f"A day={day} hour={hour} self_ms={self_ms * 1000:.1f} "
                  f"gap_ms={gap_ms:.1f} {reason}", flush=True)


RUNTIME = Runtime()
