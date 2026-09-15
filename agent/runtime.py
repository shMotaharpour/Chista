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

Probes settled on the competition grader (2026-09-15, probe script by
the owner):
- **P2 - sequential.** The grader runs the two agents one after the
  other, so `gap_ms` carries the opponent's thinking time. Calibration:
  against an instant opponent `gap_p50` = 36.76 ms (35.02-39.33 across
  29 days) - that is the engine+harness constant; anything above ~37 ms
  is opponent deliberation (#21 reads it from turn one).
- **P3 - imports.** numpy is free on the grader (already in
  sys.modules); scipy.optimize costs ~0.5 s of the 60 s bank ONCE;
  ortools is ABSENT on the grader. Never import scipy/torch in the
  agent's hot path.
- **F046 machine-speed correction:** the grader runs the tile-DP sweep
  1.17-1.41x SLOWER than the dev box (2 cores vs 4), not 1.95x faster -
  F046 needs that note; the billing constants (1.0 s, 35 ms, 60 s) are
  all confirmed by measurement (1.25 s sleep drew 0.2822 s).
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
# Turn-length anomaly threshold for the timing log: the issue's
# acceptance max (0.4 s), kept until the M1 distribution under the
# grader's measured conditions (1.17-1.41x slower than this box, P2/P3
# settled) justifies a different line.
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
        # TODO(bank-policy, #9 section 4): the self-calibrating gate
        # `reserve = max(FLOOR_S, SAFETY * worst_overrun_s)` is NOT
        # implemented yet - M1 has no replan to gate. worst_overrun_s is
        # tracked and logged so the policy lands on a measured history;
        # SAFETY / FLOOR_S come from bench_turn_budget's distribution.
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
        # TODO(repair, #9 section 3 rung 2): this dispatches yesterday's
        # plan at today's hour with NO repair step yet - the rung's name
        # promises more than it does. Unreachable until a replanner fills
        # `plan` (#11); the repair lands with it.
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
            # The deadline gates the ladder BETWEEN rungs (review 2, finding
            # 1): a rung that has burned the budget yields its turn to the
            # cheapest remaining rung, and the except path below honours the
            # same gate.
            #
            # TODO(rung self-bail): `self._deadline` is published so a rung can
            # poll it and raise TimeoutError mid-work, but NO rung polls it
            # yet - nothing in M1 runs long enough to need it, so a
            # between-rung gate is the whole protection today. The first heavy
            # rung (the replanner, #11) must poll;
            # tests/test_agent_runtime.py::test_deadline_gates_the_ladder pins
            # that contract with a polling rung of its own.
            self._deadline = deadline
            action = self._rung_plan(obs)
            if action is None and not deadline.expired():
                action = self._rung_prev_plan(obs)
            if action is None and not deadline.expired():
                action = self._rung_greedy(obs)
            if action is None:
                action = self._rung_pass(obs)
        except Exception as exc:                     # noqa: BLE001 - never raise
            error = exc
            # the gate holds on this path too: an expired budget hands the
            # turn straight to PASS (the raising rung already burned it)
            if deadline.expired():
                action = dict(PASS_ACTION)
            else:
                try:
                    action = self._rung_greedy(obs)
                except Exception:                    # noqa: BLE001
                    action = dict(PASS_ACTION)
        # timing bookkeeping (outside the try: logging must not change behaviour)
        self_s = time.perf_counter() - t_entry          # seconds
        overrun_s = max(0.0, self_s - WORKING_BUDGET_S)
        self.worst_overrun_s = max(self.worst_overrun_s, overrun_s)
        self._log_turn(obs, deadline, self_s, gap_ms, error)
        self.last_return_t = time.perf_counter()
        self.turn_index += 1
        return action

    def _log_turn(self, obs, deadline: Deadline, self_s: float,
                  gap_ms: float, error: Exception | None) -> None:
        day = obs.get("day", -1) if isinstance(obs, dict) else getattr(obs, "day", -1)
        hour = _hour_of(obs)
        anomalous = (self_s > ANOMALY_S or error is not None)
        if self.trace:
            kind = "ERR" if error else "ok"
            print(f"T {self.turn_index} day={day} hour={hour} "
                  f"self_ms={self_s * 1000:.1f} gap_ms={gap_ms:.1f} "
                  f"overage={deadline.remaining_overage_s:.3f} {kind}",
                  flush=True)
        # day line and anomaly line are independent (review 2, finding 4):
        # hour 0 of a new day is where the replan will run - the one turn
        # whose cost we most need in the log; suppressing it hid a 1.2 s
        # overrun behind the day line.
        if day != self.day_logged and hour == 0:
            # one compact line per day, always
            print(f"D {day} overage={deadline.remaining_overage_s:.3f} "
                  f"worst_overrun_ms={self.worst_overrun_s * 1000:.1f}",
                  flush=True)
            self.day_logged = day
        if anomalous:
            reason = "deadline" if error is None else f"error={error!r}"
            print(f"A day={day} hour={hour} self_ms={self_s * 1000:.1f} "
                  f"gap_ms={gap_ms:.1f} {reason}", flush=True)


RUNTIME = Runtime()
