"""Agent runtime: per-turn loop, deadline guard, fallback ladder, timing log.

The spine issue (#9): turn the harness's observation into one legal action
dict, never raise, never draw from the 60 s overage bank (F046: 1 free
second per turn, unbankable; the harness bills ~35 ms more than measured,
so the working budget is 0.965 s).

Per turn:
  with the deadline enforced, decode the observation (#10),
  replan at hour 0 (the replanner rung, #11 - opt-in, see agent/replan.py),
  dispatch the committed plan.
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


def _replanner_rung(runtime, obs):
    """The replanner rung (#11), imported on first use.

    `tile_dp` and the graph artifact are imported lazily: a run that never
    replans (the default) pays nothing for them, and the harness's 60 s bank is
    never charged for code the turn does not use (F046).
    """
    from agent.replan import replan_day
    return replan_day(runtime, obs)


def _attach_market(runtime, action, obs):
    """The market layer (#15): `CHISTA_MARKET=spread|dump`, else untouched.

    Imported on first use for the same reason as the replanner: a run with
    the layer off must not pay for `secretary.market`'s engine imports.
    """
    mode = getattr(runtime, "_market_mode", "")
    if mode not in ("spread", "dump"):
        return action
    layer = getattr(runtime, "market", None)
    if layer is None:
        from agent.market_layer import MarketLayer
        layer = MarketLayer(mode)
        runtime.market = layer
    from agent.market_layer import attach
    return attach(layer, action, obs)


class Runtime:
    """Per-turn state singleton: plans, timing log, fallback ladder."""

    def __init__(self) -> None:
        self.plan: Any = None                # committed day plan (rung 1's input)
        self.prev_plan: Any = None           # previous day's plan, for rung 2
        # The replanner rung (#11). `None` means the socket is empty and the
        # ladder falls through to greedy: the rung prices one tile per unit and
        # cannot yet carry the inputs its chains assume (see agent/replan.py),
        # so it stays opt-in until the secretary layer (#14) exists.
        self.replanner: Any = (_replanner_rung
                               if os.environ.get("CHISTA_REPLAN") == "1"
                               else None)
        # The market layer (#15). Read once at construction, like the
        # replanner switch, so a turn never pays for the lookup.
        self._market_mode = os.environ.get("CHISTA_MARKET", "")
        self.market: Any = None              # built on first use
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
            # The replanner rung (#11) fills `self.plan` at hour 0 when it is
            # enabled; with no replanner (or one that bailed) the greedy policy
            # is the brain, as in M1. At hour 0 a committed plan archives to
            # prev_plan (rung 2: the previous day's plan, dispatched again).
            hour = _hour_of(obs)
            if hour == 0 and self.plan is not None:
                self.prev_plan, self.plan = self.plan, None
            # The deadline gates the ladder BETWEEN rungs (review 2, finding
            # 1): a rung that has burned the budget yields its turn to the
            # cheapest remaining rung, and the except path below honours the
            # same gate.
            #
            # The replanner (#11) is the first heavy rung, so it polls the
            # deadline itself and raises TimeoutError mid-work; the between-rung
            # gate could only ever see a replan that had already spent the turn.
            # It runs ONCE PER DAY, at hour 0: within a day the graph the DP
            # prices is unchanged except by our own execution, so the stored
            # plan is dispatched for the remaining 23 hours.
            self._deadline = deadline
            if hour == 0 and self.replanner is not None:
                self.plan = self.replanner(self, obs)
            action = self._rung_plan(obs)
            if action is None and not deadline.expired():
                action = self._rung_prev_plan(obs)
            if action is None and not deadline.expired():
                action = self._rung_greedy(obs)
            if action is None:
                action = self._rung_pass(obs)
            # The market half (#15) rides on WHICHEVER rung answered: SELL
            # reads the shed (F043), and the shed is filled by the nightly
            # drop regardless of the unit plan, so sells are not a unit
            # decision. `attach` never raises; on failure the rung's own
            # action survives.
            action = _attach_market(self, action, obs)
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
