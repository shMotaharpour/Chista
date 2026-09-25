"""Agent spine: one turn in, one legal action dict out. The manager decides.

The spine owns three things and nothing else: the turn's clock, the call into
`agent/manager/`, and the promise that the harness is never handed an
exception.

Per turn:
  hour 0   -> `Manager.observe(obs, config)`: solve inside the turn's budget and
              commit today's plan. The units act this turn, so the plan has to
              exist now, and this is the one call that may spend the budget.
  hour > 0 -> `Manager.step(obs, budget_ms=...)`: spend what is left of the free
              second on tomorrow's column pool, and feed the rival tracker the
              turn it is in (its bucket is a window over 24 TURNS, so a tracker
              fed once a day would read 24 days and call it a day).
  then     -> `dispatch_plan(Manager.best(), obs)`: slice the committed plan by
              hour into the dict the engine reads.

**No fallback ladder.** The four-rung ladder answered every turn with `greedy`
while the rung above it could not even import, and the agent looked like it
worked: 2,840 coins against a PASS opponent's 3,000, worse than doing nothing
(`docs/ARCHITECTURE.md` §5). Failure is visible instead: an exception is
recorded on `Runtime.failures`, its day is marked, the turn PASSes, and the log
says so. PASS is legal and honest; a worse policy wearing the same shape is not.
That is `Config.never_raise` ON, the submission's setting. With it OFF the same
record is written and the exception is then re-raised with its traceback — a
diagnostic arm, so a hunt sees what reached the boundary instead of a PASSed day.

Budget (F046): one free second per turn, unbankable; the harness bills ~35 ms
more than measured, and an exhausted 60 s episode bank forfeits. The working
budget and the reserve live in `agent/config.py` (`turn_budget_ms`,
`reserve_ms`) because they are numbers; the live bank is read back from the
observation, which is authoritative.

Timing log: one `D` line per day, one `A` line per anomaly (a turn over the
working budget, or a failure), and — when `Config.log_gaps` is on — one `G`
line per turn carrying the wall clock between this call and the last, with the
running mean and sd. The gap is the opponent's turn as seen from inside: the
two seats run one after the other (F058), so above the engine's own ~36.8 ms
floor it is the rival spending the shared second. There is no switch for the
rest of it: those numbers are the evidence (R005), and a switch that can hide
them is how a dead rung stayed invisible for four days.
"""

from __future__ import annotations

import time
from typing import Any

from agent.config import Config
from agent.dispatch import PASS_ACTION, dispatch_plan
from agent.manager.core import Manager, opponent_model

#: F046's episode overage bank, as the default for a caller that omits it. The
#: observation carries the live figure (`remainingOverageTime`); this constant
#: is only what `_overage_of` falls back to.
EPISODE_BANK_S = 60.0


def _hour_of(obs) -> int:
    try:
        return int(obs.get("hour", 0)) if isinstance(obs, dict) \
            else int(getattr(obs, "hour", 0))
    except (TypeError, ValueError):
        return 0


def _day_of(obs) -> int:
    try:
        return int(obs.get("day", -1)) if isinstance(obs, dict) \
            else int(getattr(obs, "day", -1))
    except (TypeError, ValueError):
        return -1


def _overage_of(obs) -> float:
    """The remaining bank, from the observation itself (F046)."""
    if isinstance(obs, dict):
        return float(obs.get("remainingOverageTime", EPISODE_BANK_S))
    return float(getattr(obs, "remainingOverageTime", EPISODE_BANK_S))


class GapStats:
    """Running mean and sd of the wall clock between two calls of the agent.

    The harness runs the two seats one after the other (F058), so the gap from
    the end of our previous turn to the start of this one is the engine's own
    overhead plus, above that floor, the opponent's thinking time. Measured
    against an instant opponent the floor is ~36.8 ms (`gap_p50`, 35.0-39.3
    across 29 days, the P2 probe in this repo's history), so anything above it
    is the rival spending the shared turn — the only reading of the opponent's
    resource use a submission can take from inside the game.

    Welford's update: one pass, no stored history, and `sd` is 0.0 until there
    are two readings to compare. The first turn of a season has no gap at all
    (there is no previous return), so `n` lags the turn count by one.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.last_ms = 0.0

    def add(self, ms: float) -> None:
        self.n += 1
        delta = ms - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (ms - self.mean)
        self.last_ms = float(ms)

    @property
    def sd(self) -> float:
        """Sample sd; 0.0 before there is a second reading."""
        return (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class Runtime:
    """The turn's clock, the manager, and the never-raise boundary.

    `Config.never_raise` decides whether a failed turn answers all-PASS (the
    submission's contract) or is re-raised after it is recorded and logged (the
    diagnostic arm, OFF by the owner's order while the day layer's short-horizon
    refusal is being hunted).
    """

    def __init__(self, config: Config | None = None) -> None:
        #: The numbers the agent plays with: `agent/artifact/config.json` if it
        #: shipped, the committed defaults otherwise (never a mode - see
        #: `agent/config.py`).
        self.cfg = config if config is not None else Config.load()
        #: Built on the first turn, inside `act`'s try: loading the graph and
        #: the contractor is the one step that can fail on a fresh checkout,
        #: and a failure here must be recorded like any other.
        self.manager: Manager | None = None
        #: The pretrained rival model is loaded HERE, at import, and never in a
        #: turn: 1,222 ms measured against a 965 ms hour-0 budget, so loading it
        #: lazily would cost the season's first day its plan (#95). `Manager`
        #: shares this one instance.
        opponent_model()
        self.turns = 0
        #: One entry per failed turn, newest last, capped so a broken season
        #: cannot grow the log without bound.
        self.failures: list[str] = []
        self.failed_days: set[int] = set()
        self.worst_overrun_s = 0.0
        self.day_logged = -1
        #: The wall clock between two calls of the agent: the opponent's turn,
        #: read from inside (see `GapStats`).
        self.gaps = GapStats()
        self.last_return_t: float | None = None

    # ---- the per-turn entry ----
    def act(self, obs, config=None) -> dict:
        """The harness's per-turn call: one legal action dict, never raising."""
        started = time.perf_counter()
        # The gap carries the opponent's turn (see `GapStats`); the first turn
        # of a season has no previous return and is not a reading.
        if self.last_return_t is not None:
            self.gaps.add((started - self.last_return_t) * 1000.0)
        error: Exception | None = None
        raise_after: Exception | None = None
        try:
            if self.manager is None:
                self.manager = Manager(self.cfg)
            if _hour_of(obs) == 0:
                self.manager.observe(obs, config)
            else:
                self.manager.step(obs, budget_ms=self._remaining_ms(started))
            action = dispatch_plan(self.manager.best(), obs)
        except Exception as exc:                  # noqa: BLE001 - the harness contract
            # The ladder's job, done honestly: a legal answer plus a record.
            error = exc
            day, hour = _day_of(obs), _hour_of(obs)
            if len(self.failures) < 64:
                self.failures.append(
                    f"day {day} hour {hour}: {type(exc).__name__}: {exc}")
            self.failed_days.add(day)
            action = dict(PASS_ACTION)
            if not self.cfg.never_raise:
                # `Config.never_raise` is OFF: the failure is recorded and the
                # `A` line below still carries it, and then it is RE-RAISED with
                # its traceback. Raised after the log, so the season's log holds
                # the same evidence either way — the switch decides whether the
                # run continues, never whether the failure is written down.
                raise_after = exc
        self_s = time.perf_counter() - started
        if self.cfg.turn_budget_ms:      # no budget, no overrun: there is nothing to overrun
            self.worst_overrun_s = max(
                self.worst_overrun_s,
                max(0.0, self_s - self.cfg.turn_budget_ms / 1000.0))
        self._log(obs, self_s, error)
        self.last_return_t = time.perf_counter()
        self.turns += 1
        if raise_after is not None:
            raise raise_after
        return action

    def _remaining_ms(self, started: float) -> float | None:
        """What is left of this turn's working budget, for the manager's step.

        `step` improves tomorrow's pool, so it may only spend what the turn has
        not already spent: at hour 0 the observe call has taken its share, and
        the reserve is never handed out (`Config.solve_budget_ms`).

        None with no turn budget (`Config.turn_budget_ms = 0`): there is no share to divide, and
        `None` is what `Manager.step` reads as "no deadline". Subtracting from it was a
        TypeError on every turn after the first (measured: eight `test_agent_runtime` guards).
        """
        budget = self.cfg.solve_budget_ms
        if budget is None:
            return None
        spent_ms = (time.perf_counter() - started) * 1000.0
        return max(0.0, budget - spent_ms)

    # ---- the log (evidence, never a switch) ----
    def _log(self, obs, self_s: float, error: Exception | None) -> None:
        day, hour = _day_of(obs), _hour_of(obs)
        budget_s = self.cfg.turn_budget_ms / 1000.0
        # With no turn budget there is no overrun to report: `self_s > 0.0` is always true, and
        # the `A` line would call every turn over budget for a budget nobody set.
        over = bool(budget_s) and self_s > budget_s
        if day != self.day_logged and hour == 0:
            print(f"D {day} self_ms={self_s * 1000:.1f} "
                  f"overage={_overage_of(obs):.3f} "
                  f"certified={self._certified()} pool={self._pool()} "
                  f"worst_overrun_ms={self.worst_overrun_s * 1000:.1f} "
                  f"failures={len(self.failures)}", flush=True)
            self.day_logged = day
        if error is not None or over:
            reason = "over_budget" if error is None else f"error={error!r}"
            print(f"A day={day} hour={hour} self_ms={self_s * 1000:.1f} "
                  f"overage={_overage_of(obs):.3f} {reason}", flush=True)
        # The opponent's turn, from inside: off unless the run's log is one we
        # mean to read (`Config.log_gaps`).
        if self.cfg.log_gaps and self.gaps.n:
            print(f"G turn={self.turns} day={day} hour={hour} "
                  f"gap_ms={self.gaps.last_ms:.1f} n={self.gaps.n} "
                  f"mean_ms={self.gaps.mean:.1f} sd_ms={self.gaps.sd:.1f}",
                  flush=True)

    def _certified(self) -> Any:
        return getattr(self.manager, "certified", None)

    def _pool(self) -> int:
        pool = getattr(self.manager, "pool", None)
        return len(pool) if pool is not None else -1


RUNTIME = Runtime()
