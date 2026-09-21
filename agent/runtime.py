"""Agent spine: one turn in, one legal action dict out. The manager decides.

The spine owns three things and nothing else: the turn's clock, the call into
`agent/manager/`, and the promise that the harness is never handed an
exception.

Per turn:
  hour 0   -> `Manager.observe(obs, config)`: solve inside the turn's budget and
              commit today's plan. The units act this turn, so the plan has to
              exist now, and this is the one call that may spend the budget.
  hour > 0 -> `Manager.step(budget_ms=...)`: spend what is left of the free
              second on tomorrow's column pool. F058 measured `actTimeout` as
              per turn with the first second free, so hours 1..23 are ~23 free
              seconds a day the solve otherwise never sees.
  then     -> `dispatch_plan(Manager.best(), obs)`: slice the committed plan by
              hour into the dict the engine reads.

**No fallback ladder.** The four-rung ladder answered every turn with `greedy`
while the rung above it could not even import, and the agent looked like it
worked: 2,840 coins against a PASS opponent's 3,000, worse than doing nothing
(`docs/ARCHITECTURE.md` §5). Failure is visible instead: an exception is
recorded on `Runtime.failures`, its day is marked, the turn PASSes, and the log
says so. PASS is legal and honest; a worse policy wearing the same shape is not.

Budget (F046): one free second per turn, unbankable; the harness bills ~35 ms
more than measured, and an exhausted 60 s episode bank forfeits. The working
budget and the reserve live in `agent/config.py` (`turn_budget_ms`,
`reserve_ms`) because they are numbers; the live bank is read back from the
observation, which is authoritative.

Timing log: one `D` line per day, one `A` line per anomaly (a turn over the
working budget, or a failure). There is no switch for it: these numbers are the
evidence (R005), and a switch that can hide them is how a dead rung stayed
invisible for four days.
"""

from __future__ import annotations

import time
from typing import Any

from agent.config import Config
from agent.dispatch import PASS_ACTION, dispatch_plan
from agent.manager.core import Manager

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


class Runtime:
    """The turn's clock, the manager, and the never-raise promise."""

    def __init__(self, config: Config | None = None) -> None:
        #: The numbers the agent plays with: `agent/artifact/config.json` if it
        #: shipped, the committed defaults otherwise (never a mode - see
        #: `agent/config.py`).
        self.cfg = config if config is not None else Config.load()
        #: Built on the first turn, inside `act`'s try: loading the graph and
        #: the contractor is the one step that can fail on a fresh checkout,
        #: and a failure here must be recorded like any other.
        self.manager: Manager | None = None
        self.turns = 0
        #: One entry per failed turn, newest last, capped so a broken season
        #: cannot grow the log without bound.
        self.failures: list[str] = []
        self.failed_days: set[int] = set()
        self.worst_overrun_s = 0.0
        self.day_logged = -1

    # ---- the per-turn entry ----
    def act(self, obs, config=None) -> dict:
        """The harness's per-turn call: one legal action dict, never raising."""
        started = time.perf_counter()
        error: Exception | None = None
        try:
            if self.manager is None:
                self.manager = Manager(self.cfg)
            if _hour_of(obs) == 0:
                self.manager.observe(obs, config)
            else:
                self.manager.step(budget_ms=self._remaining_ms(started))
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
        self_s = time.perf_counter() - started
        self.worst_overrun_s = max(
            self.worst_overrun_s,
            max(0.0, self_s - self.cfg.turn_budget_ms / 1000.0))
        self._log(obs, self_s, error)
        self.turns += 1
        return action

    def _remaining_ms(self, started: float) -> float:
        """What is left of this turn's working budget, for the manager's step.

        `step` improves tomorrow's pool, so it may only spend what the turn has
        not already spent: at hour 0 the observe call has taken its share, and
        the reserve is never handed out (`Config.solve_budget_ms`).
        """
        spent_ms = (time.perf_counter() - started) * 1000.0
        return max(0.0, self.cfg.solve_budget_ms - spent_ms)

    # ---- the log (evidence, never a switch) ----
    def _log(self, obs, self_s: float, error: Exception | None) -> None:
        day, hour = _day_of(obs), _hour_of(obs)
        budget_s = self.cfg.turn_budget_ms / 1000.0
        if day != self.day_logged and hour == 0:
            print(f"D {day} self_ms={self_s * 1000:.1f} "
                  f"overage={_overage_of(obs):.3f} "
                  f"certified={self._certified()} pool={self._pool()} "
                  f"worst_overrun_ms={self.worst_overrun_s * 1000:.1f} "
                  f"failures={len(self.failures)}", flush=True)
            self.day_logged = day
        if error is not None or self_s > budget_s:
            reason = "over_budget" if error is None else f"error={error!r}"
            print(f"A day={day} hour={hour} self_ms={self_s * 1000:.1f} "
                  f"overage={_overage_of(obs):.3f} {reason}", flush=True)

    def _certified(self) -> Any:
        return getattr(self.manager, "certified", None)

    def _pool(self) -> int:
        pool = getattr(self.manager, "pool", None)
        return len(pool) if pool is not None else -1


RUNTIME = Runtime()
