"""Chista's agent: one entry point, one turn, one legal action dict.

The harness takes the LAST callable the module defines — `kaggle_environments`
picks `[v for v in env.values() if callable(v)][-1]` out of this file's globals
(`agent.get_last_callable`) — and truncates the call to that callable's own
argcount (`Agent.act`: `args[: self.agent.__code__.co_argcount]`). So the shape
of this file IS the contract:

    Agent        the agent as an object: the manager, and the record of failures
    AGENT        the one instance, built at import (it warms the rival model)
    agent(obs)   the entry point, defined LAST, ONE argument, no configuration

**An object is not the entry point.** An instance has no `__code__`, so the
harness cannot truncate the call and would hand it `(obs, config)`; the last
callable here must therefore be a plain one-argument function, and nothing
callable may be defined after it.

**There is no clock in this file, and nothing here feeds one.** The turn's
length is the harness's business (`actTimeout`, and the overage bank it enforces
itself); a spine that measured its own wall clock made how much work a turn did
depend on the machine, so two runs of one seed disagreed. What decides here is
the observation: hour 0 plans the day (`Manager.observe`), every later hour
works the column pool (`Manager.step`).

**A failure is recorded and passed.** `Agent.failures` and `Agent.failed_days`
carry it, the `A` line prints it, and the turn answers all-PASS: an exception out
of this module is an episode the harness cannot score. That is
`Config.never_raise` ON, the submission's setting. With it OFF (the owner's
setting while something is being hunted) the same record is written and the same
line printed, and then the exception is re-raised with its traceback. The switch
decides whether a failure is VISIBLE, never which policy runs.
"""

from __future__ import annotations

from agent.config import Config
from agent.dispatch import PASS_ACTION, dispatch_plan
from agent.manager.core import Manager, opponent_model


def _hour_of(obs) -> int:
    """The turn's hour: 0 for anything that does not carry one."""
    try:
        return int(obs.get("hour", 0)) if isinstance(obs, dict) \
            else int(getattr(obs, "hour", 0))
    except (TypeError, ValueError):
        return 0


def _day_of(obs) -> int:
    """The turn's day: -1 for anything that does not carry one."""
    try:
        return int(obs.get("day", -1)) if isinstance(obs, dict) \
            else int(getattr(obs, "day", -1))
    except (TypeError, ValueError):
        return -1


class Agent:
    """The whole agent: the manager, and the record of what failed.

    The manager is built on the FIRST turn and inside the boundary: loading the
    tile graph is the one step that can fail on a fresh checkout, and a failure
    there is a failure like any other — recorded, its day marked, the turn
    passed.
    """

    def __init__(self, config: Config | None = None) -> None:
        #: The numbers the agent plays with: `agent/artifact/config.json` if it
        #: shipped, the committed defaults otherwise (never a mode).
        self.cfg = config if config is not None else Config.load()
        self.manager: Manager | None = None
        #: The config object the manager was built FROM. Identity, not equality:
        #: the run's config is built once, so the same object is the same run and
        #: nothing is rebuilt mid-run — but a DIFFERENT object is a different
        #: run (a test pinning a regime, a probe injecting a run's numbers), and
        #: its numbers must be built INTO the machinery rather than left on the
        #: shelf beside a manager still holding the old ones. That stale manager
        #: is exactly how an injected config failed to reach the model: the land
        #: count a test set on `AGENT.cfg` never left the object. `None` means
        #: "no manager was built here" (a caller assigned one), and then nothing
        #: is rebuilt: an injected manager is the caller's, not ours to replace.
        self._manager_cfg: Config | None = None
        #: One entry per failed turn, newest last, capped so a broken season
        #: cannot grow the log without bound — and the days it happened on.
        self.failures: list[str] = []
        self.failed_days: set[int] = set()
        self.day_logged = -1

    def __call__(self, obs, config=None) -> dict:
        """One turn in, one legal action dict out.

        Hour 0 is the day's plan (`observe`), every later hour is pool work
        (`step`). The plan the manager holds is what the engine gets, sliced by
        hour — this function decides nothing else.
        """
        day, hour = _day_of(obs), _hour_of(obs)
        error: Exception | None = None
        raise_after: Exception | None = None
        try:
            if self.manager is None or (self._manager_cfg is not None
                                        and self._manager_cfg is not self.cfg):
                self.manager = Manager(self.cfg)
                self._manager_cfg = self.cfg
            if hour == 0:
                self.manager.observe(obs, config)
            else:
                self.manager.step(obs)
            action = dispatch_plan(self.manager.best(), obs,
                                   getattr(self.manager, "terms", None))
        except Exception as exc:              # noqa: BLE001 - the harness contract
            error = exc
            if len(self.failures) < 64:
                self.failures.append(
                    f"day {day} hour {hour}: {type(exc).__name__}: {exc}")
            self.failed_days.add(day)
            action = dict(PASS_ACTION)
            if not self.cfg.never_raise:
                raise_after = exc
        self._log(day, hour, error)
        if raise_after is not None:
            raise raise_after
        return action

    # ---- the log (evidence, never a switch) ----
    def _log(self, day: int, hour: int, error: Exception | None) -> None:
        """One `D` line per day, one `A` line per anomaly.

        A switch that can hide evidence does not exist: these lines are what a
        season's log is read for. What is NOT here is any wall-clock figure —
        the `A` line used to fire on a turn over a budget of OUR choosing, which
        is the measurement that made the plan follow the machine.
        """
        if error is not None:
            print(f"A day={day} hour={hour} error={error!r}", flush=True)
            return
        if hour == 0 and day != self.day_logged:
            print(f"D {day} certified={self._certified()} "
                  f"pool={self._pool()} failures={len(self.failures)}",
                  flush=True)
            self.day_logged = day

    def _certified(self):
        return getattr(self.manager, "certified", None)

    def _pool(self) -> int:
        pool = getattr(self.manager, "pool", None)
        return len(pool) if pool is not None else -1


#: The rival model, loaded ONCE per process and never inside a turn: an
#: `OpponentModel(pretrained=True)` reads `agent/artifact/opponent_counts.npz`
#: and measured 1,222 ms, more than a turn can spend. Every `Manager` shares
#: this one instance.
opponent_model()

#: The one agent. Built here, so the graph and the model are warm before the
#: harness's first turn.
AGENT = Agent()


def agent(obs, config=None) -> dict:
    """The harness's entry point: TWO arguments, and the LAST callable here.

    Two because that is the only way the run's own configuration reaches an
    agent. `kaggle_environments.agent.Agent.act` builds `[observation,
    configuration]` and truncates the call to this function's `co_argcount`
    (`agent.py:171-172`; the raw-callable path does the same at `:151-153`), so a
    one-argument entry is handed the observation alone and never sees the run's
    numbers. Measured on this machine: a 1-arg callable received
    `day, farms, hour, market, player, private, remainingOverageTime, step, town`
    and no configuration; a 2-arg callable received all fifteen configuration
    keys with the run's overrides applied (`farmHandCostMult`, `shedCapacity`,
    the town's intervals, `boardSize`, `turnsPerDay`, `weedSpawnChance`, ...).

    `config` defaults to None so a direct caller (a test, a probe, FastSim) may
    still pass the observation alone; `EngineTerms` then answers with the world's
    transcription of the engine's own defaults — one place, and never a second
    copy of a number.
    """
    return AGENT(obs, config)
