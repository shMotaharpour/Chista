"""The turn's clock, and the memory between days.

Everything that decides is in `planner/`. What is left is real work all the
same, and it is the work nobody had done: giving the solve a deadline it will
respect, and keeping what one day found so the next does not pay for it again.

## Why the memory matters more than the deadline

A cold master solve on a day-0 board is 58 pricing rounds and 936 ms. Handed
its own column pool back it certifies the SAME objective in one round and
21 ms. A column is a plan, and a plan is still a plan when the prices move —
so the pool is carried across days and the pricing step spends its rounds on
what is missing rather than rediscovering what is not.

The pool is matched by class KEY and not by class index. An index is
positional: tomorrow's class 3 is not today's, and a plan for an empty field
arriving as a plan for a grown crop is a fiction the master would then commit
tiles to.

## What the 24 turns are for

`actTimeout` is per turn, and hours 1..23 replay a plan already made — about
23 free seconds a day, 690 a season. Today's plan has to exist at hour 0, so
those turns cannot help it. They go into the pool, which is what tomorrow's
hour 0 starts from.

## It does not catch its own errors

A manager that swallows an exception plays a worse policy and looks exactly
like one that worked. That is how a greedy rung answered every turn for four
days while nothing was wired. The entry point catches, records that the
manager failed, and passes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from agent.config import Config
from agent.planner import columns as C
from agent.planner import day as D
from agent.planner import master as M
from agent.planner.colgen import classes_of
from agent.planner.inputs import GRAPH_PATH
from agent.world.rules import TURNS_PER_DAY

IDLE_PLAN = {"units": [[["PASS"]] * TURNS_PER_DAY], "market": []}

#: The pretrained rival model, loaded ONCE per process (#95).
#:
#: `OpponentModel(pretrained=True)` reads `agent/artifact/opponent_counts.npz`
#: (2,811 states) and takes 1,222 ms measured — more than the whole hour-0 turn
#: budget (965 ms), so it cannot be loaded inside a turn. It is warmed at import
#: by `agent/runtime.py` and shared by every manager.
#:
#: An artifact that is missing or empty gives `None` rather than an empty table:
#: the caller then passes no model at all and the queue keeps its uniform spread,
#: instead of re-timing the day on a table with nothing in it.
_OPPONENT: object | None = None
_OPPONENT_TRIED = False


def opponent_model() -> Any:
    """The rival model, or None when there is no trained table to read."""
    global _OPPONENT, _OPPONENT_TRIED
    if not _OPPONENT_TRIED:
        _OPPONENT_TRIED = True
        try:
            from agent.belief.opponent import OpponentModel
            model = OpponentModel(pretrained=True)
            _OPPONENT = model if getattr(model, "counts", None) else None
        except Exception:                       # noqa: BLE001 - belief is optional
            _OPPONENT = None
    return _OPPONENT


class Manager:
    """`observe` at hour 0, `step` every turn, `best` whenever asked."""

    def __init__(self, config: Config | None = None, graph=None) -> None:
        self.cfg = config or Config.load()
        self.graph = graph if graph is not None else _load_graph()
        self.keys = frozenset(self.graph.key_index)
        self.contractor = _contractor(self.cfg)
        self.steps = C.shed_distance()
        self.pool: list = []            # columns carried between days
        self.duals = None               # yesterday's published prices
        self.plan: dict = dict(IDLE_PLAN)
        self.day: D.DayPlan | None = None
        self.obs = None
        self.config = None
        self.certified = False
        #: The rival's own history, fed every observation (#95). Cheap: 0.1 ms
        #: measured per call. Its `activity_bucket` is what selects the regime
        #: row of the trained table, so the model needs the tracker and the
        #: tracker needs every turn, not only the day's first.
        self.tracker = None
        self.opponent = opponent_model()

    # -- the day ----------------------------------------------------------
    def observe(self, obs, config=None) -> None:
        """Start a day: solve inside the turn's budget and commit a plan.

        Today's plan must exist NOW — the units act this turn — so this is the
        one call that may spend the whole budget. What it finds goes into the
        pool either way.
        """
        started = time.perf_counter()
        self.obs, self.config = obs, config
        supply = M.supply_from_obs(obs)
        owned = M._owned_states(object(), obs)
        _reps, _counts, of_tile = classes_of(
            owned, M._owned_distances(obs, self.steps))
        class_of_tile = self._class_of_tile(obs, of_tile)

        deadline = started + self.cfg.solve_budget_ms / 1000.0
        # ONE forecast per turn, handed to both consumers: the master prices its
        # objective from it and `market_queue` re-times the day's sells against
        # it. Without the hand-off belief built the same curve twice — 1.1 ms
        # for the master and 1.4 ms inside the queue — over two horizons.
        forecast_obj = self._forecast(obs, config)
        self.day = D.plan(obs, self.contractor, supply,
                          class_of_tile=class_of_tile,
                          iter_cap=self.cfg.master_rounds,
                          hands=0, max_hands=self.cfg.max_hands,
                          budget_s=self.cfg.search_budget_s,
                          rounds=self.cfg.fit_rounds,
                          pool=self.pool, deadline=deadline,
                          forecast_obj=forecast_obj,
                          smoothing=self.cfg.smoothing)
        self.pool = list(self.day.master.pool)
        self.duals = self.day.master.w
        self.certified = bool(self.day.master.certified)
        self._watch(obs)
        self.plan = D.compile(self.day, obs, hands=self.day.hands,
                              config=config, model=self.opponent,
                              activity=self._activity(),
                              forecast_obj=forecast_obj)

    def _forecast(self, obs, config):
        """This turn's market forecast, or None when belief cannot build one.

        `market_queue` builds its own when it is not handed one, so a failure
        here costs the duplicate work and nothing else — the degrade is the
        behaviour that shipped, not a second policy.
        """
        try:
            from agent.belief.market import forecast
            from agent.belief.shed import SEASON_DAYS
            day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
            horizon = max(self.contractor.days, SEASON_DAYS - day)
            return forecast(obs, days=horizon, config=config)
        except Exception:                      # noqa: BLE001 - belief is optional
            return None

    def _watch(self, obs) -> None:
        """Feed the rival tracker, building it on the first observation.

        Not in `__init__`: the seat index is in the observation (`player`), not
        in the config, and a tracker built for the wrong seat reads the wrong
        farm's flows without saying so.

        The record the tracker returns is handed to the model as well: belief's
        documented chain is `tracker.observe() -> model.observe(rec, tracker) ->
        expected_sell(...)`, and the middle link was the one nobody walked, so
        the pretrained table was frozen for the whole season while the queue
        read its predictions.
        """
        if obs is None:
            return
        if self.tracker is None:
            from agent.belief.tracker import MarketTracker
            player = int(obs.get("player", 0)) if isinstance(obs, dict) else 0
            self.tracker = MarketTracker(player=player)
        record = self.tracker.observe(obs)
        if record is not None and self.opponent is not None:
            self.opponent.observe(record, self.tracker)

    def _activity(self) -> int | None:
        """The rival's own sell bucket over the last day, or None if unwatched.

        Bucket 0 is "no history yet", which is a row the trained table has, so
        day 0 is priced from the start regime rather than from no regime.
        """
        if self.tracker is None or self.tracker.step < 0:
            return None
        return int(self.tracker.activity_bucket(self.tracker.step))

    def step(self, obs=None, budget_ms: float | None = None) -> bool:
        """Spend a turn improving the pool. Today's plan is not touched.

        It cannot be: the units have already acted on it. What this buys is
        tomorrow's hour 0, which starts from a pool that is closer to done.
        Returns whether the master's answer is certified.

        `obs` is this turn's observation and is optional: it feeds the rival
        tracker (the pool work below needs no observation at all), so a caller
        that only wants the pool can leave it out.
        """
        self._watch(obs)
        if self.obs is None or self.certified:
            return self.certified
        budget = self.cfg.solve_budget_ms if budget_ms is None else budget_ms
        result = M.equilibrate(object(), self.obs, self.contractor,
                               M.supply_from_obs(self.obs),
                               iter_cap=self.cfg.master_rounds,
                               pool=self.pool,
                               deadline=time.perf_counter() + budget / 1000.0)
        if not result.used_fallback:
            self.pool = list(result.pool)
            self.certified = bool(result.certified)
        return self.certified

    def best(self) -> dict:
        """The day's plan. Never None once a day has been observed.

        Idle before the first observe, and idle for a day nothing could carry —
        which is legal, and the right answer for a farm that cannot act.
        """
        return self.plan

    # -- the board --------------------------------------------------------
    def _class_of_tile(self, obs, of_tile) -> list:
        """The class index of every board position, None where nothing is planned.

        `of_tile` covers the tiles the master priced, in board order; the rest
        of the board is a quadrant we have not bought (F042) and has no class.
        """
        from agent.obs import decode_world
        view = decode_world(obs, at_day_start=True, graph_keys=self.keys)
        walker = iter(of_tile)
        return [next(walker, None) if int(k) >= 0 else None
                for k in np.asarray(view.me.keys).reshape(-1)]


def _load_graph():
    from agent.tile_dp.graph import TileGraph
    return TileGraph.load(GRAPH_PATH)


def _contractor(cfg: Config):
    from agent.planner.inputs import load_contractor
    return load_contractor(days=cfg.horizon_days)
