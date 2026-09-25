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

## The day moves, and so does everything indexed by it

The horizon is the season that is LEFT (F029): 30 days on day 0, 29 on day 1,
..., 1 on day 29. There is no horizon after the season ends, and a fixed
look-ahead is wrong at both ends of it — 20 days on day 0 stops the DP short of
the season's own end, 20 days on day 20 prices ten days the season does not
have. `_roll_day` sets the contractor to that horizon at every day start and
moves the carried pool onto it: a column is a plan for the days of the board it
was built on, so its day-indexed arrays shift up one (`colgen.advance_pool`),
which is also where the columns the LP gave no weight to are dropped.

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
from agent.planner.colgen import advance_pool, classes_of
from agent.planner.land_plan import (purchase_order, quadrants_bought,
                                     with_quadrant_open)
from agent.planner.inputs import GRAPH_PATH
from agent.tile_dp.contractor import HORIZON_DAYS
from agent.world.rules import TURNS_PER_DAY

IDLE_PLAN = {"units": [[["PASS"]] * TURNS_PER_DAY], "market": []}


def _sell_hours(plan) -> dict:
    """`{good: hour}` from the committed plan's own market orders.

    The FIRST hour a good is sold in is the one that matters for the walk: the
    units are in the market from then on. A plan with no sells for a good leaves
    it out, and the projection then lands that good at hour 0 — the behaviour
    that shipped.
    """
    hours: dict[str, int] = {}
    market = (plan or {}).get("market") or []
    for hour, orders in enumerate(market[:TURNS_PER_DAY]):
        for order in orders or []:
            if order and str(order[0]) == "SELL" and str(order[1]) not in hours:
                hours[str(order[1])] = int(hour)
    return hours

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
        self.contractor = _contractor(self.graph, HORIZON_DAYS)
        self.steps = C.shed_distance()
        self.pool: list = []            # columns carried between days
        #: Quadrant the day decided to buy (0 or 1). The secretary turns it into a BUY_LAND
        #: order and funds it through the day's sell queue, so the decision lives here and the
        #: price/queue live there.
        self.lands_today: int = 0
        self.land_ramp: np.ndarray | None = None
        self.land_note: str = ""
        #: The last solve's mix, in `pool` order — what the day roll prunes on.
        #: Kept BESIDE the pool and never apart from it: `generate` rebuilds the
        #: pool as [idle, warm, new] on every call, so `step` leaves the pool in
        #: a different order from the one `observe` solved, and a weight indexed
        #: against the wrong order drops the columns the last solve was using.
        self.lam: np.ndarray | None = None
        #: The day the pool's columns are indexed from. A column is a plan for
        #: the days of the board it was built on, so carrying it into another
        #: day without moving it is carrying a plan for the wrong days.
        self.pool_day: int | None = None
        self.duals = None               # yesterday's published prices
        self.forecast_obj = None
        self.plan: dict = dict(IDLE_PLAN)
        self.day: D.DayPlan | None = None
        self.obs = None
        self.config = None
        self.certified = False
        #: The sells the committed plan projects ({step: {good: units}}), fed
        #: into the NEXT day's forecast (`forecast(our_sells=)`). #110's
        #: own-supply half: the plan moves the price path it was priced on.
        self.own_sells: dict = {}
        #: The rival's own history, fed every observation (#95). Cheap: 0.1 ms
        #: measured per call. Its `activity_bucket` is what selects the regime
        #: row of the trained table, so the model needs the tracker and the
        #: tracker needs every turn, not only the day's first.
        self.tracker = None
        self.opponent = opponent_model()

    # -- the day ----------------------------------------------------------
    def _roll_day(self, obs) -> None:
        """Move the carried memory onto today's board: the horizon and the pool.

        Both are a function of the day and neither may be stale.

        **The horizon is the season that is LEFT** (F029) and the contractor
        prices over it, so the DP's terminal row is the season's own end rather
        than a cliff inside it: 30 days on day 0, 29 on day 1, ..., 1 on day 29.
        The contractor is rebuilt from the graph already in hand when the day
        changes — measured 0.4 ms against the artifact read — and everything
        downstream reads `contractor.days`: the master's LP, `to_mixes`, the
        hire bill and the projected sells.

        **The pool is a plan for the days of the board it was built on**, so
        every column moves up with the day (`colgen.advance_pool`) and the ones
        the LP gave no weight to are dropped there. `self.lam` is the mix of the
        solve that produced `self.pool` — `step` re-solves and reorders it, so
        the pair is what travels, never a remembered weight.
        """
        day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
        days = M.season_horizon(obs)
        if int(self.contractor.days) != days:
            self.contractor = _contractor(self.graph, days)
        if self.pool_day is None:                 # nothing carried yet
            self.pool_day = day
            return
        step = day - self.pool_day
        if step <= 0:                             # same day, or a re-observe
            return
        self.pool = advance_pool(self.pool, self.lam, step=step)
        self.lam = None                           # the new day has not solved yet
        self.pool_day = day

    def observe(self, obs, config=None) -> None:
        """Start a day: solve inside the turn's budget and commit a plan.

        Today's plan must exist NOW — the units act this turn — so this is the
        one call that may spend the whole budget. What it finds goes into the
        pool either way.
        """
        started = time.perf_counter()
        self.obs, self.config = obs, config
        # Today's horizon, and the pool moved onto it. Before anything reads
        # either: `supply`/`class_of_tile` are day-invariant, but the contractor
        # is not, and `_forecast` sizes its walk from the horizon.
        self._roll_day(obs)
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
        #
        # #110's own-supply half rides along: the sells the plan we committed
        # YESTERDAY projects go into today's path (the day-over-day fixed
        # point). Measured on a day-3 MILK-heavy plan, the flat path overstated
        # its earn by ~16% — the ladder walks down under your own supply too.
        self.forecast_obj = self._forecast(obs, config)
        # --- land: the plan's OWN decision, inside the master LP --------------------------
        # The purchase is a soft ramp in the LP (`colgen.MasterLP.solve`): `s_d` is the scale
        # bought BY day d, its price is charged on the step in the objective AND in the cash
        # rows, and the quadrant's tiles enter the class counts. So the day needs no separate
        # valuation to be told what land is worth - it competes with the long crop for the same
        # coin, in the same model, and answers with a DAY.
        #
        # It is handed to this one solve, NOT to a second one on a what-if board. The earlier
        # shape solved a second plan against its own board, and that was wrong twice over: the
        # second solve ran on the day's spent clock (measured: fresh deadline 90,933.6, same
        # deadline spent 0.0, fresh again 90,933.6), and even when it did not, the day's own
        # spending never reserved the coins the purchase needed - the failure the owner named,
        # the manager buying long crops and never saving for the land.
        self.lands_today = 0
        self.land_note = ""
        #: (days,) the LP's own ramp: what the plan decided to buy by each day. None when the
        #: day was not offered a purchase (nothing left to buy). Read off the same LP as the
        #: plan, so a report can say WHEN the plan wanted the land, not just whether.
        self.land_ramp: np.ndarray | None = None
        land_option = None
        step = purchase_order(quadrants_bought(obs))
        if step is not None:
            try:
                added = _tiles_added_by(obs, step[0], self.steps)
                if float(added.sum()) > 0.0:
                    land_option = (float(step[1]), added)
            except Exception as exc:                # a land option must never break the day
                self.land_note = f"{type(exc).__name__}: {exc}"
        self.day = D.plan(obs, self.contractor, supply,
                          class_of_tile=class_of_tile,
                          land=land_option,
                          iter_cap=self.cfg.master_rounds,
                          hands=0, max_hands=self.cfg.max_hands,
                          budget_s=self.cfg.search_budget_s,
                          rounds=self.cfg.fit_rounds,
                          pool=self.pool, deadline=deadline,
                          forecast_obj=self.forecast_obj,
                          smoothing=self.cfg.smoothing)
        self.pool = list(self.day.master.pool)
        self.lam = self.day.master.lam
        self.duals = self.day.master.w
        self.certified = bool(self.day.master.certified)
        self._project_own_sells()
        self._watch(obs)
        # The ramp the ONE plan already decided. The tiles are workable from the day AFTER the
        # purchase (the market settles it in the turn it is ordered), so the LP works them from
        # `s_{d-1}`: `s_0` is the statement "the tiles are needed tomorrow", i.e. buy today.
        ramp = self.day.master.land
        self.land_ramp = None if ramp is None else np.asarray(ramp, dtype=np.float64)
        if (self.land_ramp is not None and self.land_ramp.size
                and float(self.land_ramp[0]) >= 0.5):
            self.lands_today = 1
        self.plan = D.compile(self.day, obs, hands=self.day.hands,
                              config=config, model=self.opponent,
                              activity=self._activity(),
                              forecast_obj=self.forecast_obj,
                              lands=self.lands_today,
                              # The rival's dated supply, gated to the days their
                              # board says have goods: the risk half of the sell
                              # rank. One day is all the rank reads.
                              rival_supply=self._rival_hours(obs, 1))

    def _forecast(self, obs, config):
        """This turn's market forecast, or None when belief cannot build one.

        `market_queue` builds its own when it is not handed one, so a failure
        here costs the duplicate work and nothing else — the degrade is the
        behaviour that shipped, not a second policy.

        The rival's own supply goes in DATED (`_rival_supply`): their board is
        public, so the days their planted tiles pay out are derivable, and a
        price path that assumes no rival supply is a path that ignores half the
        board.

        OUR projected supply goes in too (`self.own_sells`, #110): the plan
        committed yesterday moves today's path. None is not a second policy —
        it is the day-0 case, before any plan exists to project.

        The walk covers the season that is LEFT (F029) and not a fixed
        look-ahead: the master prices its product rows off this path, so a path
        longer than the horizon is wasted work and a shorter one is padded with
        a quote for a day that does not exist.
        """
        try:
            from agent.belief.market import forecast
            horizon = M.season_horizon(obs)
            return forecast(obs, days=horizon, config=config,
                            our_sells=self.own_sells or None,
                            rival_supply=self._rival_supply(obs, horizon),
                            rival_sells=self._rival_hours(obs, horizon) or None)
        except Exception:                      # noqa: BLE001 - belief is optional
            return None

    def _rival_hours(self, obs, horizon: int) -> dict:
        """The rival's DATED supply: what they sold, then what the model expects.

        `_rival_supply` is the calendar — which days they pay out, not which
        hour — so the walk dated all of it at hour 0 and the rest of the day was
        quoted on a market they never sold into (#16). Two sources know the hour:
        the tracker's inferred `rival_sales` for the turns already recorded, and
        the opponent model's `expected_sell` for the rest, evaluated at the
        quotes in hand.

        A failure here leaves the calendar's daily curve standing: one degrade,
        never a second policy.
        """
        from agent.belief.market import PRODUCTS
        try:
            out: dict[int, dict[str, int]] = {}
            tracker = self.tracker
            if tracker is not None:
                for rec in getattr(tracker, "records", ()) or ():
                    units = np.asarray(rec.rival_sales, dtype=np.float64)
                    for i, good in enumerate(PRODUCTS):
                        n = int(round(float(units[i])))
                        if n > 0:
                            out.setdefault(int(rec.step), {})[good] = n
            model = self.opponent
            if model is not None and getattr(model, "counts", None):
                activity = self._activity()
                quotes = (obs.get("market") or {}).get("prices") or {}
                step0 = int(obs.get("step", int(obs.get("day", 0))
                                     * TURNS_PER_DAY))
                # The gate: the model only says WHICH HOUR of a day that the
                # board already says has goods. On a day the calendar is empty
                # there is nothing to date, and the model's priors would invent
                # a rival who sells — measured at 4,058 coins against a seat
                # that does nothing.
                calendar = np.asarray(self._rival_supply(obs, horizon),
                                      dtype=np.float64)
                for d in range(max(1, int(horizon))):
                    if float(calendar[d].sum()) <= 0.0:
                        continue
                    for h in range(TURNS_PER_DAY):
                        step = step0 + d * TURNS_PER_DAY + h
                        for good in PRODUCTS:
                            price = int(quotes.get(good, 0))
                            if price <= 0:
                                continue
                            n = int(round(float(
                                model.expected_sell(good, step, price,
                                                    activity=activity))))
                            if n > 0:
                                out.setdefault(step, {})[good] = n
            return out
        except Exception:                      # noqa: BLE001
            if not self._degrades():           # see `_project_own_sells`
                raise
            return {}

    def _rival_supply(self, obs, horizon):
        """The rival's dated supply curve, or None when it cannot be read.

        `(horizon, 9)` units per day per good, from their packed tile states
        (`belief/rival_calendar.supply_curve`). Measured 0.55 ms against the
        forecast's own 0.70 ms, so it goes on the per-turn path as it is.

        None is not a second policy: `forecast` then falls back to its flat
        `residual` input, which is what a caller without a calendar has.
        """
        try:
            from agent.belief.market import PRODUCTS
            from agent.belief.rival_calendar import supply_curve
            return supply_curve(obs, tuple(PRODUCTS), horizon)
        except Exception:                      # noqa: BLE001 - the flat path stands
            return None

    def _project_own_sells(self) -> None:
        """Record the committed plan's projected sells, for tomorrow's path.

        `plan_supply_sells_from_pool` reads the solved mix (`master.lam` x the
        columns' `produce`) — the plan the units are ABOUT to act on. It feeds
        `forecast(our_sells=)` on the next observation, so tomorrow's objective
        is priced on a path that includes what today's plan dumps. A failure
        leaves `own_sells` empty, which is the no-supply assumption, not an
        error (R002: one degrade, the flat path).
        """
        try:
            from agent.planner.plan_supply import plan_supply_sells_from_pool
            day = int(self.obs.get("day", 0)) if self.obs else 0
            # The columns' produce covers `contractor.days` only — asking
            # for the season horizon beyond it filters every column out
            # (shape[0] >= days fails), so project over the column horizon.
            solve = self.day.master if self.day is not None else None
            lam = getattr(solve, "lam", None)
            pool = getattr(solve, "pool", None)
            if lam is None or not len(lam) or pool is None:
                self.own_sells = {}
                return
            self.own_sells = plan_supply_sells_from_pool(
                pool, lam, self.contractor.days, day,
                hours=_sell_hours(self.plan))
        except Exception:                      # noqa: BLE001
            # The flat path stands only when the run ASKED for it. Swallowing
            # unconditionally is how a broken projection hides: `own_sells`
            # stays empty, the walk prices a market without our own supply, and
            # nothing in the season's log says why (the same shape as the
            # NameError that hid behind `_rival_hours`'s except).
            if not self._degrades():
                raise
            self.own_sells = {}

    def _degrades(self) -> bool:
        """May a broken optional wire fall back instead of raising?

        The run says so, once, through `Config.never_raise` — the same switch the
        runtime already honours at the turn boundary. Off (the shipped default)
        means the error spills where it can be seen.
        """
        return bool(getattr(getattr(self, "cfg", None), "never_raise", False))

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
        days = int(np.asarray(self.contractor.days))
        hands = getattr(self.day, "hands", 0) if self.day is not None else 0
        supply = M.supply_from_obs(self.obs)
        if hands > 0:
            hours = D.hours_for(hands, days)
            supply = M.CouplingSupply(
                hours=hours, seed_stock=supply.seed_stock,
                animal_stock=supply.animal_stock, fert_stock=supply.fert_stock,
                wheat_feed_stock=supply.wheat_feed_stock, money=supply.money,
                quotes=supply.quotes, shed_stock=supply.shed_stock,
                shed_capacity=supply.shed_capacity)
        result = M.equilibrate(object(), self.obs, self.contractor,
                               supply,
                               iter_cap=self.cfg.master_rounds,
                               pool=self.pool,
                               deadline=time.perf_counter() + budget / 1000.0,
                               forecast_obj=self.forecast_obj,
                               smoothing=self.cfg.smoothing)
        if not result.used_fallback:
            self.pool = list(result.pool)
            self.lam = result.lam           # the mix of THIS pool, in its order
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
        Tiles with states outside the graph get None and do not consume from
        `walker`, keeping subsequent tiles aligned with their own classes (#152).
        """
        from agent.obs import LOCKED_KEY, decode_world
        view = decode_world(obs, at_day_start=True, graph_keys=self.keys)
        walker = iter(of_tile)
        return [next(walker, None)
                if int(k) != LOCKED_KEY and int(k) in self.keys
                else None
                for k in np.asarray(view.me.keys).reshape(-1)]


def _tiles_added_by(obs, quadrant: str, steps) -> np.ndarray:
    """The coming quadrant's tiles, counted in the BASE board's own classes.

    The LP knows the classes the day already has, so the new capacity has to be expressed in
    them. Matching is by the class's own representative and never by position: `classes_of`
    orders classes by what it finds, and the board with the quadrant open is a different board.

    A tile whose class the base board does not have is DROPPED, and that is deliberate - there
    is no column for it in this pool, so counting it would promise capacity nothing can use.
    """
    owned = M._owned_states(object(), obs)
    reps, counts, _of = classes_of(owned, M._owned_distances(obs, steps))
    obs_buy = with_quadrant_open(obs, quadrant, 0)
    owned_buy = M._owned_states(object(), obs_buy)
    reps_b, counts_b, _of_b = classes_of(owned_buy, M._owned_distances(obs_buy, steps))
    by_rep = {tuple(np.asarray(r).reshape(-1)): i for i, r in enumerate(reps_b)}
    added = np.zeros(len(counts), dtype=np.float64)
    for i, rep in enumerate(reps):
        j = by_rep.get(tuple(np.asarray(rep).reshape(-1)))
        if j is not None:
            added[i] = float(counts_b[j]) - float(counts[i])
    return added


def _load_graph():
    from agent.tile_dp.graph import TileGraph
    return TileGraph.load(GRAPH_PATH)


def _contractor(graph, days: int):
    """The tile DP's pricing oracle, cast over the graph already in hand.

    `days` is the horizon the sweep runs over, so it changes with the day
    (`_roll_day`). Building it from `self.graph` rather than through
    `load_contractor` keeps the artifact read to the one the manager already
    paid for: `TileGraph.load` per day would be a file read per day for a table
    that never changes.
    """
    from agent.tile_dp.contractor import TileContractor
    return TileContractor(graph, days=days)
