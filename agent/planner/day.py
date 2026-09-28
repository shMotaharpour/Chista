"""The assignment, asked of the day layer: does this plan's FIRST day fit?

The master's labour row charges each worked day the ops it runs plus the walk
to reach the tile. That is a lower bound and it is deliberately one — a route
may walk further, never less — so the master's answer is an upper bound on
what the farm can do, and something has to tell it how much further.

`wsr` is that something. It takes the day's chains and says whether a pool of
hands can actually walk them, and the gap between what it can carry and what
the master committed is the correction the hours row has been missing. The
#12 brief put a number on the placeholder and said so out loud:

    `H_d` = 24·(1 + hands_d) minus the travel/carry overhead the secretary
    reports. In M3 this is a flat constant (start at 35 % and measure); in M4
    it is fed back from the actual routing solve.

This is M4. Nothing here decides anything: it asks, and it reports.
"""

from __future__ import annotations

from typing import Any
from dataclasses import replace, dataclass

from agent.config import Config
from agent.world.terms import EngineTerms

import numpy as np

from agent.tile_dp.chains import chain_ops
from agent.world.rules import BOARD_SIZE, TURNS_PER_DAY


@dataclass(frozen=True)
class DayFit:
    """What the day layer made of the master's first day."""

    chains: tuple                  # (cell, ops, entity) per committed tile
    placed: int                    # tasks wsr walked
    tasks: int                     # tasks the day asked for
    pool: int                      # hands the route used
    complete: bool                 # every task placed
    hours_used: float              # worker-hours the route actually spent
    hours_committed: float         # what the master's row charged for day 0
    reason: str = ""               # "" when it fits: "hours", "budget", "unstable"
    #: Worker-turns the route left unspent, walks included (`Result.spare`): what
    #: more work the SAME hands could carry. The search computes it and it used to
    #: stop here — but it is the number the manager needs to decide whether to lay
    #: more on the day, hire more, or confirm.
    spare: int = 0
    #: The arithmetic floor on hands the day's own work needs (`B.lower_bound`
    #: less the farmer): the MINIMUM a shortfall is measured against, so the
    #: manager can decide to offer it or lay less on the day instead of walking
    #: up one hand at a time and paying for a search each step.
    floor: int = 0
    #: How many extra hands the day needed before it was carried, 0 when the
    #: priced pool was enough. The number the manager decides on: hire them, lay
    #: less on the day, or accept a day that does not fit.
    short: int = 0

    @property
    def overhead(self) -> float:
        """What the master's day-0 row understated the real day by, as a ratio.

        1.0 means the row was right. Above it, the route spent more than the
        row charged — which is the #12 brief's flat 35 %, now a measurement.
        Undefined with nothing committed, and reported as 1.0 rather than as a
        division by zero: no work is not an overhead of infinity.
        """
        if self.hours_committed <= 0.0:
            return 1.0
        return self.hours_used / self.hours_committed


def day_chains(choices, mixes, pool, board_size: int = BOARD_SIZE) -> tuple:
    """The assignment -> wsr's `(cell, ops, entity)` for day 0.

    A tile whose day-0 chain is empty is not in the day. That is the DP
    declining to act on it at these prices, not the caller dropping it — a
    dropped tile comes back unchanged while a declined one was re-planned and
    chose to wait, and the difference is the whole point of re-pricing.
    """
    out = []
    for position, choice in enumerate(choices):
        if choice is None:
            continue
        mix = mixes.get(choice.class_key)
        if mix is None or choice.plan_index >= len(mix.plans):
            continue
        plan = mix.plans[choice.plan_index]
        if not plan.chains:
            continue
        ops = chain_ops(int(plan.chains[0]))
        if not ops:
            continue
        column = _column_for(pool, choice, mix)
        entity = (column.entities[0]
                  if column is not None and column.entities else None)
        cell = (position % board_size, position // board_size)
        out.append((cell, tuple(ops), entity))
    return tuple(out)


def _column_for(pool, choice, mix):
    """The `colgen.Column` behind a `Choice`, by class and plan order.

    `to_mixes` builds a class's plans in pool order, so the n-th plan of a
    class is the n-th column of that class. Recovering it by walking the pool
    keeps that the only place the ordering is assumed.
    """
    seen = 0
    for column in pool:
        if column.cls != choice.class_key:
            continue
        if seen == choice.plan_index:
            return column
        seen += 1
    return None


def _find_plan_with_op(mix, op_name: str) -> int | None:
    """Find the best plan index in a class mix that includes `op_name` on day 0."""
    best_pi = None
    best_score = -float("inf")
    lam = getattr(mix, "lam", None)
    for pi, plan in enumerate(mix.plans):
        if not plan.chains:
            continue
        ops = chain_ops(int(plan.chains[0]))
        if op_name in ops:
            score = (float(lam[pi]) if lam is not None and pi < len(lam)
                     else float(getattr(plan, "revenue", 0.0)))
            if score > best_score:
                best_score = score
                best_pi = pi
    return best_pi


def protect_at_risk_assignments(choices: list[Any],
                                mixes: dict[int, Any],
                                obs: dict | None,
                                board_size: int = BOARD_SIZE) -> list[Any]:
    """Ensure at-risk animals and crops receive survival ops on day 0 (#149).

    Animals with consecutive_unfed >= 1 escape tonight unless fed today (F017).
    Crops with consecutive_unwatered >= 1 become weeds tonight unless watered (F001).
    Quota rounding can assign them an idle plan (Plan 0) when labour is tight.
    This promotes their choice to the best plan in their class that contains
    the necessary survival op ('FEED' for animals, 'WATER' for plants) on day 0.
    """
    from agent.planner import columns as C
    if not obs or not isinstance(obs, dict):
        return choices
    player = int(obs.get("player", 0))
    farms = obs.get("farms") or []
    if player >= len(farms):
        return choices
    tiles = farms[player].get("tiles") or []
    out = list(choices)
    for pos, choice in enumerate(out):
        if choice is None:
            continue
        y, x = pos // board_size, pos % board_size
        if y >= len(tiles) or x >= len(tiles[y]):
            continue
        t = tiles[y][x]
        if not isinstance(t, dict):
            continue
        kind = t.get("kind")
        needed_op = None
        if kind in ("COOP", "PASTURE") and t.get("animal"):
            if int(t.get("consecutive_unfed", 0)) >= 1:
                needed_op = "FEED"
        elif kind == "PLANT":
            if int(t.get("consecutive_unwatered", 0)) >= 1:
                needed_op = "WATER"
        if needed_op:
            mix = mixes.get(choice.class_key)
            if mix and choice.plan_index < len(mix.plans):
                plan = mix.plans[choice.plan_index]
                ops = chain_ops(int(plan.chains[0])) if plan.chains else ()
                if needed_op not in ops:
                    better_pi = _find_plan_with_op(mix, needed_op)
                    if better_pi is not None:
                        out[pos] = C.Choice(choice.class_key, better_pi)
    return out


def availability(obs, chains) -> dict:
    """wsr's `available` for this day, from the observation's own stocks."""
    from agent.planner.market import availability as _availability
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    return _availability(chains, dict(private.get("seeds", {}) or {}),
                         dict(private.get("shed", {}) or {}))


def fit(chains, *, hands: int, available: dict | None = None,
        hire_times: tuple[int, ...] | None = None,
        beam: int | None = None,
        hours_committed: float = 0.0, warm=None) -> DayFit:
    """Hand the day to wsr and report what it made of it.

    `hands` is the pool offered, all of them from hour 1 — a hand hired in turn
    0 first acts at hour 1 (F040). The search is given the RANGE `floor..hands`
    and finds the smallest pool itself: wsr's `_smallest_pool` halves the
    interval (O(log2 N)) and contracts its ceiling to the hands it actually
    used, which is both the cheap way to ask and the only way to learn the
    answer is smaller than `hands`. There is no wall-clock deadline to pass —
    wsr is bounded structurally by the task count and the beam width, so
    identical inputs give bit-identical routes (agent/wsr/README.md, caller
    guidelines 3 and 4).
    """
    from agent.wsr import beam as B
    from agent.wsr import tasks as T
    from agent.wsr.emit import check_route

    if not chains:
        return DayFit((), 0, 0, 0, True, 0.0, hours_committed)

    available = available or {}
    tasks = T.build(chains, available=available)
    # `hands` is the offer the master priced, and zero is an offer: the farmer walks alone. The
    # hours each hand starts at are the engine's (`rules.hire_hour`), which `Day` fills in.
    from agent.world.rules import earliest_hire_times

    # The hours are the HOURLY secretary's when it has laid the day out
    # (`DayMarket.hire_hours`: the settlement turn plus one, F040). Only when it
    # has not does the engine's earliest bound stand in — and that bound assumes a
    # whole turn's order budget goes to hires, which the real queue does not.
    day = B.Day(chains=tuple(chains), available=available,
                hire_times=(tuple(hire_times) if hire_times is not None
                            else earliest_hire_times(hands)))
    floor = max(0, B.lower_bound(day, tasks) - len(day.units))
    result = B.search(day, tasks, beam=beam,
                      hands=min(floor, hands), max_hands=hands, warm=warm)

    hours = float(max((turn for turn, _t, _w in result.route), default=0) + 1) \
        * max(1, result.pool)
    if result.complete and check_route(day, tasks, result):
        # A route the compiler will not take is a day that did not fit, which
        # is an answer the caller already knows how to use (#73 is why this is
        # checked here rather than discovered inside `compile_route`).
        return DayFit(tuple(chains), len(result.route), tasks.n, result.pool,
                      False, hours, hours_committed, "unstable",
                      spare=int(result.spare), floor=int(floor))
    reason = "" if result.complete else ("budget" if result.can_improve
                                         else "hours")
    return DayFit(tuple(chains), len(result.route), tasks.n, result.pool,
                  bool(result.complete), hours, hours_committed, reason,
                  spare=int(result.spare), floor=int(floor))


def hours_for(hands: int, days: int, overhead: float) -> np.ndarray:
    """The labour a day holds with `hands` hired, per day of the horizon.

    `24·(1 + hands) − hands`: the farmer's full day plus one per hand, less the
    hour each hand loses to being hired (F040 — a hand hired in turn 0 first
    acts at hour 1). The overhead is #12's M3 placeholder for the walking a
    route does beyond what a column charges; the columns now carry the walk to
    the tile, so what is left is the walking BETWEEN them.
    """
    gross = 24.0 * (1 + int(hands)) - int(hands)
    return np.full(days, gross * (1.0 - overhead))


def hire_bill(hands: int, hires_today: int = 0,
              multiplier: int | None = None) -> int:
    """What `hands` hires cost today. Fibonacci, and it resets nightly (F039).

    One formula with the LP's: `rules.hire_cost(n, multiplier)`, where
    `multiplier` is the run's own `farmHandCostMult` when the caller resolved it.
    """
    from agent.world.rules import hire_cost
    return sum(hire_cost(int(hires_today) + i, multiplier)
               for i in range(max(0, int(hands))))


@dataclass(frozen=True)
class DayPlan:
    """A master solve, its assignment, and what the day layer made of it."""

    master: object                 # the MasterResult
    choices: list                  # one Choice (or None) per board position
    mixes: dict                    # class index -> ClassMix
    day: DayFit
    rounds: int = 1                # which solve this candidate came from
    overhead: float = 1.0          # the hours correction that was applied
    hands: int = 0                 # hands this plan pays for
    net: float = 0.0               # objective less what the hands cost
    #: How many times the master was solved in total. `rounds` is the winner's
    #: own number and a later solve can lose to an earlier one, so the two are
    #: different questions: "which answer is this" and "what did it cost".
    solves: int = 1


def _better(candidate: "DayPlan", best: "DayPlan") -> bool:
    """A day that fits beats one that does not; then the honest hours row wins."""
    if candidate.day.complete != best.day.complete:
        return candidate.day.complete
    return (abs(candidate.day.overhead - 1.0)
            < abs(best.day.overhead - 1.0) - 1e-9)


def plan(obs, contractor, supply, *, class_of_tile, iter_cap: int | None = None,
         hands: int = 0,
         terms: "EngineTerms | None" = None,
         rounds: int | None = None, tolerance: float | None = None,
         pool: list | None = None,
         max_hands: int | None = None, w_warm=None,
         forecast_obj=None, smoothing: float | None = None,
         cfg: "Config | None" = None) -> DayPlan:
    """Enumerate the pool of hands, and keep the day worth the most net of it.

    **Hiring is a decision, and it was not one.** `supply.hours` came from
    `len(farm["hands"])`, which is zero at every hour 0 because the engine
    clears the field overnight (F040) — so the master planned for one farmer
    for thirty days, and a season worked three to five tiles of twenty-five,
    banked its money and never bought a quadrant. #12's brief said to treat
    `hands_d` as an outer enumeration and this is it. What it is worth:

        hands   H_d     hire    objective    tiles
            0   15.6       0       34,197        5
            1   30.6       1       57,739        8
            2   45.5       2       78,266       11
            4   75.4       7      113,812       16
            8  135.2      54      166,449       23

    Eight hands cost 54 coins against a purse of 3,000 and are worth five
    times the plan. The enumeration is cheap because each solve is handed the
    last one's column pool: the same plans are still plans at a different
    wage, so only what is missing has to be priced.

    The hire bill is charged over the WHOLE horizon, not once: the hands are
    cleared every night and hired again every morning (F039), so a plan that
    counts them once buys twenty days of labour for one day's wages.
    """
    """Master, assign, ask wsr, correct the hours, repeat.

    **wsr is a feasibility oracle here, not a calibration source, and that is
    a measurement rather than a preference.** #12's brief asks for `H_d`'s
    flat 35 % overhead to be "fed back from the actual routing solve", and the
    obvious reading — divide the hours by the ratio the route came in over —
    does not converge, because the ratio is a property of the PLAN and not of
    the board:

        hours 15.60   5 tiles   route 16 turns   committed 14.0   1.14x
        hours 13.68   4 tiles   route 18 turns   committed 14.0   1.29x
        hours 12.00   4 tiles   route 18 turns   committed 14.0   1.29x

    Fewer hours bought fewer tiles and a LONGER route: the tiles that survive
    a tighter budget are not the cheap ones to walk between. A scalar chasing
    that lands further away each time.

    So the hours only ever come DOWN, and only when the day did not fit. A day
    that fits is evidence the row was not too generous; it is not evidence of
    how much slack is left, and crediting slack back would let the next solve
    commit a day on hours nothing has proved.
    """
    from agent.planner import columns as C
    from agent.planner import master as M

    cfg = Config() if cfg is None else cfg
    # The run's own numbers (the wage multiplier among them): resolved once here
    # when the caller did not already (`Manager.observe` does).
    terms = EngineTerms.from_obs(obs) if terms is None else terms
    iter_cap = int(cfg.master_rounds if iter_cap is None else iter_cap)
    rounds = int(cfg.fit_rounds if rounds is None else rounds)
    tolerance = (float(cfg.hours_tolerance) if tolerance is None
                 else float(tolerance))
    smoothing = float(cfg.smoothing if smoothing is None else smoothing)
    max_hands = int(cfg.max_hands if max_hands is None else max_hands)
    ceiling = int(hands if max_hands is None else max_hands)
    carried = list(pool or [])
    chosen: DayPlan | None = None
    # Every offer is solved, largest pool FIRST: more hands is where the value
    # is (0 hands 34,197, eight hands 166,449 for 54 coins), so a caller that
    # caps the enumeration shorter than this keeps a good day instead of the
    # emptiest one. What this function no longer does is cut the walk short on a
    # clock: the offers are all priced and the best NET wins.
    for offer in range(max(0, ceiling), -1, -1):
        current = _solve_at(obs, contractor, supply, class_of_tile, offer,
                            iter_cap, rounds, tolerance, carried,
                            w_warm, forecast_obj, smoothing, cfg, terms)
        carried = list(current.master.pool)
        if chosen is None or current.net > chosen.net:
            chosen = current
    return chosen


def _solve_at(obs, contractor, supply, class_of_tile, hands, iter_cap,
              rounds, tolerance, pool, w_warm=None,
              forecast_obj=None, smoothing: float = 0.0,
              cfg: "Config | None" = None,
              terms: "EngineTerms | None" = None) -> DayPlan:
    """One pool size: solve, assign, ask wsr, and price the hands."""
    from agent.planner import columns as C
    from agent.planner import master as M

    cfg = Config() if cfg is None else cfg
    terms = EngineTerms.from_obs(obs) if terms is None else terms
    days = int(np.asarray(supply.hours).size)
    hours = hours_for(hands, days, cfg.hours_overhead)
    best: DayPlan | None = None
    applied = 1.0
    candidate: DayPlan | None = None

    for spent in range(1, max(1, rounds) + 1):
        current = M.CouplingSupply(
            hours=hours, seed_stock=supply.seed_stock,
            animal_stock=supply.animal_stock, fert_stock=supply.fert_stock,
            wheat_feed_stock=supply.wheat_feed_stock, money=supply.money,
            quotes=supply.quotes, shed_stock=supply.shed_stock,
            shed_capacity=supply.shed_capacity)
        result = M.equilibrate(object(), obs, contractor, current,
                               w_warm=w_warm, iter_cap=iter_cap, pool=pool,
                               forecast_obj=forecast_obj,
                               smoothing=smoothing, cfg=cfg)
        mixes = M.to_mixes(result, contractor.days)
        choices = C.assign_by_quota(class_of_tile, mixes)
        choices = protect_at_risk_assignments(choices, mixes, obs)
        # The LP's λ is fractional and fits; rounding it to whole tiles need
        # not, and on a day-0 board it does not — the quota rounding overruns
        # the labour row on ten of twenty days (21.0 hours against 15.6).
        #
        # #13's `demote_to_feasible` is the repair written for exactly this
        # and it is NOT wired, because it was measured and it makes the
        # assignment worse: 10 violated rows in, 16 out, all 1000 steps spent.
        # It demotes the tile that loses the least value on the EARLIEST
        # violated day, and the plan it demotes to can use more on a later one
        # — so each step fixes one day and can break another. A repair that
        # walks one day at a time cannot answer a constraint that spans them.
        #
        # What catches the overrun instead is wsr: a day it cannot walk comes
        # back incomplete, `plan` cuts the hours and re-solves. That is slower
        # and it is honest, where shipping a repair that raises the violation
        # count would not be.
        chains = day_chains(choices, mixes, result.pool)
        committed = sum(
            float(np.asarray(mixes[c.class_key].plans[c.plan_index]
                             .row("labour"))[0])
            for c in choices if c is not None)
        fitted = fit(chains, hands=hands,
                     available=availability(obs, chains),
                     hours_committed=committed)
        if not fitted.complete and fitted.reason == "hours":
            # wsr's OWN number, not a count of retries: `floor` is the arithmetic
            # minimum the day's work needs, and the search already started there
            # (`hands=min(floor, hands)`, day.py:150). So ONE ask at the floor
            # answers "what is the least I could give it" - walking up one hand at
            # a time would buy the same answer for a search per step.
            #
            # "budget" is never re-asked: more hands do not buy more time.
            fitted = replace(fitted, short=max(0, int(fitted.floor) - int(hands)))
            # `Config.ask_rounds` is the cap, and it is the CONFIG's number, not
            # a module constant beside it: zero means no re-ask at all.
            if int(fitted.floor) > int(hands) and int(cfg.ask_rounds) > 0:
                again = fit(chains, hands=int(fitted.floor),
                            available=availability(obs, chains),
                            hours_committed=committed)
                if again.complete:
                    fitted = again
        # A complete answer needs no second ask for a leaner pool: the search
        # STARTS at the floor and grows (`min(floor, hands)`, day.py:150), so
        # `result.pool` already IS the least it carried the day with - offering 5
        # and using 1 reports 1, and the bill follows it.
        # The hands are hired again every morning (F039), so their wage is a
        # cost on every day of the horizon and not a one-off.
        # The bill follows the pool that actually carries the day (`fitted.pool`
        # is what `compile` hires, day.py:396), not the pool the master priced
        # with before the search had its say.
        bill = hire_bill(int(fitted.pool),
                         multiplier=terms.hand_cost_mult) * contractor.days
        candidate = DayPlan(result, choices, mixes, fitted, spent, applied,
                            solves=spent, hands=hands,
                            net=float(result.objective) - bill)
        if best is None or _better(candidate, best):
            best = candidate
        # `solves` is the only field that may move: rebuilding the winner field
        # by field dropped `hands` and `net` to their dataclass defaults (0 and
        # 0.0), and this object is what `plan` ranks offers by and what the
        # market side reads for the bill.
        best = replace(best, solves=spent)

        # Each correction round re-solves from the pool the last one left, so
        # a second solve is cheap even when the first was not.
        pool = list(result.pool)
        if fitted.complete or not chains:
            return candidate
        if fitted.reason == "budget":
            # wsr's own search stopped before placing everything — its report,
            # not a clock this layer set. Shrinking the supply on that would
            # price the search's own stopping rule into the farm's day.
            return candidate
        applied *= max(1.0 + tolerance, float(fitted.overhead))
        hours = hours_for(hands, days, cfg.hours_overhead) / applied

    return best if best is not None else candidate


def compile(day_plan: "DayPlan", obs, *, hands: int | None = None,
            rival_supply: dict | None = None,
            terms: "EngineTerms | None" = None, model=None,
            activity: int | None = None,
            forecast_obj=None) -> dict:
    """A `DayPlan` -> the `{"units": [...], "market": [...]}` the dispatcher slices.

    The unit ops come from the day layer's own compiler, against the doors the
    search priced the hands on — `Result.doors`, the only positions
    `compile_route` writes from (#73, #162). The market side is
    assembled from the SAME chains, so a seed the plan needs and a seed the
    queue buys cannot disagree.

    A day that did not fit compiles to nobody doing anything. That is a legal
    answer and the honest one: the alternative is dispatching a route the
    engine will refuse op by op without a word (F047).
    """
    from agent.planner import market as K
    from agent.wsr import beam as B
    from agent.wsr import tasks as T
    from agent.wsr.emit import compile_route, to_plan

    fitted = day_plan.day
    pool = int(fitted.pool if hands is None else hands)

    master_sells = {}
    solve_sells = getattr(getattr(day_plan, "master", None), "sells", None)
    if solve_sells is not None and getattr(solve_sells, "ndim", 0) == 2 and solve_sells.shape[1] > 0:
        from agent.world.model import PRODUCTS
        for gi, prod in enumerate(PRODUCTS):
            if gi < solve_sells.shape[0]:
                qty = int(round(float(solve_sells[gi, 0])))
                if qty > 0:
                    master_sells[prod] = qty

    if not fitted.complete or not fitted.chains:
        rows = K.build(obs, (), hands=0, terms=terms,
                       model=model, activity=activity,
                       forecast_obj=forecast_obj,
                       master_sells=master_sells).rows
        return {"units": [[["PASS"]] * TURNS_PER_DAY], "market": rows}

    from agent.world.rules import earliest_hire_times

    sold: dict = {}          # our own units per (good, step), from the last queue

    def sell_rank():
        """`rank(order, turn)`: the value of a sale there minus the rival's risk.

        The two surfaces belief already publishes (`hourly_value` and
        `rival_risk`), keyed the same way. Built only when there is a forecast to
        read the prices from: without one there is nothing to rank with, and the
        queue keeps belief's own order. `sold` is left empty for now, so the value
        is that turn's own quote — pricing our own planned volume into it is the
        next refinement, not a hidden assumption.
        """
        if forecast_obj is None:
            return None
        from agent.belief.depth import hourly_value, rival_risk

        # `c` is (cell, ops, entity): the GOOD is the entity, `c[2]` — `c[1]` is the
        # ops tuple, and passing that here raised KeyError the first time a day with
        # FEED/CARE chains reached this. Only market goods can be ranked.
        from agent.belief.market import PRODUCTS

        goods = tuple(sorted({str(c[2]) for c in fitted.chains
                              if len(c) > 2 and c[2] in PRODUCTS}))
        if not goods:
            return None
        day = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
        value = hourly_value(forecast_obj, goods, day, 1, sold=sold)
        risk = rival_risk(rival_supply or {}, goods, day, 1)

        def rank(order, turn):
            if not (order and str(order[0]) == "SELL" and len(order) > 1):
                return 0.0
            key = (str(order[1]), day * TURNS_PER_DAY + int(turn))
            return float(value.get(key, 0.0)) - float(risk.get(key, 0.0))

        return rank

    def queue(harvest_expected: int, hands: int, wsr_check: bool,
              arrivals: dict | None = None):
        built = K.build(obs, fitted.chains, hands=hands,
                        harvest_expected=harvest_expected, terms=terms,
                        model=model, activity=activity,
                        forecast_obj=forecast_obj, wsr_check=wsr_check,
                        arrivals=arrivals, rank=sell_rank(),
                        master_sells=master_sells)
        # Remember what this queue sells so the NEXT build ranks on our own volume
        # too: the ladder prices the lot we put in, and the correction round is
        # exactly the place that number exists.
        day0 = int(obs.get("day", 0)) if isinstance(obs, dict) else 0
        sold.clear()
        for turn, row in enumerate(built.rows):
            for order in row:
                if order and str(order[0]) == "SELL" and len(order) > 2:
                    key = (str(order[1]), day0 * TURNS_PER_DAY + turn)
                    sold[key] = int(sold.get(key, 0)) + int(order[2])
        return built

    def arrival_hours(ops) -> dict:
        """The hour each good is IN THE SHED today, from the route's own drops.

        `ops.arrivals` is `(hour, item, units)` per drop — the timetable the route
        already publishes and `compile` used to sum away. The EARLIEST hour wins:
        the first drop that carries the good is what makes it sellable.
        """
        out: dict = {}
        for hour, item, _units in ops.arrivals:
            good = str(item)
            out[good] = min(int(out.get(good, int(hour))), int(hour))
        return out

    def timetable(base: dict, check) -> dict:
        """`available`, with each bought good's hour taken from the queue.

        The queue settles a BUY and the good lands in the shed the turn after
        (F030). `availability`'s constant 1 is the same optimistic assumption on
        the goods side that the bound was on the labour side, and the queue is
        what actually knows.
        """
        out = dict(base)
        for good, hour in check.bought_hours:
            out[good] = max(int(out.get(good, 0)), int(hour))
        return out

    def priced(hire_times, warm=None):
        """The day, the route and the queue it implies, from one set of hours.

        `warm` is the route from a PREVIOUS call at the SAME pool. That is
        the only place a warm start means anything here: wsr's warm row
        carries the earlier route's own workers and `when`s, so it is legal
        only while the hand count is unchanged and the tasks move up or
        down. Change the pool and the row describes a day with different
        hands (owner, 2026-09-27)."""
        day = B.Day(chains=tuple(fitted.chains), available=available,
                    hire_times=hire_times)
        result = B.search(day, tasks, hands=pool, max_hands=pool, warm=warm)
        if warm is not None:
            # A warm start is an optimization, and it is only legal while the
            # day it was built for still holds: the warm row carries that route's
            # own `when`/`who`, so a schedule the compiler refuses is a route
            # wsr's `check_route` accepted and `compile_route` did not (measured:
            # `d4_build_pasture on worker 0 at turn 0 ... only -2 are free`).
            # Search again without it rather than lose the turn.
            from agent.wsr.emit import check_route as _check
            # `check_route` returns its COMPLAINTS: a non-empty list is a route
            # the compiler will not take (`fit` reads it the same way).
            if _check(day, tasks, result):
                result = B.search(day, tasks, hands=pool, max_hands=pool)
        ops = compile_route(day, tasks, result, horizon=TURNS_PER_DAY)
        harvest = sum(int(units) for _hour, _item, units in ops.arrivals)
        return day, result, ops, harvest

    # The hours come from the QUEUE, not from the engine's bound: the hourly
    # secretary lays the day's orders out with `wsr_check=False` (a check commits
    # no hires) and the day is priced on the hours those orders actually land in
    # (`DayMarket.hire_hours`: settlement plus one, F040). The bound assumed a
    # whole turn's order budget went to hires; this queue puts the sells first
    # (F032), so the bound was optimistic by construction.
    check = queue(0, pool, wsr_check=False)
    available = timetable(availability(obs, fitted.chains), check)
    tasks = T.build(fitted.chains, available=available)
    hours = tuple(check.hire_hours) if check.hire_hours else earliest_hire_times(pool)
    day, result, ops, harvest = priced(hours)

    # The plan is PRICED for `pool` hands and the search is held to exactly those: the market
    # hires what the day was costed with, and a route may leave some of them idle.
    market = queue(harvest, result.pool, wsr_check=True,
                   arrivals=arrival_hours(ops))
    if tuple(market.hire_hours) and tuple(market.hire_hours) != tuple(day.hire_times):
        # ONE correction round, and the reason the check's harvest was a stand-in:
        # the committed queue knows the real one, so if its hires land in other
        # hours the day is priced again on them. The invariant is that the day the
        # engine executes is the day the queue's own timetable priced.
        # SAME pool, same tasks, the queue's hours: the one re-solve a warm
        # start is for.
        again = priced(tuple(market.hire_hours), warm=result)
        if again[1].complete:
            day, result, ops, harvest = again
            market = queue(harvest, result.pool, wsr_check=True,
                           arrivals=arrival_hours(ops))
    return to_plan(ops, market=market.rows)
