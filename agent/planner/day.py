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

import time
from dataclasses import dataclass

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


def availability(obs, chains) -> dict:
    """wsr's `available` for this day, from the observation's own stocks."""
    from agent.planner.market import availability as _availability
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    return _availability(chains, dict(private.get("seeds", {}) or {}),
                         dict(private.get("shed", {}) or {}))


def fit(chains, *, hands: int, available: dict | None = None,
        budget_s: float | None = None, beam: int | None = None,
        hours_committed: float = 0.0, warm=None) -> DayFit:
    """Hand the day to wsr and report what it made of it.

    `hands` is the pool offered, all of them from hour 1 — a hand hired in turn
    0 first acts at hour 1 (F040). Offering them all and letting the search
    grow from the arithmetic floor is what keeps the cost predictable; asking
    wsr to find the smallest pool ITSELF is a halving search and costs several
    times a turn.
    """
    from agent.wsr import beam as B
    from agent.wsr import tasks as T
    from agent.wsr.emit import check_route

    if not chains:
        return DayFit((), 0, 0, 0, True, 0.0, hours_committed)

    available = available or {}
    tasks = T.build(chains, available=available)
    day = B.Day(chains=tuple(chains), available=available,
                hire_times=(1,) * max(1, hands))
    floor = max(0, B.lower_bound(day, tasks) - len(day.units))
    result = B.search(day, tasks, beam=beam,
                      hands=min(floor, max(1, hands)), max_hands=max(1, hands),
                      budget_s=budget_s, warm=warm)

    hours = float(max((turn for turn, _t, _w in result.route), default=0) + 1) \
        * max(1, result.pool)
    if result.complete and check_route(day, tasks, result, result.settled):
        # A route the compiler will not take is a day that did not fit, which
        # is an answer the caller already knows how to use (#73 is why this is
        # checked here rather than discovered inside `compile_route`).
        return DayFit(tuple(chains), len(result.route), tasks.n, result.pool,
                      False, hours, hours_committed, "unstable")
    reason = "" if result.complete else ("budget" if result.can_improve
                                         else "hours")
    return DayFit(tuple(chains), len(result.route), tasks.n, result.pool,
                  bool(result.complete), hours, hours_committed, reason)


def hours_for(hands: int, days: int, overhead: float = 0.35) -> np.ndarray:
    """The labour a day holds with `hands` hired, per day of the horizon.

    `24·(1 + hands) − hands`: the farmer's full day plus one per hand, less the
    hour each hand loses to being hired (F040 — a hand hired in turn 0 first
    acts at hour 1). The overhead is #12's M3 placeholder for the walking a
    route does beyond what a column charges; the columns now carry the walk to
    the tile, so what is left is the walking BETWEEN them.
    """
    gross = 24.0 * (1 + int(hands)) - int(hands)
    return np.full(days, gross * (1.0 - overhead))


def hire_bill(hands: int, hires_today: int = 0, multiplier: int = 1) -> int:
    """What `hands` hires cost today. Fibonacci, and it resets nightly (F039)."""
    from agent.world.rules import hire_cost
    return sum(hire_cost(int(hires_today) + i) * int(multiplier)
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


def plan(obs, contractor, supply, *, class_of_tile, iter_cap: int = 200,
         hands: int = 0, budget_s: float | None = None,
         rounds: int = 3, tolerance: float = 0.02,
         pool: list | None = None, deadline: float | None = None,
         max_hands: int | None = None, w_warm=None,
         forecast_obj=None, smoothing: float = 0.0) -> DayPlan:
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

    ceiling = int(hands if max_hands is None else max_hands)
    carried = list(pool or [])
    chosen: DayPlan | None = None
    # Largest pool FIRST. The first solve is the cold one and the budget may
    # cut the enumeration after it, so whichever offer runs first is the one a
    # short turn keeps — and more hands is where the value is (0 hands 34,197,
    # eight hands 166,449 for 54 coins). Walking down from the ceiling means a
    # cut enumeration keeps a good day instead of the emptiest one.
    for offer in range(max(0, ceiling), -1, -1):
        current = _solve_at(obs, contractor, supply, class_of_tile, offer,
                            iter_cap, budget_s, rounds, tolerance, carried,
                            deadline, w_warm, forecast_obj, smoothing)
        carried = list(current.master.pool)
        if chosen is None or current.net > chosen.net:
            chosen = current
        if deadline is not None and time.perf_counter() >= deadline:
            break
    return chosen


def _solve_at(obs, contractor, supply, class_of_tile, hands, iter_cap,
              budget_s, rounds, tolerance, pool, deadline, w_warm=None,
              forecast_obj=None, smoothing: float = 0.0) -> DayPlan:
    """One pool size: solve, assign, ask wsr, and price the hands."""
    from agent.planner import columns as C
    from agent.planner import master as M

    days = int(np.asarray(supply.hours).size)
    hours = hours_for(hands, days)
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
                               deadline=deadline, forecast_obj=forecast_obj,
                               smoothing=smoothing)
        mixes = M.to_mixes(result, contractor.days)
        choices = C.assign_by_quota(class_of_tile, mixes)
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
        fitted = fit(chains, hands=hands, budget_s=budget_s,
                     available=availability(obs, chains),
                     hours_committed=committed)
        # The hands are hired again every morning (F039), so their wage is a
        # cost on every day of the horizon and not a one-off.
        bill = hire_bill(hands) * contractor.days
        candidate = DayPlan(result, choices, mixes, fitted, spent, applied,
                            solves=spent, hands=hands,
                            net=float(result.objective) - bill)
        if best is None or _better(candidate, best):
            best = candidate
        best = DayPlan(best.master, best.choices, best.mixes, best.day,
                       best.rounds, best.overhead, solves=spent)

        # Each correction round re-solves from the pool the last one left, so
        # a second solve is cheap even when the first was not.
        pool = list(result.pool)
        if fitted.complete or not chains:
            return candidate
        if deadline is not None and time.perf_counter() >= deadline:
            return candidate
        if fitted.reason == "budget":
            # wsr ran out of time, not out of hours. Shrinking the supply on
            # that would price the search's clock into the farm's day.
            return candidate
        applied *= max(1.0 + tolerance, float(fitted.overhead))
        hours = hours_for(hands, days) / applied

    return best if best is not None else candidate


def compile(day_plan: "DayPlan", obs, *, hands: int | None = None,
            config=None, model=None, activity: int | None = None,
            forecast_obj=None) -> dict:
    """A `DayPlan` -> the `{"units": [...], "market": [...]}` the dispatcher slices.

    The unit ops come from the day layer's own compiler, against the doors the
    search priced the hands on — `Result.doors`, which `compile_route` reads
    when it is not handed a `settled` of its own (#73). The market side is
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
    if not fitted.complete or not fitted.chains:
        rows = K.build(obs, (), hands=0, config=config,
                       model=model, activity=activity,
                       forecast_obj=forecast_obj).rows
        return {"units": [[["PASS"]] * TURNS_PER_DAY], "market": rows}

    available = availability(obs, fitted.chains)
    tasks = T.build(fitted.chains, available=available)
    day = B.Day(chains=tuple(fitted.chains), available=available,
                hire_times=(1,) * max(1, pool))
    result = B.search(day, tasks, hands=min(pool, max(1, pool)),
                      max_hands=max(1, pool))
    # The plan is PRICED for `pool` hands; the route may need fewer, but the
    # market must hire what the day was costed with or the hours row was a
    # fiction. wsr reports what it used, and the smaller of the two is what
    # gets paid for.
    # The day is written from the doors the search priced the hands on (`Result.doors`, F040). An
    # explicit `settled=` would switch those off and place the hands from where the farmer STARTS,
    # which is a different day whenever the farmer walks off its door in turn 0 (#73).
    ops = compile_route(day, tasks, result, horizon=TURNS_PER_DAY)
    harvest = sum(int(units) for _hour, _item, units in ops.arrivals)
    market = K.build(obs, fitted.chains, hands=min(pool, result.pool) if pool else result.pool,
                     harvest_expected=harvest, config=config,
                     model=model, activity=activity, forecast_obj=forecast_obj)
    return to_plan(ops, market=market.rows)
