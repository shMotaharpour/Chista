"""The manager: the one feedback loop, and the only thing in the agent that decides.

Everything else answers a question. `belief` says what a sale will be worth,
`tile_dp` says what one tile should do at a given price, `wsr` says whether a
day's chains can actually be walked by n hands. None of them chooses. The
manager chooses, and it chooses by moving ONE number until the answers agree:

    the scarcity price of an hour.

Labour free means the DP asks for every op it can justify and wsr cannot walk
the result. Labour dear means the day fits and half the farm idles. Between
those is the smallest wage whose day fits, and that wage IS the plan: the DP
re-planned every tile at it, so nothing was dropped to make the day fit — the
tiles chose cheaper chains, which is a different thing and a better one.

## Why this is resumable

Finding that wage is a bisection, and a bisection is a sequence of independent
probes. One probe is one `step()`. `actTimeout` is per turn and hours 1..23
replay a plan already made, so a day holds 24 seconds of thinking and the old
shape used one of them. `observe()` at hour 0, `step()` every turn, `best()`
whenever the dispatcher needs a plan — and the plan only ever improves.

Each probe hands wsr the previous probe's route as `warm`, so the beam repairs
instead of restarting: measured on this layer at 138 tasks in 26.7 ms against
403 ms cold.

## What it does NOT do

It does not catch its own exceptions. A manager that swallows an error plays a
worse policy and looks exactly like one that worked — which is how a greedy
rung answered every turn for four days while nothing was wired, and scored
2,840 against a PASS opponent's 3,000. The runtime catches, records that the
manager failed, and passes. Failure has to be visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from agent.config import Config
from agent.manager.prices import cost_vector, hours_demanded, value_vector
from agent.obs import decode_world
from agent.tile_dp.chains import chain_ops
from agent.tile_dp.contractor import TileContractor
from agent.world.rules import ANIMAL_RULES, CROP_RULES, TURNS_PER_DAY, hire_cost
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import check_route, compile_route, to_plan


@dataclass(frozen=True)
class Probe:
    """One wage, asked and answered.

    `fits` is the whole verdict; `reason` says which wall it hit, because the
    two walls need different reading. A day that wsr could not walk is short of
    HOURS. A day the purse could not buy is short of COINS. Both are answered
    by the same lever — a dearer hour buys fewer ops, and fewer ops need fewer
    seeds — but a log that cannot tell them apart cannot say which one bound.
    """

    wage: float
    fits: bool
    reason: str                  # "": "hours", "cash", "budget", "unstable"
    placed: int = 0              # tasks wsr walked
    tasks: int = 0               # tasks the day asked for
    hands: int = 0
    bill: int = 0                # coins the day's inputs and hires cost
    hours: int = 0               # worker-hours the DP asked for
    elapsed_ms: float = 0.0


class Manager:
    """The decision. `observe` once a day, `step` every turn, `best` any time."""

    def __init__(self, graph, config: Config | None = None) -> None:
        self.cfg = config or Config.load()
        self.graph = graph
        self.graph_keys = frozenset(graph.key_index)
        # One contractor for the process: the casts and the slice check are
        # per-instance, and paying them 720 times is 720 times too many.
        self.contractor = TileContractor(graph, days=self.cfg.horizon_days)
        self.probes: list[Probe] = []
        self._reset()

    # -- the day ---------------------------------------------------------
    def _reset(self) -> None:
        self.view = None
        self.raw_obs = None
        self.raw_config = None
        self.owned: list[int] = []
        self.forward = None
        self.hand_mult = 1
        self.lo = self.cfg.wage_floor        # highest wage known NOT to fit
        self.hi: float | None = None         # lowest wage known TO fit
        self.plan: dict | None = None        # the plan at `hi`; idle until one fits
        self.fitted = False                  # whether any wage has fit today
        self.warm = None                     # wsr's last route, for the next probe
        self.probes = []

    def observe(self, obs, config=None) -> None:
        """Start a new day from the board as it stands at hour 0.

        The decode asserts hour 0 for a reason that has already cost us once:
        `decode_tile` reads a DAY-START value, and a mid-day value fed to the
        DP is the pre-v15 rebuild bug (184 of 394 nodes wrong). So the manager
        observes at the day's start and thinks for the rest of it.
        """
        self.raw_obs, self.raw_config = obs, config
        self.view = decode_world(obs, config, at_day_start=True,
                                 graph_keys=self.graph_keys)
        self.hand_mult = int((config or {}).get("farmHandCostMult", 1) or 1)
        self.owned = self._owned_states(self.view)
        self.forward = self._forward_prices(obs)
        self.lo = self.cfg.wage_floor
        self.hi = None
        self.warm = None
        self.probes = []
        # The floor answer, installed before the first probe: a day in which
        # nobody does anything. It is legal, it is what the farm should do when
        # it cannot afford to act, and above all it means `best()` is never
        # None once a day has been observed. The previous shape returned None
        # until a wage fit, and on a farm that had run itself out of money no
        # wage ever did - so the agent passed for whole days and the only sign
        # was the score.
        self.plan = to_plan(_empty_ops(), market=[])
        self.fitted = False

    def _owned_states(self, view) -> list[int]:
        """The graph state id of every tile we own and can plan.

        A LOCKED tile is not ours (F042) and spends a unit's hours for
        nothing, so it is not in the list. `decode_world` has already mapped
        any key the shipped graph does not carry onto its nearest modelled
        neighbour, so a key that is still missing here is a bug in the graph,
        not a tile to skip quietly.
        """
        keys = np.asarray(view.me.keys)
        out: list[int] = []
        for key in keys.reshape(-1):
            k = int(key)
            if k < 0:                       # LOCKED sentinel
                continue
            state = self.graph.key_index.get(k)
            if state is None:
                raise KeyError(
                    f"tile key {k} is not in the shipped graph after remapping: "
                    f"{view.unknown_keys} unmodelled keys were seen this decode")
            out.append(int(state))
        return out

    def _forward_prices(self, obs) -> np.ndarray:
        """Belief's day table, or today's quote repeated if belief cannot walk.

        Belief estimates; it does not decide, and it is allowed to be absent.
        What is NOT allowed is for its absence to be invisible, so the fallback
        is the flattest possible one — today's quote, held — and the manager
        records that it used it.
        """
        from agent.belief.market import forecast
        fc = forecast(obs, days=self.cfg.horizon_days)
        return np.asarray(fc.prices, dtype=np.float64)

    # -- one probe -------------------------------------------------------
    @property
    def settled(self) -> bool:
        """Whether another probe would buy less than the turn it costs."""
        if self.view is None:
            return True
        if self.hi is None:
            return self.lo >= self.cfg.wage_ceiling
        return (self.hi - self.lo) <= self.cfg.wage_tolerance

    def _next_wage(self) -> float:
        """The wage to ask about next.

        While no wage has been found that fits, the probe climbs: there is no
        bracket to halve, and the answer is somewhere above. Once one fits, the
        bracket exists and the midpoint is the only probe worth paying for.
        """
        if self.hi is not None:
            return (self.lo + self.hi) / 2.0
        if not self.probes:
            return self.cfg.wage_floor       # labour free: the most work possible
        climb = max(self.cfg.wage_first, self.lo * self.cfg.wage_growth)
        return min(climb, self.cfg.wage_ceiling)

    def step(self, budget_ms: float | None = None) -> Probe | None:
        """One probe. Returns what it learned, or None when nothing is left.

        The plan can only improve: a probe that fits at a lower wage replaces
        the stored plan, and a probe that does not fit never touches it.
        """
        if self.view is None or self.settled:
            return None
        budget_s = (self.cfg.search_budget_s if budget_ms is None
                    else max(0.0, (budget_ms - self.cfg.reserve_ms) / 1000.0))
        wage = self._next_wage()
        probe = self._probe(wage, budget_s)
        self.probes.append(probe)
        if probe.fits:
            self.hi = wage
        else:
            self.lo = max(self.lo, wage)
        return probe

    def _probe(self, wage: float, budget_s: float) -> Probe:
        """Price the board at this wage, and find out whether the day is real."""
        started = time.perf_counter()

        def done(fits: bool, reason: str, **kw) -> Probe:
            return Probe(wage=wage, fits=fits, reason=reason,
                         elapsed_ms=(time.perf_counter() - started) * 1000.0, **kw)

        p = value_vector(self.forward, self.cfg.horizon_days)
        w = cost_vector(wage, self.view.market_prices, self.cfg.horizon_days)
        board = self.contractor.price(p, w, self.owned)
        hours = hours_demanded(board.per_day_cost)

        chains = self._chains(board)
        if not chains:
            # Nothing to do is a plan, and a legal one - and it is already the
            # floor plan installed by `observe`. It fits at any wage, so the
            # bisection is over: there is no cheaper answer than idle.
            self.fitted = True
            return done(True, "", hours=hours)

        bill, orders, hires = self._shopping(board, chains)
        purse = float(self.view.me.money) - self.cfg.cash_reserve
        if bill > purse:
            return done(False, "cash", tasks=0, bill=bill, hours=hours)

        available = self._available(chains)
        tasks = T.build(chains, available=available)
        day = B.Day(chains=tuple(chains), available=available,
                    hire_times=(1,) * self.cfg.max_hands)
        floor = max(0, B.lower_bound(day, tasks) - len(day.units))
        result = B.search(day, tasks, beam=self.cfg.beam,
                          hands=min(floor, self.cfg.max_hands),
                          max_hands=self.cfg.max_hands,
                          budget_s=budget_s, warm=self.warm)
        self.warm = result

        if not result.complete:
            reason = "budget" if result.can_improve else "hours"
            return done(False, reason, placed=len(result.route), tasks=tasks.n,
                        hands=result.pool, bill=bill, hours=hours)

        # The route has to be compiled against the SAME hand positions it was
        # searched with, and `search` does not hand them back: the fixed point
        # settles them internally and returns only the route. So they are
        # re-derived here, and then the route is checked against them before it
        # is compiled - because when the search was cut by its deadline the two
        # can disagree, and `compile_route` answers that disagreement by
        # raising in the agent's hot path. Measured: one season in four, 24
        # raises in a day ("d17_water on worker 1 at turn 3: it needs 3 turns
        # from 1 and only 2 are free"). A route we cannot compile is a day that
        # did not fit, which is an answer the bisection already knows how to use.
        settled = B._settled_after_first_turn(day, tasks, result)
        problems = check_route(day, tasks, result, settled)
        if problems:
            return done(False, "unstable", placed=len(result.route),
                        tasks=tasks.n, hands=result.pool, bill=bill, hours=hours)
        ops = compile_route(day, tasks, result, horizon=TURNS_PER_DAY,
                            settled=settled)
        market = self._market_rows(orders, hires[:result.pool])
        self.plan = to_plan(ops, market=market)
        self.fitted = True
        return done(True, "", placed=len(result.route), tasks=tasks.n,
                    hands=result.pool, bill=bill, hours=hours)

    # -- the day's pieces -------------------------------------------------
    def _chains(self, board) -> list[tuple]:
        """Day 0 of every priced tile, as wsr's `(cell, ops, entity)`.

        A tile whose day-0 chain is empty is not in the day. That is the DP
        declining to act on it at this wage, not the manager dropping it — and
        the difference matters, because a dropped tile comes back unchanged
        while a declined one was re-planned and chose to wait.
        """
        h, w = np.asarray(self.view.me.keys).shape
        cells = [(x, y) for y in range(h) for x in range(w)
                 if int(self.view.me.keys[y][x]) >= 0]
        out = []
        for i, plan in enumerate(board.plans):
            if not plan or i >= len(cells):
                continue
            day0 = plan[0]
            ops = chain_ops(int(day0[2]))
            if not ops:
                continue
            entity = _entity_name(board, i)
            out.append((cells[i], tuple(ops), entity))
        return out

    def _available(self, chains) -> dict[str, int]:
        """The hour each good the day needs is in the shed.

        What is already on the shelf is there at hour 0. What the day buys is
        ordered in turn 0 and lands at hour 1, because the engine settles units
        before market inside a turn (F030) — so a task that consumes it cannot
        run in turn 0 however early the order is queued.
        """
        shed = dict(self.view.private.shed)
        seeds = dict(self.view.private.seeds)
        out: dict[str, int] = {}
        for _cell, _ops, entity in chains:
            if entity is None:
                continue
            held = seeds.get(entity, 0) or shed.get(entity, 0)
            out[entity] = 0 if held > 0 else 1
        return out

    def _shopping(self, board, chains) -> tuple[int, list, list]:
        """What today's chains have to buy, and what the hands cost.

        The bill is checked against the purse BEFORE the day is searched, which
        is the arm `planner/repair.py` never had: it built the day first and
        then deleted whatever the money would not cover, so the plan that ran
        was never the plan that was priced.
        """
        need: dict[str, int] = {}
        for _cell, ops, entity in chains:
            if entity is None:
                continue
            if "PLANT" in ops and entity in CROP_RULES:
                need[entity] = need.get(entity, 0) + 1
            elif "PLACE" in ops and entity in ANIMAL_RULES:
                need[entity] = need.get(entity, 0) + 1

        seeds = dict(self.view.private.seeds)
        shed = dict(self.view.private.shed)
        orders, bill = [], 0
        for good, units in sorted(need.items()):
            held = seeds.get(good, 0) if good in CROP_RULES else shed.get(good, 0)
            short = max(0, units - held)
            if not short:
                continue
            if good in CROP_RULES:
                orders.append(["BUY_SEED", good, short])
                bill += short * int(CROP_RULES[good]["seed"])
            else:
                orders.append(["BUY_ANIMAL", good, short])
                bill += short * int(ANIMAL_RULES[good]["cost"])

        already = int(self.view.me.hires_today)
        hires = [["HIRE"] for _ in range(self.cfg.max_hands)]
        ladder = [hire_cost(already + i) * self.hand_mult
                  for i in range(self.cfg.max_hands)]
        # The bill carries only the hands the search will actually use, and the
        # search has not run yet — so it carries the floor: one hand. A probe
        # that cannot afford one hand cannot afford any day that needs them.
        bill += ladder[0] if self.cfg.max_hands else 0
        return bill, orders, hires

    def _market_rows(self, orders, hires) -> list[list]:
        """The day's buys and hires, one row per turn.

        **The sell side is deliberately not here, and that is measured rather
        than assumed.** `belief.shed.market_queue` exists and takes exactly the
        two numbers this manager can now supply — `harvest_expected` from the
        compiled route's own DROPs (which is the #14 wiring its docstring has
        been waiting for) and `cash_needed` from the bill below. Wiring it made
        the agent worse, over a full 30-day season against a PASS opponent:

            buys and hires only            2,950
            + belief's queue, cash_needed  1,706
            + belief's queue, cash_needed=0  907

        against PASS's 3,000. So the seam is left open and the number is on the
        record, rather than shipping a sell side that loses a thousand coins and
        calling the loop closed.

        The cap is per TURN (F031: ten), so a day with more opening orders than
        that spills into the next turn instead of being silently refused — the
        engine does not say no, it just does nothing (F047).
        """
        rows: list[list] = [[] for _ in range(TURNS_PER_DAY)]
        queue = [list(o) for o in hires] + [list(o) for o in orders]
        turn, cap = 0, self.cfg.max_orders_per_turn
        while queue and turn < TURNS_PER_DAY:
            rows[turn] = queue[:cap]
            queue = queue[cap:]
            turn += 1
        if queue:
            raise ValueError(
                f"{len(queue)} opening orders had no slot in the day: the cap is "
                f"{cap} per turn (F031) and every turn was full")
        return rows

    # -- the answer -------------------------------------------------------
    def best(self) -> dict | None:
        """The plan of the cheapest wage that fit, or the idle day if none did.

        None only before the first `observe`. After it there is always a legal
        answer, and `fitted` says whether that answer is a searched day or the
        farm standing still because nothing it could afford would fit.
        """
        return self.plan


def _entity_name(board, tile: int) -> str | None:
    """What day 0's chosen edge constructs on this tile, if anything."""
    from agent.tile_dp.chains import entity_of_code
    if board.per_day_entity is None:
        return None
    return entity_of_code(int(board.per_day_entity[tile, 0]))


def _empty_ops():
    """A day in which nobody does anything: the farmer passes, 24 turns long."""
    from agent.wsr.emit import DayOps
    return DayOps(units=[[("PASS",)] * TURNS_PER_DAY], arrivals=())
