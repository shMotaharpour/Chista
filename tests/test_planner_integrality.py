"""Integrality tests (issue #13): rounding, the row check, repair, land.

Run:  .venv/bin/python -m tests.test_planner_integrality

Contracts under test:
- Rounding is a function of (board, λ) — same input, same assignment, twice —
  and a tile the farm does not own never receives a plan (F042).
- The assignment is checked against the master's coupling rows, and the demotion
  loop clears the violations it can; a tight row is shown to fail the check
  first, so the check is not vacuous.
- The repaired plan dispatches into shape-valid actions for all 30 days and
  `world.actions.validate_action` accepts each.
- F047's drops are counted, per rule: the F031 cap, a short purse, a FERTILIZE
  with no fertilizer in the bag (F004), a SELL of what the shed does not hold
  (F043), an op on a LOCKED tile (F042), and F032's queue order.
- The land loop is the engine's prefix arithmetic (R002), enumerated with the
  master injected, with the cadence cap of at most three evaluations per episode.

The two acceptance numbers that need #12 (the integrality gap and the land
enumeration's 400 ms) are NOT tested here — they are named TODOs on the modules,
because a number measured against a stand-in master would be a wrong number
dressed as a real one (R005).
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.dispatch import dispatch_plan, MAX_MARKET_ORDERS
from agent.replan import dual_stand_in, load_contractor
from agent.world.model import compile_chain
from agent.obs import decode_world
from agent.tile_dp.chains import chain_ops, entity_of_code
from planner.columns import (DAYS, Choice, ClassMix, Plan, assign_tiles,
                             counts, demote_to_feasible, plan_from_board,
                             rounded_value, row_use, violations)
from planner.land import LandPlanner, best_land, candidates, prefix_cost
from planner.repair import order_cost, repair_day
from agent.world.action_rules import validate_action
from offline_lab.fast_sim import FastSim

LOCKED = -1
_CLASS = 7


def _plan(value: float, labour: float, chains=(1,) * DAYS) -> Plan:
    """A synthetic column: `labour` hours a day, everything else zero."""
    zeros = (0.0,) * DAYS
    return Plan(chains=tuple(chains), value=value,
                rows={"labour": (labour,) * DAYS, "cash_out": zeros,
                      "wheat_net": zeros, "fert_net": zeros,
                      "stored": zeros})


def _mix(lam=(0.6, 0.4)) -> ClassMix:
    return ClassMix(class_key=_CLASS, count=10,
                    plans=(_plan(10.0, 8.0), _plan(6.0, 2.0)),
                    lam=tuple(lam))


def _obs(day: int = 0, hour: int = 0, money: int = 100,
         shed=None, seeds=None, tile=None, inventory=None,
         quadrants=None) -> dict:
    """A real engine observation, with the fields a repair looks at."""
    sim = FastSim({"episodeSteps": 24 * 3, "seed": 1})
    obs = dict(sim.observations()[0])
    obs["day"], obs["hour"] = day, hour
    obs["farms"] = [dict(obs["farms"][0])]
    tiles = [[None] * 10 for _ in range(10)]
    if tile is not None:
        tiles[4][4] = tile
    obs["farms"][0].update({"tiles": tiles, "money": money, "hands": [],
                            "hires_today": 0})
    if quadrants is not None:
        obs["farms"][0]["unlocked_quadrants"] = list(quadrants)
    obs["private"] = {"shed": shed or {}, "seeds": seeds or {},
                      "inventories": [inventory or {}]}
    return obs


# --------------------------------------------------------------- the rounding

def test_rounding_is_a_function() -> None:
    """Same λ, same board, same assignment — twice, byte-identical."""
    mixes = {_CLASS: _mix()}
    board = [_CLASS, LOCKED, _CLASS, 999]
    first = assign_tiles(board, mixes)
    second = assign_tiles(board, mixes)
    assert first == second
    assert [c.plan_index if c else None for c in first] == [0, None, 0, None]
    assert counts(first) == {(_CLASS, 0): 2}


def test_tie_falls_to_the_lowest_plan_index() -> None:
    """Determinism, made explicit: an even λ is not a coin toss."""
    mixes = {_CLASS: _mix(lam=(0.5, 0.5))}
    choice = assign_tiles([_CLASS], mixes)[0]
    assert choice is not None and choice.plan_index == 0


def test_locked_tile_never_receives_a_plan() -> None:
    """F042: working a locked quadrant is a silent no-op — never plan one."""
    mixes = {_CLASS: _mix()}
    for board in ([LOCKED] * 4, [LOCKED, _CLASS, LOCKED, _CLASS]):
        choices = assign_tiles(board, mixes)
        assert all(c is None for c, key in zip(choices, board) if key == LOCKED)
        assert all(c is not None for c, key in zip(choices, board)
                   if key == _CLASS)


def test_unmodelled_class_is_dropped_not_guessed() -> None:
    """A class the master never priced gets no plan, and is silently absent
    from the costs — the caller counts the misses."""
    choices = assign_tiles([999], {_CLASS: _mix()})
    assert choices == [None]


# ----------------------------------------------------------- the coupling rows

def test_coupling_rows_are_checked_and_demoted() -> None:
    """The acceptance's row check, on 20 boards, with the check shown to bite.

    10 tiles on an 8-hour plan need 80 hours against a 24-hour day: the
    assignment violates the row, the demotion loop moves the cheapest tiles to
    the 2-hour plan until it fits, and the value it costs is recorded.
    """
    capacities = {"labour": [24.0] * DAYS}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        lam = (0.5 + 0.5 * float(rng.random()), 0.0)
        lam = (lam[0], 1.0 - lam[0])
        mixes = {_CLASS: _mix(lam=lam)}
        board = [_CLASS] * 10
        choices = assign_tiles(board, mixes)

        bad = violations(choices, mixes, capacities)
        assert bad, f"seed {seed}: the 8-hour plan should not fit a 24-hour day"
        assert row_use(choices, mixes, "labour")[0] == 80.0

        fixed, demotions, remaining = demote_to_feasible(choices, mixes,
                                                         capacities)
        assert remaining == [], f"seed {seed}: {remaining}"
        assert demotions, f"seed {seed}: demotions were needed but not made"
        assert rounded_value(fixed, mixes) < rounded_value(choices, mixes)
        assert all(c is not None and c.plan_index == 1 for c in fixed[:3])


def test_row_use_sums_per_tile_not_per_class() -> None:
    """The rows are per TILE: two tiles on the same plan use it twice."""
    mixes = {_CLASS: _mix()}
    choices = [Choice(_CLASS, 0), Choice(_CLASS, 0)]
    assert row_use(choices, mixes, "labour")[0] == 16.0


def test_rows_come_from_the_pricing_oracle() -> None:
    """`plan_from_board` reads #11's output: chains, labour, and the nin rows."""
    graph, contractor = load_contractor_graph()
    obs = _obs()
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    state = graph.key_index[int(view.me.keys[4][4])]
    p, w = dual_stand_in(obs)
    board = contractor.price(p, w, [state])
    plan = plan_from_board(board, 0, wages=w, prices=p)
    assert len(plan.chains) == DAYS
    assert list(plan.chains) == [chain for _d, _s, chain in board.plans[0]]
    assert plan.row("labour")[0] == float(board.per_day_cost[0][0][0])
    assert len(plan.row("stored")) == DAYS


def load_contractor_graph():
    contractor = load_contractor()
    return contractor.graph, contractor


# -------------------------------------------------------------------- repair

def test_repaired_plan_dispatches_and_validates() -> None:
    """Every repaired plan is a legal action, hour by hour, for 30 days."""
    graph, contractor = load_contractor_graph()
    obs = _obs()
    view = decode_world(obs, at_day_start=True,
                        graph_keys=frozenset(graph.key_index))
    p, w = dual_stand_in(obs)
    state = graph.key_index[int(view.me.keys[4][4])]
    board = contractor.price(p, w, [state])
    _day, _state, chain_id = board.plans[0][0]
    entity = entity_of_code(int(board.per_day_entity[0, 0]))
    turns = compile_chain(chain_ops(chain_id), entity)
    plan = {"units": [turns], "market": []}
    result = repair_day(plan, obs)
    for day in range(30):
        for hour in range(24):
            turn_obs = dict(obs)
            turn_obs["day"], turn_obs["hour"] = day, hour
            action = dispatch_plan(result.plan, turn_obs)
            validate_action(0, action)
            assert set(action) == {"farmer", "hands", "market"}


def test_market_cap_and_short_purse_are_counted() -> None:
    """F031: the 11th order is dropped — and said so.

    Note for whoever writes the next cap test: 11 × SELL against an EMPTY shed
    proves nothing about the cap — the shed check runs first and consumes every
    order (11 drops, all F043). Stock the shed, or use purchases, as below; the
    reviewer made exactly this point on PR #35 (N1).
    """
    obs = _obs(money=1000)
    cheap = [["BUY_SEED", "WHEAT", 1] for _ in range(15)]
    result = repair_day({"units": [], "market": cheap}, obs)
    assert len(result.plan["market"]) == MAX_MARKET_ORDERS == 10
    assert result.by_reason() == {"F031: an 11th order is dropped silently": 5}

    # the engine refuses what the purse cannot pay, in silence (F031/F047)
    poor = _obs(money=50)
    result = repair_day({"units": [], "market": [["BUY_SEED", "MELON", 1]]},
                        poor)
    assert result.plan["market"] == []
    assert result.dropped == 1 and "purse" in result.drops[0].reason

    # an order the purse CAN pay survives, and the purse is spent
    rich = _obs(money=300)
    result = repair_day({"units": [],
                         "market": [["BUY_SEED", "MELON", 1],
                                    ["BUY_SEED", "MELON", 1],
                                    ["BUY_SEED", "MELON", 1]]}, rich)
    assert len(result.plan["market"]) == 3      # 3x80 = 240 <= 300


def test_market_queue_follows_f032() -> None:
    """Land, then sells, then hires, then purchases (F032)."""
    obs = _obs(money=3000, shed={"WHEAT": 10})
    market = [["BUY_SEED", "WHEAT", 1], ["HIRE"], ["SELL", "WHEAT", 5],
              ["BUY_LAND"]]
    result = repair_day({"units": [], "market": market}, obs)
    assert [order[0] for order in result.plan["market"]] == \
        ["BUY_LAND", "SELL", "HIRE", "BUY_SEED"]


def test_fertilize_without_fertilizer_is_dropped() -> None:
    """F004: the acting unit must carry the fertilizer; nobody else's works."""
    obs = _obs()
    plan = {"units": [[["FERTILIZE"]]], "market": []}
    result = repair_day(plan, obs)
    assert result.plan["units"] == [[["PASS"]]]
    assert "F004/F047" in result.drops[0].reason

    carried = _obs(inventory={"FERTILIZER": 1})
    result = repair_day(plan, carried)
    assert result.plan["units"] == [[["FERTILIZE"]]]
    assert result.dropped == 0


def test_sell_without_stock_is_dropped() -> None:
    """F043/F047: the shed is the only source of a SELL."""
    empty = repair_day({"units": [], "market": [["SELL", "WHEAT", 5]]}, _obs())
    assert empty.plan["market"] == [] and empty.dropped == 1
    stocked = repair_day({"units": [], "market": [["SELL", "WHEAT", 5]]},
                         _obs(shed={"WHEAT": 5}))
    assert stocked.plan["market"] == [["SELL", "WHEAT", 5]]
    assert stocked.dropped == 0


def test_ops_on_a_locked_tile_are_dropped() -> None:
    """F042: a LOCKED tile spends the hours for nothing."""
    obs = _obs(tile="LOCKED")
    result = repair_day({"units": [[["WATER"], ["HARVEST"]]], "market": []},
                        obs)
    assert result.plan["units"] == [[["PASS"], ["PASS"]]]
    assert result.dropped == 2
    assert all("F042" in drop.reason for drop in result.drops)


def test_repair_summary_counts_by_rule() -> None:
    """A plan that quietly loses its ops is the failure this module catches."""
    obs = _obs(tile="LOCKED", money=1)
    plan = {"units": [[["WATER"]]],
            "market": [["BUY_SEED", "MELON", 1]]}
    result = repair_day(plan, obs)
    assert result.dropped == 2
    assert "F042" in result.summary() and "F031/F047" in result.summary()


def test_land_purchases_escalate_inside_a_day() -> None:
    """B1 of the PR #35 review: three BUY_LAND are not three first prices.

    The real prefix is 1000 + 2000 + 4000 = 7000; before the counter, all three
    were priced 1000 and a 7000-coin plan walked through a 3000-coin purse.
    """
    obs = _obs(money=3000, quadrants=("NW",))
    result = repair_day({"units": [], "market": [["BUY_LAND"]] * 3}, obs)
    # 1000 + 2000 = 3000 fits exactly; the third purchase needs 4000 and the
    # purse is empty. Before the counter all three were priced 1000 and kept.
    assert len(result.plan["market"]) == 2, result.summary()
    assert result.dropped == 1
    assert "4000 coins against a 0-coin purse" in result.drops[0].reason

    rich = repair_day({"units": [], "market": [["BUY_LAND"]] * 3},
                      _obs(money=7000, quadrants=("NW",)))
    assert len(rich.plan["market"]) == 3 and rich.dropped == 0

    priced = [order_cost(["BUY_LAND"], obs, 0, bought) for bought in (0, 1, 2)]
    assert priced == [float(K.LAND_PRICES[0]), float(K.LAND_PRICES[1]),
                      float(K.LAND_PRICES[2])]


def test_buy_land_on_a_complete_prefix_is_dropped() -> None:
    """F042: with every quadrant owned the buy refuses in silence."""
    obs = _obs(money=99999, quadrants=["NW"] + list(K.LAND_ORDER))
    result = repair_day({"units": [], "market": [["BUY_LAND"]]}, obs)
    assert result.plan["market"] == []
    assert result.dropped == 1 and "F042" in result.drops[0].reason


def test_order_cost_uses_engine_tables() -> None:
    """R002: every price the repair compares comes from the engine."""
    obs = _obs(money=1000)
    assert order_cost(["BUY_SEED", "TOMATO", 2], obs, 0) == \
        2 * float(K.CROPS["TOMATO"]["seed"])
    assert order_cost(["BUY_ANIMAL", "COW", 1], obs, 0) == \
        float(K.ANIMALS["COW"]["cost"])
    assert order_cost(["HIRE"], obs, 0) == float(K._hire_cost(0))
    assert order_cost(["HIRE"], obs, 3) == float(K._hire_cost(3))
    assert order_cost(["BUY_LAND"], obs, 0) == float(K.LAND_PRICES[0])
    assert order_cost(["BUY_LAND"], obs, 0, 1) == float(K.LAND_PRICES[1])
    assert order_cost(["SELL", "WHEAT", 5], obs, 0) is None


# ---------------------------------------------------------------------- land

def test_land_prefix_is_the_engine_table() -> None:
    """F042: prefix-locked, no gaps, prices from the engine."""
    assert prefix_cost(1) == 0
    assert prefix_cost(2) == int(K.LAND_PRICES[0])
    assert prefix_cost(4) == int(sum(K.LAND_PRICES))
    assert len(candidates(days=30, grid=2)) == 1 + 3 * 15


def test_land_enumeration_picks_the_best() -> None:
    """Enumerate with the master injected; keep the best and count the solves."""
    def solve(candidate):
        return -float(candidate.cost) - (candidate.day or 0)

    result = best_land(solve, days=30, grid=2)
    assert result.best.cost == 0 and result.best.day is None
    assert result.solves == result.considered == 46

    try:
        best_land(solve, days=30, grid=2, max_solves=10)
    except ValueError as exc:
        assert "budget" in str(exc), exc
    else:
        raise AssertionError("an over-budget enumeration ran anyway")


def test_land_cadence_is_capped() -> None:
    """Day 0 always evaluates; later days only when affordable, at most 3x."""
    calls = []

    def solve(candidate):
        calls.append(candidate.describe())
        return 1.0

    planner = LandPlanner(solve, days=30, grid=2)
    assert planner.consider(0, cash=0, quadrants=1) is not None
    assert planner.consider(1, cash=0, quadrants=1) is None       # unaffordable
    assert planner.consider(1, cash=2000, quadrants=1) is not None
    assert planner.consider(2, cash=9999, quadrants=1) is not None
    assert planner.evaluations == 3
    assert planner.consider(3, cash=9999, quadrants=1) is None    # cap spent
    assert planner.consider(3, cash=9999, quadrants=4) is None    # prefix done
    # day 0 considers the whole grid; a later re-check drops the days behind it
    whole = len(candidates(days=30, grid=2))
    later = len([c for c in candidates(days=30, grid=2)
                 if c.day is None or c.day >= 1])
    assert [result.considered for result in planner.history] == \
        [whole, later, later]
    assert whole > later, "the later re-check must actually drop candidates"
    assert len(calls) == sum(result.considered for result in planner.history)


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all integrality tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())