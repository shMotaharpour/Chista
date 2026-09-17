"""The day plan replayed on the engine: every op must move what it promises.

See `docs/ARCHITECTURE.md` §2 and issue #57.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.dispatch import PASS_ACTION, dispatch_plan
from day.routing import (MAX_ORDERS_PER_TURN, MOVE_OPS, hire_cost,
                               merge_market, plan_day, spawn_position)
from tile_dp.chains import chains_for
from tile_dp.tile_state import decode_tile
from world.fast_sim import FastSim

TILES = ((1, 1), (2, 1))


def _fingerprint(obs: dict) -> tuple:
    """Everything our side can change in one turn."""
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    return (
        tuple(sorted(private.get("shed", {}).items())),
        tuple(sorted(private.get("seeds", {}).items())),
        tuple(tuple(sorted(bag.items())) for bag in private.get("inventories", [])),
        float(farm["money"]),
        tuple(farm["farmer"]),
        tuple(tuple(h) for h in farm["hands"]),
        json.dumps(farm["tiles"], sort_keys=True),
        json.dumps(obs["market"]["inventory"], sort_keys=True),
    )


def _unit_pos(obs: dict, index: int) -> tuple[int, int]:
    """Where unit `index` stands; (-1,-1) for a hand that does not exist yet."""
    farm = obs["farms"][obs["player"]]
    if index < 0:                      # a market order belongs to no unit
        return (-1, -1)
    if index == 0:
        return tuple(farm["farmer"])
    hands = farm["hands"]
    return tuple(hands[index - 1]) if index - 1 < len(hands) else (-1, -1)


def _bag(obs: dict, index: int) -> dict:
    inv = obs["private"].get("inventories", [])
    return dict(inv[index]) if 0 <= index < len(inv) else {}


def _tile_at(obs: dict, pos: tuple[int, int]):
    if pos[0] < 0:
        return None
    farm = obs["farms"][obs["player"]]
    return farm["tiles"][pos[1]][pos[0]]


def _gain(before: dict, after: dict, item: str) -> int:
    return int(after.get(item, 0)) - int(before.get(item, 0))


def _op_changed(op: tuple, before: dict, after: dict, index: int) -> tuple[bool, str]:
    """Did this op move what it promises? (False, why) is a silent no-op."""
    kind = str(op[0])
    b_pos, a_pos = _unit_pos(before, index), _unit_pos(after, index)
    b_tile, a_tile = _tile_at(before, b_pos), _tile_at(after, b_pos)
    b_bag, a_bag = _bag(before, index), _bag(after, index)
    b_shed, a_shed = before["private"].get("shed", {}), after["private"].get("shed", {})
    b_seeds, a_seeds = before["private"].get("seeds", {}), after["private"].get("seeds", {})
    b_money = float(before["farms"][before["player"]]["money"])
    a_money = float(after["farms"][after["player"]]["money"])

    if kind in MOVE_OPS:
        return (a_pos != b_pos), f"stayed at {a_pos}"
    if kind == "PLANT":
        ok = isinstance(a_tile, dict) and a_tile.get("kind") == "PLANT"
        return ok, f"tile at {b_pos} is {a_tile!r}"
    if kind == "WATER":
        ok = isinstance(a_tile, dict) and bool(a_tile.get("watered_today"))
        return ok, f"watered_today={a_tile.get('watered_today') if isinstance(a_tile, dict) else a_tile!r}"
    if kind == "HARVEST":
        gained = sum(a_bag.values()) - sum(b_bag.values())
        drained = (isinstance(b_tile, dict) and isinstance(a_tile, dict)
                   and int(a_tile.get("yield_units", 0)) < int(b_tile.get("yield_units", 0)))
        return (gained > 0 or drained), f"bag {b_bag} -> {a_bag}, tile {b_tile} -> {a_tile}"
    if kind == "FERTILIZE":
        ok = (isinstance(a_tile, dict)
              and int(a_tile.get("fertilized_until_day", 0))
              > int(b_tile.get("fertilized_until_day", 0) if isinstance(b_tile, dict) else 0))
        return ok, f"fertilized_until_day unchanged on {a_tile!r}"
    if kind == "DIG":
        return (a_tile is None and b_tile is not None), f"tile is {a_tile!r}"
    if kind in ("BUILD_COOP", "BUILD_PASTURE"):
        want = kind.split("_", 1)[1]
        ok = isinstance(a_tile, dict) and a_tile.get("kind") == want
        return ok, f"tile is {a_tile!r}, wanted {want}"
    if kind == "PLACE":
        ok = isinstance(a_tile, dict) and "animal" in a_tile
        return ok, f"no animal on {a_tile!r}"
    if kind == "FEED":
        ok = isinstance(a_tile, dict) and bool(a_tile.get("fed_today"))
        return ok, f"fed_today not set on {a_tile!r}"
    if kind == "CARE":
        ok = isinstance(a_tile, dict) and bool(a_tile.get("cared_today"))
        return ok, f"cared_today not set on {a_tile!r}"
    if kind == "COLLECT_FERTILIZER":
        return (_gain(b_bag, a_bag, "FERTILIZER") > 0), f"bag {b_bag} -> {a_bag}"
    if kind == "PICKUP":
        item = str(op[1])
        ok = (_gain(b_shed, a_shed, item) < 0 and _gain(b_bag, a_bag, item) > 0)
        return ok, f"shed {b_shed.get(item, 0)}->{a_shed.get(item, 0)}, bag {b_bag.get(item, 0)}->{a_bag.get(item, 0)}"
    if kind == "DROP":
        moved = sum(a_shed.values()) - sum(b_shed.values())
        return (moved > 0 and not a_bag), f"shed +{moved}, bag {a_bag}"
    if kind == "HIRE":
        grew = len(after["farms"][after["player"]]["hands"]) > len(
            before["farms"][before["player"]]["hands"])
        return grew, "no hand appeared"
    if kind == "BUY_SEED":
        item = str(op[1])
        return (_gain(b_seeds, a_seeds, item) > 0 and a_money < b_money), \
            f"seeds {b_seeds.get(item, 0)}->{a_seeds.get(item, 0)}, money {b_money}->{a_money}"
    if kind in ("BUY_PRODUCT", "BUY_ANIMAL"):
        item = str(op[1])
        return (_gain(b_shed, a_shed, item) > 0 and a_money < b_money), \
            f"shed {b_shed.get(item, 0)}->{a_shed.get(item, 0)}, money {b_money}->{a_money}"
    if kind == "SELL":
        item = str(op[1])
        return (_gain(b_shed, a_shed, item) < 0 and a_money > b_money), \
            f"shed {b_shed.get(item, 0)}->{a_shed.get(item, 0)}, money {b_money}->{a_money}"
    return (_fingerprint(after) != _fingerprint(before)), "no rule for this op"


def _actions_in(action: dict) -> list[tuple[int, tuple]]:
    """(unit index, op) for every non-PASS op, farmer 0 then hands in order."""
    out: list[tuple[int, tuple]] = []
    farmer = action.get("farmer", [])
    if farmer and farmer[0] != "PASS":
        out.append((0, tuple(farmer)))
    for i, hand in enumerate(action.get("hands", [])):
        if hand and hand[0] != "PASS":
            out.append((i + 1, tuple(hand)))
    for order in action.get("market", []):
        if order:
            out.append((-1, tuple(order)))
    return out


def _pick_chain(tile, day: int, entity: str | None) -> tuple[str, ...] | None:
    """The graph's own applicable chains for this tile, preferring the money path."""
    state = decode_tile(tile, day)
    age = state.age if state.kind in ("PLANT", "ANIMAL") else None
    chains = chains_for(state.kind, age=age, yield_units=state.yield_units,
                        entity=state.crop or state.animal)
    # prefer the money path, and never destroy the crop to reach it: a chain that
    # digs up the tile is realisable but it makes the instrument measure replanting
    for wanted in ("HARVEST", "PLANT", "WATER"):
        for chain in chains:
            if wanted in chain and "DIG" not in chain:
                return chain
    for chain in chains:
        if "DIG" not in chain:
            return chain
    return chains[0] if chains else None


def _tiles_now(obs: dict) -> list[tuple[tuple[int, int], tuple[str, ...], str | None]]:
    """One realisable chain per owned tile, from the graph's own filter."""
    farm = obs["farms"][obs["player"]]
    day = int(obs["day"])
    out = []
    for xy in TILES:
        raw = farm["tiles"][xy[1]][xy[0]]
        if raw == "LOCKED":
            continue
        crop = raw.get("crop") if isinstance(raw, dict) else None
        chain = _pick_chain(raw, day, crop)
        if chain is None:
            continue
        out.append((xy, chain, crop or "WHEAT"))
    return out


def _plan_for(sim: FastSim, **over):
    obs = sim.observations()[0]
    farm = obs["farms"][obs["player"]]
    kwargs = dict(new_hands=1, bags=[obs["private"]["inventories"][0]],
                  shed=obs["private"]["shed"], money=farm["money"],
                  hires_today=farm["hires_today"],
                  sells=[[] for _ in range(24)],
                  prices=obs["market"]["prices"])
    kwargs.update(over)
    return plan_day([((1, 1), ("PLANT", "WATER"), "WHEAT"),
                     ((2, 1), ("PLANT", "WATER"), "WHEAT"),
                     ((1, 2), ("PLANT", "WATER"), "CARROT")],
                    [tuple(farm["farmer"])], **kwargs), obs


def test_every_op_moves_what_it_promises() -> None:
    """Eight compiled days, every emitted op checked against its promise."""
    sim = FastSim(configuration={"seed": 3, "episodeSteps": 720}, validate="dev")
    silent: list[tuple[int, int, int, tuple, str]] = []
    checked = 0
    kinds: set[str] = set()
    start_day = int(sim.observations()[0]["day"])
    while int(sim.observations()[0]["day"]) < start_day + 8 and not sim.done:
        obs = sim.observations()[0]
        tiles = _tiles_now(obs)
        if not tiles:
            break
        farm = obs["farms"][obs["player"]]
        plan = plan_day(tiles, [tuple(farm["farmer"])], new_hands=1,
                        bags=[obs["private"]["inventories"][0]],
                        shed=obs["private"]["shed"], money=farm["money"],
                        hires_today=farm["hires_today"],
                        sells=[[] for _ in range(24)],
                        prices=obs["market"]["prices"])
        day = int(obs["day"])
        while int(obs["day"]) == day and not sim.done:
            action = dispatch_plan(plan.as_plan(), obs)
            before = obs
            sim.step([action, PASS_ACTION])
            obs = sim.observations()[0]
            if int(obs["day"]) != day:
                break                      # the nightly refresh rewrites positions
            for index, op in _actions_in(action):
                checked += 1
                kinds.add(str(op[0]))
                ok, why = _op_changed(op, before, obs, index)
                if not ok:
                    silent.append((day, int(before.get("hour", 0)), index, op, why))
    assert checked > 40, f"the instrument barely saw work: {checked} ops"
    assert {"PLANT", "WATER", "HARVEST"} <= kinds, f"missed a stage: {sorted(kinds)}"
    assert kinds & set(MOVE_OPS), f"no travel was exercised: {sorted(kinds)}"
    assert "DROP" in kinds, f"no harvest ever reached the shed: {sorted(kinds)}"
    assert not silent, (
        f"{len(silent)} silent no-op op(s) — the engine refused them in silence: "
        f"{silent[:4]}")


def test_the_plan_actually_farms_the_day_it_promised() -> None:
    """The day as an outcome: seeds bought, crops planted, a hand hired."""
    sim = FastSim(configuration={"seed": 3, "episodeSteps": 720}, validate="dev")
    plan, obs = _plan_for(sim)
    day = int(obs["day"])
    hired_mid_day = 0
    while int(obs["day"]) == day and not sim.done:
        sim.step([dispatch_plan(plan.as_plan(), obs), PASS_ACTION])
        obs = sim.observations()[0]
        if int(obs["day"]) == day:
            hired_mid_day = max(hired_mid_day,
                                len(obs["farms"][obs["player"]]["hands"]))
    farm = sim.observations()[0]["farms"][0]
    planted = [t for row in farm["tiles"] for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT"]
    assert planted, "the day ended with nothing planted"
    assert hired_mid_day == 1, (
        f"the hand was hired for {hired_mid_day} turn(s): hands are cleared "
        "nightly (F039), so this has to be read inside the day")
    assert farm["money"] != 3000.0, "money never moved: the day did nothing"


def test_the_hand_spawns_after_the_turns_unit_actions() -> None:
    """HIRE settles after the units act, so the hand takes the farmer's old tile (F060)."""
    sim = FastSim(configuration={"seed": 3, "episodeSteps": 720}, validate="dev")
    obs = sim.observations()[0]
    farm = obs["farms"][obs["player"]]
    start = tuple(farm["farmer"])
    sim.step([{"farmer": ["WEST"], "hands": [], "market": [["HIRE"]]}, PASS_ACTION])
    real = tuple(sim.observations()[0]["farms"][0]["hands"][0])
    after_move = (start[0] - 1, start[1])
    assert real == spawn_position([after_move]), (
        f"engine spawned at {real}, the post-move rule says "
        f"{spawn_position([after_move])}")
    assert real != spawn_position([start]), (
        "the hand took the tile the farmer was standing on: the engine settles "
        "HIRE after that turn's unit actions, so the spawn must be predicted "
        "from the post-move positions")
    assert real == start, f"expected the farmer's old tile {start}, got {real}"


def test_the_plan_never_buys_what_it_cannot_pay_for() -> None:
    """A refused buy is a silent no-op: a broke farm gets a PASS day."""
    sim = FastSim(configuration={"seed": 3, "episodeSteps": 720}, validate="dev")
    plan, _obs = _plan_for(sim, money=0.0)
    buys = [order for row in plan.market for order in row
            if order and str(order[0]).startswith("BUY")]
    assert not buys, f"a purse of 0 still queued {buys[:3]}"
    for unit in plan.units:
        assert all(op == ("PASS",) for op in unit), \
            "an op that needs an unaffordable buy was still emitted"


def test_assignment_is_nearest_first_and_idle_units_are_counted() -> None:
    """Nearest tile per unit, farmer first; idle units counted."""
    tiles = [((1, 1), ("PLANT",), "WHEAT"), ((2, 1), ("PLANT",), "WHEAT"),
             ((1, 2), ("PLANT",), "CARROT")]
    plan = plan_day(tiles, [(1, 1), (9, 9)], new_hands=0, money=10000.0,
                    hires_today=0, shed={}, prices={})
    assert plan.assignments[0] == (1, 1), plan.assignments
    # (2,1) and (1,2) are equidistant from (9,9): the tie goes to the tile
    # position, so the assignment is a function of its inputs
    assert plan.assignments[1] == (1, 2), plan.assignments
    assert plan.idle_units == 0
    one_tile = plan_day(tiles[:1], [(1, 1), (9, 9)], new_hands=0, money=10000.0,
                        shed={}, prices={})
    assert one_tile.idle_units == 1


def test_the_market_queue_follows_f032_and_the_cap() -> None:
    """F032 order inside a turn, and the 11th order raises."""
    from day.routing import Need
    sells = [[["SELL", "WHEAT", 5]], []]
    needs = [Need(hour=3, order=("BUY_SEED", "WHEAT", 1), reason="PLANT")]
    queue, unplaced = merge_market(sells, needs, shed_total=0.0, capacity=100,
                                   hires=2)
    assert [str(o[0]) for o in queue[0]] == ["SELL", "HIRE", "HIRE", "BUY_SEED"], \
        queue[0]        # F032 inside a turn: sells, then hires, then purchases
    assert not unplaced, unplaced
    # a product buy lands in the shed, which has no room: it must not be queued,
    # and it must not vanish either
    full = [Need(hour=0, order=("BUY_PRODUCT", "WHEAT", 1), reason="FEED")]
    queue2, unplaced2 = merge_market([], full, shed_total=100.0, capacity=100)
    assert not queue2[0], "a buy into a full shed must not be queued"
    assert unplaced2, "an unplaceable buy must be returned, not dropped quietly"
    crowded = [["SELL"] * (MAX_ORDERS_PER_TURN + 1)]
    try:
        merge_market(crowded, [Need(hour=0, order=("HIRE",), reason="x")],
                     shed_total=0.0, capacity=100)
        raise AssertionError("11 orders in a turn must raise (F031)")
    except ValueError:
        pass


def test_the_compilers_own_output_dispatches_per_hour() -> None:
    """The compiler's market rows are per-hour; the dispatcher must read them so."""
    from agent.dispatch import market_at
    sim = FastSim(configuration={"seed": 3, "episodeSteps": 720}, validate="dev")
    plan, _obs = _plan_for(sim)
    market = plan.as_plan()["market"]
    assert market_at(market, 0), "hour 0 should carry the day's buys and hires"
    assert market_at(market, 23) == [] or all(
        order[0] == "SELL" for order in market_at(market, 23)), market_at(market, 23)
    assert market_at(market, 1) != market_at(market, 0), (
        "hour 1 repeats hour 0: the rows are being read as a flat order list")


def test_hire_cost_is_the_engines_ladder() -> None:
    """The engine's Fibonacci hire ladder."""
    assert hire_cost(0, 1) == 1.0
    assert hire_cost(0, 2) == 2.0
    assert hire_cost(1, 1) == 1.0
    assert hire_cost(2, 3) == 2 + 3 + 5


def main() -> int:
    failures = 0
    checks = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            checks += 1
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:                      # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"FAIL day plan: {failures}/{checks} guard(s) failed")
        return 1
    print(f"PASS day plan: {checks} guards (per-op instrument over 8 days, the "
          "day's outcome, the spawn rule, the purse, the assignment, F032 order "
          "and the cap, the hire ladder)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
