"""Guards for `secretary/routing.py` — the day compiler.

Every number and every op name here comes from the engine (`kaggriculture.py`),
and the module's own precondition check runs on every route it produces: a
PICKUP or a DROP the route places anywhere but a shed-access tile is refused by
the engine **in silence**, so the compiler refuses to emit one (R007: the check
was seen to fail — see the PR's evidence section).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from secretary.routing import (MOVE_OPS, shed_access, nearest_shed, op_turns,
                               route_unit, walk)

# Every op the compiler may emit, with the engine line that accepts it
# (`_apply_unit_action`, kaggriculture.py). Anything outside this set is not a
# legal action string at all.
LEGAL_OPS = frozenset(
    list(MOVE_OPS) + ["PASS", "PICKUP", "DROP", "PLACE", "PLANT", "WATER",
                      "HARVEST", "FERTILIZE", "DIG", "FEED", "CARE",
                      "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE"])


def test_walk_is_shortest_and_legal() -> None:
    """`walk` reaches its goal in Manhattan steps, all of them on the board."""
    steps = walk((0, 0), (4, 5))
    assert len(steps) == 9, f"expected 9 steps, got {len(steps)}"
    x, y = 0, 0
    for (op,) in steps:
        dx, dy = MOVE_OPS[op]
        x, y = x + dx, y + dy
        assert 0 <= x < 10 and 0 <= y < 10, f"{op} leaves the board"
    assert (x, y) == (4, 5), f"walk ended at {(x, y)}"
    assert walk((3, 3), (3, 3)) == [], "a zero-distance walk emits nothing"


def test_shed_access_is_the_engines_own_list() -> None:
    """PICKUP/DROP work on exactly four tiles — the engine's, not a copy."""
    assert shed_access() == tuple((int(a), int(b))
                                 for a, b in K._shed_access_tiles(10)), \
        "the compiler's shed tiles drifted from the engine's"
    assert len(shed_access()) == 4
    assert nearest_shed((0, 0)) in shed_access()


def test_op_turns_matches_the_engine_vocabulary() -> None:
    """The expansion names real ops: PLANT/PLACE carry their entity, BUILD a structure."""
    turns = op_turns(("PLANT", "WATER", "HARVEST"), "WHEAT")
    assert turns == [("PLANT", "WHEAT"), ("WATER",), ("HARVEST",)], turns
    assert op_turns(("BUILD", "PLACE"), "COW") == [("BUILD_PASTURE",),
                                                   ("PLACE", "COW")]
    assert op_turns(("BUILD",), "GOOSE") == [("BUILD_COOP",)]
    assert op_turns(("NO_ACT",), None) == [("PASS",)]
    for ops, entity in ((("PLANT",), "WHEAT"), (("PLACE",), "COW"),
                        (("BUILD",), "SHEEP")):
        for step in op_turns(ops, entity):
            assert step[0] in LEGAL_OPS, f"{step} is not an engine op"


def test_a_crop_chain_needs_a_seed_and_not_a_carry() -> None:
    """PLANT is fed by the market into `private["seeds"]`: no trip, no pickup."""
    route = route_unit(("PLANT", "WATER", "HARVEST"), "WHEAT", (1, 1), unit=0)
    emitted = [op for op in route.ops if op and op[0] != "PASS"]
    kinds = [op[0] for op in emitted]
    assert kinds[:3] == ["PLANT", "WATER", "HARVEST"], kinds
    assert "PICKUP" not in kinds, "a crop needs no pickup to be planted"
    buys = [n for n in route.needs if n.order[0] == "BUY_SEED"]
    assert len(buys) == 1 and buys[0].order == ("BUY_SEED", "WHEAT", 1)
    plant_turn = kinds.index("PLANT")
    assert buys[0].hour < plant_turn, (
        f"the seed buy may land at hour {buys[0].hour} but the plant is at "
        f"turn {plant_turn}: the market runs after the units, so the seed would "
        "not exist yet")
    # HARVEST fills the bag, and a SELL can only reach the shed: the route must
    # walk back and DROP, or the harvest is unselleable today.
    assert "DROP" in kinds, f"harvest with no drop: {kinds}"
    assert route.drop_hour is not None and route.ops[route.drop_hour][0] == "DROP"


def test_a_fertilize_chain_buys_carries_and_fertilizes() -> None:
    """FERTILIZE eats FERTILIZER from the bag: buy, walk to the shed, carry back."""
    route = route_unit(("FERTILIZE",), "WHEAT", (0, 0), unit=0)
    kinds = [op[0] for op in route.ops if op and op[0] != "PASS"]
    assert kinds[-1] == "FERTILIZE", kinds
    assert "PICKUP" in kinds, "fertilizer must be carried to the tile"
    pickup_turn = kinds.index("PICKUP")
    need = [n for n in route.needs if n.order[0] == "BUY_PRODUCT"][0]
    assert need.order == ("BUY_PRODUCT", "FERTILIZER", 1)
    assert need.hour < pickup_turn, (
        f"buy lands at hour {need.hour}, pickup at turn {pickup_turn}: the "
        "market of a turn runs after its units (F030)")
    # the unit walks to the shed, picks up, and walks back to the tile
    assert kinds[:pickup_turn] == ["NORTH", "WEST"] or all(
        k in MOVE_OPS for k in kinds[:pickup_turn]), kinds
    assert route.walked >= 2


def test_an_animal_chain_places_what_it_picked_up() -> None:
    """PLACE consumes the animal from the bag, and the buy lands in the shed."""
    route = route_unit(("BUILD", "PLACE"), "COW", (0, 0), unit=0)
    kinds = [op[0] for op in route.ops if op and op[0] != "PASS"]
    assert kinds.count("PICKUP") == 1 and kinds[-1] == "PLACE", kinds
    assert [n for n in route.needs if n.order[0] == "BUY_ANIMAL"] == [
        n for n in route.needs if n.order == ("BUY_ANIMAL", "COW", 1)], route.needs


def test_a_chain_that_does_not_fit_is_reported_not_emitted() -> None:
    """Ops the day cannot carry are dropped by name; nothing late is emitted."""
    far = (0, 0)
    route = route_unit(("PLANT", "WATER", "WATER", "HARVEST", "FERTILIZE",
                        "HARVEST", "HARVEST"), "WHEAT", far, unit=0, hours=6)
    assert route.dropped, "a 7-op chain with two shed trips cannot fit in 6 turns"
    assert route.hours_used <= 6, route.hours_used
    for op in route.ops:
        if op and op[0] != "PASS":
            assert op[0] in LEGAL_OPS, op
    # every dropped name is a chain op, and none of them was emitted late
    for name in route.dropped:
        assert name.split()[0] in {op[0] for op in op_turns(
            ("PLANT", "WATER", "WATER", "HARVEST", "FERTILIZE", "HARVEST",
             "HARVEST"), "WHEAT")}, name


def test_the_route_never_exceeds_the_day() -> None:
    """24 turns, padded with PASS, for the real chains and real positions."""
    for pos in ((0, 0), (4, 4), (9, 9), (5, 4)):
        for ops, entity in ((("PLANT", "WATER", "HARVEST"), "WHEAT"),
                            (("FERTILIZE", "WATER", "HARVEST"), "CARROT"),
                            (("BUILD", "PLACE", "FEED"), "COW"),
                            (("NO_ACT",), None)):
            route = route_unit(ops, entity, pos, unit=1)
            assert len(route.ops) == 24, (ops, pos, len(route.ops))
            assert route.hours_used <= 24


def test_the_compiler_refuses_a_pickup_off_the_shed() -> None:
    """The precondition check can fail — re-introducing the bug must be caught.

    A route that emits PICKUP/DROP anywhere but a shed-access tile is a plan the
    engine refuses in silence (`:344, 359`), which is the whole failure class
    this module exists to close. Pinning that the check fires keeps it from
    becoming decoration.
    """
    import secretary.routing as routing
    original = routing.nearest_shed
    routing.nearest_shed = lambda pos, board=10: (1, 0)     # near, not shed-access
    try:
        fired = ""
        try:
            routing.route_unit(("FERTILIZE",), "WHEAT", (0, 0))
        except AssertionError as exc:
            fired = str(exc)
        assert "not a shed-access tile" in fired, (
            "an off-shed pickup was emitted without complaint: " + repr(fired))
    finally:
        routing.nearest_shed = original


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:                      # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"FAIL routing: {failures} guard(s) failed")
        return 1
    print("PASS routing: 8 guards (walk, shed tiles, vocabulary, crop chain, "
          "fertilize carry, animal carry, overflow, day length)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
