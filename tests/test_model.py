"""Guards for `world/model.py`: one vocabulary, and the drift it must stop.

See `docs/ARCHITECTURE.md` §1.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import inspect

from kaggle_environments.envs.kaggriculture import kaggriculture as K

import agent.tile_dp.chains as chains
from agent.wsr import models as wsr
from agent.world import model as M

#: Hand-spelled good names per module (R002 debt), measured 2026-09-17. A module
#: may only shrink. See ARCHITECTURE §5.
NAME_DEBT = {
    "bench/bench_market_analyzer.py": 35,
    "planner/master.py": 20,
    "tile_dp/chains.py": 18,
    "tile_dp/graph.py": 18,
    "agent/greedy.py": 6,
    "planner/columns.py": 4,
    "agent/replan.py": 4,
    "bench/bench_market_forecast.py": 4,
    "agent/dispatch.py": 2,
    "belief/stubs.py": 2,
    "planner/repair.py": 2,
    "bench/bench_turn_budget.py": 1,
    "belief/schemas.py": 1,
    "bench/oracle_transitions.py": 1,
}

#: Vendored code, tests that assert against the engine, and the probes.
EXEMPT = ("opponents/", "tests/", "KaggleProbes/", "world/model.py")


def _engine_action_names() -> frozenset[str]:
    """The engine's action literals, re-extracted here so the model cannot drift."""
    names: set[str] = set()
    for handler in ("_apply_unit_action", "_commit_unit", "_parse_order"):
        source = inspect.getsource(getattr(K, handler))
        names.update(re.findall(r'op == "([A-Z_]+)"', source))
        for group in re.findall(r"op in \(([^)]*)\)", source):
            names.update(re.findall(r'"([A-Z_]+)"', group))
    names.update(str(move) for move in K.FARMER_MOVES)
    names.add("PASS")
    return frozenset(names)


def _engine_tile_kinds() -> frozenset[str]:
    return frozenset(re.findall(r'"kind": "([A-Z_]+)"', inspect.getsource(K)))


def _spelled(path: pathlib.Path) -> int:
    pattern = re.compile('"(' + "|".join(M.GOODS + M.ANIMALS) + ')"')
    return len(pattern.findall(path.read_text()))


def test_names_come_from_the_engine() -> None:
    """Goods, crops, species, movement and the shed tiles are the engine's."""
    assert M.GOODS == tuple(K.PRODUCTS)
    assert M.PRODUCTS == M.GOODS
    assert M.CROPS == tuple(K.CROPS)
    assert M.ANIMALS == tuple(K.ANIMALS)
    assert M.MOVEMENT == tuple(K.FARMER_MOVES)
    assert M.SHED_ACCESS == frozenset((int(x), int(y))
                                      for x, y in K._shed_access_tiles(10))


def test_resources_are_not_products() -> None:
    """Resources are bought or consumed; products are sold. Wheat and fertiliser are both."""
    assert M.RESOURCES == ("LABOR_HOURS", "FERTILIZER", "WHEAT", "SEED_WHEAT",
                           "SEED_CARROT", "SEED_TOMATO", "SEED_STRAWBERRY",
                           "SEED_MELON", "ANIMAL_GOOSE", "ANIMAL_COW",
                           "ANIMAL_SHEEP"), M.RESOURCES
    assert len(M.RESOURCES) == 11 and len(M.PRODUCTS) == 9
    assert set(M.RESOURCES) & set(M.PRODUCTS) == set(M.DUAL) == {"WHEAT", "FERTILIZER"}
    for product in ("CARROT", "MILK", "WOOL", "EGG", "MELON"):
        assert product in M.PRODUCTS and product not in M.RESOURCES, (
            f"{product} is a product, not a resource")
    for resource in ("SEED_WHEAT", "ANIMAL_COW", "LABOR_HOURS"):
        assert resource in M.RESOURCES and resource not in M.PRODUCTS


def test_the_graph_vector_is_the_union_in_the_stored_order() -> None:
    """The DP's 18 columns are resources + products, order-compatible with the graph."""
    assert set(M.VECTOR) == set(M.RESOURCES) | set(M.PRODUCTS)
    assert M.PRICE_VECTORS == ("PRODUCE", "INPUT")
    assert M.RESULT_VECTORS == ("COST", "PRODUCE")
    assert len(M.VECTOR) == 18
    assert tuple(chains.RESOURCE_NAMES) == M.VECTOR, (
        "the stored column order changed: the shipped graph indexes it")


def test_the_market_and_the_worker_do_not_share_ops() -> None:
    """No market action is a worker op; the contractor and the WSR stay out of the market."""
    assert not (set(M.TILE_OPS) & set(M.MARKET_ACTIONS))
    # The contractor's vocabulary and a worker's differ by exactly the shed trips,
    # the moves and the pass: a tile chain can never name a PICKUP or a DROP.
    assert not (set(M.SHED_OPS) & set(M.CHAIN_OPS)), "a tile chain cannot name a shed trip"
    # Every worker op is an engine action: no invented name, no abstract BUILD.
    assert set(M.WORKER_OPS) <= set(M.ACTIONS), sorted(set(M.WORKER_OPS) - set(M.ACTIONS))
    assert not (set(M.WORKER_OPS) & set(M.ABSTRACT_OPS))
    # A chain may name the buys it needs and nothing else of the market's.
    assert set(M.CHAIN_OPS) & set(M.MARKET_ACTIONS) == set(M.MARKET_BUYS), (
        f"a chain names a market op that is not an input buy: "
        f"{sorted(set(M.CHAIN_OPS) & set(M.MARKET_ACTIONS) - set(M.MARKET_BUYS))}")
    assert set(M.MARKET_BUYS) <= set(M.MARKET_ACTIONS), "a buy is not a market action"
    # The tile vocabulary is engine names too, except the abstract ones.
    assert set(M.TILE_OPS) - set(M.ABSTRACT_OPS) <= set(M.ACTIONS)
    assert set(M.TILE_OPS) & set(M.ABSTRACT_OPS) == {"BUILD"}
    # And the abstract BUILD compiles into real engine actions, one per structure.
    for species in M.ANIMALS:
        assert set(M.compile_op("BUILD", species)) <= set(M.ACTIONS)
    assert {"SELL", "BUY_LAND", "HIRE"} <= set(M.MARKET_ACTIONS)
    assert {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"} <= set(M.MARKET_ACTIONS)
    for op in sorted(set(M.CHAIN_OPS) - {"NO_ACTION"}):
        if op.startswith("BUY_"):
            assert op in M.MARKET_ACTIONS, f"{op} is a market op"
        else:
            assert op in M.TILE_OPS, f"{op} is not a tile op"


def test_the_action_vocabulary_matches_the_engine() -> None:
    """The hard-coded `ACTIONS` still equals what the engine's handlers accept."""
    assert set(M.ACTIONS) == _engine_action_names()
    assert set(M.TILE_KINDS) == _engine_tile_kinds()
    for op in ("PASS", "PLANT", "WATER", "HARVEST", "PICKUP", "DROP", "PLACE",
               "FEED", "CARE", "COLLECT_FERTILIZER", "FERTILIZE", "DIG",
               "BUILD_COOP", "BUILD_PASTURE", "SELL", "BUY_SEED", "BUY_PRODUCT",
               "BUY_ANIMAL", "HIRE", "BUY_LAND"):
        assert op in M.ACTIONS, f"the engine accepts {op} and the model lost it"
    assert set(M.MOVEMENT) <= set(M.ACTIONS)


def test_every_chain_op_compiles_to_an_engine_action() -> None:
    """The compile table is total over the chain vocabulary."""
    for op in sorted(M.CHAIN_OPS):
        entity = {"PLANT": "WHEAT", "BUILD": "COW"}.get(
            op, "COW" if op in M.PLACING_OPS else None)
        assert M.compile_op(op, entity)[0] in M.ACTIONS
    assert "PLACE_ANIMAL" not in M.CHAIN_OPS, "the duplicate is retired"
    for bad in (("NOT_AN_OP", None), ("PLANT", "COW"), ("BUILD", "WHEAT")):
        try:
            M.compile_chain((bad[0],), bad[1])
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_every_registry_chain_compiles() -> None:
    """Every chain the DP can choose has a worker-day the engine accepts."""
    for ops in chains.CHAIN_NAMES:
        for entity in ("WHEAT", "COW", None):
            try:
                compiled = M.compile_chain(ops, entity)
            except ValueError:
                continue
            for action in compiled:
                assert action[0] in M.ACTIONS, (ops, entity, action)
            break
        else:
            raise AssertionError(f"no entity compiles {ops}")


def test_the_duplicate_vocabularies_are_gone() -> None:
    """The WSR's own enums are deleted; its sets are views of the one model."""
    for gone in ("Action", "TileKind", "MajorTaskType", "ANIMAL_STRUCTURE"):
        assert not isinstance(getattr(wsr, gone, None), type), (
            f"day/models.py still defines {gone}")
    assert wsr.Item is M.Item, "the scheduling layer has its own Item again"
    assert wsr.Action is M.Action, "the scheduling layer has its own action enum"
    assert wsr.TileKind is M.TileKind
    assert {a for a in wsr.MOVEMENT_ACTIONS} == set(M.MOVEMENT) | {"PASS"}
    assert {i for i in wsr.PRODUCT_ITEMS} == set(M.PRODUCTS)
    for name, chain in wsr.MAJOR_CHAINS.items():
        assert set(chain) <= set(M.CHAIN_OPS), f"{name}: {chain} is not chain vocabulary"


def test_no_module_may_grow_its_own_good_names() -> None:
    """The name debt is a ratchet: every module may only shrink."""
    measured: dict[str, int] = {}
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if "__pycache__" in rel or any(rel.startswith(p) or rel == p for p in EXEMPT):
            continue
        count = _spelled(path)
        if count:
            measured[rel] = count
    grew = {f: (n, NAME_DEBT.get(f, 0)) for f, n in measured.items()
            if n > NAME_DEBT.get(f, 0)}
    assert not grew, (
        f"hand-spelled good names beyond the pin (file: measured vs pin): {grew}. "
        "Import them from world/model.py.")
    assert set(measured) <= set(NAME_DEBT), (
        f"new modules with hand-spelled names: {sorted(set(measured) - set(NAME_DEBT))}")


def main() -> int:
    failures = checks = 0
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
        print(f"FAIL model: {failures}/{checks} guard(s) failed")
        return 1
    print(f"PASS model: {checks} guards (engine names, 11 resources vs 9 products, "
          f"the 18-column vector, market/worker split, {len(M.ACTIONS)} extracted "
          f"actions, the compile table over {len(chains.CHAIN_NAMES)} chains, the "
          f"legacy pins, the name ratchet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
