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

from kaggle_environments.envs.kaggriculture import kaggriculture as K

import tile_dp.chains as chains
from day import models as wsr
from world import model as M

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
    assert len(M.VECTOR) == 18
    assert tuple(chains.RESOURCE_NAMES) == M.VECTOR, (
        "the stored column order changed: the shipped graph indexes it")


def test_the_market_and_the_worker_do_not_share_ops() -> None:
    """No market action is a worker op; the contractor and the WSR stay out of the market."""
    assert not (M.WORKER_OPS & M.MARKET_ACTIONS)
    assert {"SELL", "BUY_LAND", "HIRE"} <= M.MARKET_ACTIONS
    assert {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"} <= M.MARKET_ACTIONS
    for op in sorted(M.CHAIN_OPS - {"NO_ACT"}):
        if op.startswith("BUY_"):
            assert op in M.MARKET_ACTIONS, f"{op} is a market op"
        else:
            assert op in M.WORKER_OPS, f"{op} is not a worker op"


def test_the_action_vocabulary_is_extracted() -> None:
    """`ACTIONS` is read out of the engine's handlers, not typed."""
    assert M.ACTIONS == M.engine_action_names()
    for op in ("PASS", "PLANT", "WATER", "HARVEST", "PICKUP", "DROP", "PLACE",
               "FEED", "CARE", "COLLECT_FERTILIZER", "FERTILIZE", "DIG",
               "BUILD_COOP", "BUILD_PASTURE", "SELL", "BUY_SEED", "BUY_PRODUCT",
               "BUY_ANIMAL", "HIRE", "BUY_LAND"):
        assert op in M.ACTIONS, f"the engine accepts {op} and the model lost it"
    assert set(M.MOVEMENT) <= M.ACTIONS


def test_every_chain_op_compiles_to_an_engine_action() -> None:
    """The compile table is total over the chain vocabulary."""
    for op in sorted(M.CHAIN_OPS):
        entity = {"PLANT": "WHEAT", "BUILD": "COW"}.get(
            op, "COW" if op in M.PLACING_OPS else None)
        assert M.compile_op(op, entity)[0] in M.ACTIONS
    assert M.compile_op("PLACE_ANIMAL", "COW") == ("PLACE", "COW"), "the legacy alias"
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


def test_the_legacy_vocabularies_stay_pinned() -> None:
    """The WSR's enums name real objects; its two divergences are pinned."""
    engine_names = set(M.GOODS) | set(M.CROPS) | set(M.ANIMALS)
    for item in wsr.Item:
        assert item.value.upper() in engine_names, f"{item.value!r} is not an engine name"
    for action in wsr.MinorActionType:
        assert action.value in M.ACTIONS, f"{action.value} is not an engine action"
    assert {i.value.upper() for i in wsr.PRODUCT_ITEMS} == {"MILK", "WOOL", "EGG"}, (
        "PRODUCT_ITEMS changed: if it now equals the products, delete this pin "
        "and the enum (ARCHITECTURE §5)")
    assert {a.value for a in wsr.MOVEMENT_ACTIONS} == set(M.MOVEMENT) | {"PASS"}


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
