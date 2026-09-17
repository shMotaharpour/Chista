"""Guards for `world/model.py` — the one vocabulary, and the drift it must stop.

`docs/ARCHITECTURE.md` §1 says the engine is the vocabulary and no layer types its
own names. These guards make that checkable instead of aspirational:

- the canonical names equal their engine sources;
- the action vocabulary is **extracted from the engine's handlers**, so a new
  engine op cannot go missing;
- every chain in the DP's registry compiles to actions the engine accepts;
- the two legacy vocabularies (the DP's chain ops, the WSR secretary's enums) are
  measured against the model, and the divergences that remain are **pinned by
  number** — closing one shrinks the pin, and no new one can appear;
- the "no layer types its own good names" rule is a ratchet: a pinned count per
  module, which may only go down.
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
from secretary import models as wsr
from world import model as M

#: Hand-spelled good names per module (R002's debt), measured 2026-09-17. A module
#: may only shrink: the migration in `docs/ARCHITECTURE.md` §5 moves each of these
#: to `world/model.py`, and a new module with a count of its own fails the guard.
NAME_DEBT = {
    "planner/master.py": 20,
    "tile_dp/chains.py": 18,
    "tile_dp/graph.py": 18,
    "agent/greedy.py": 6,
    "secretary/routing.py": 6,
    "planner/columns.py": 4,
    "agent/replan.py": 4,
    "bench/bench_market_forecast.py": 4,
    "agent/dispatch.py": 2,
    "planner/repair.py": 2,
    "bench/bench_turn_budget.py": 1,
    "bench/oracle_transitions.py": 1,
}

#: Directories that are exempt: vendored opponents are other people's code, a test
#: naming a good is asserting against the engine rather than re-defining it, and
#: the Kaggle probes are measurement scripts keyed to the platform's own JSON.
EXEMPT = ("opponents/", "tests/", "KaggleProbes/", "world/model.py")


def _spelled_names(path: pathlib.Path) -> int:
    pattern = re.compile('"(' + "|".join(M.GOODS + M.ANIMALS) + ')"')
    return len(pattern.findall(path.read_text()))


def test_the_engine_is_the_source_of_every_name() -> None:
    """Goods, crops, species, movement and the shed tiles come from the engine."""
    assert M.GOODS == tuple(K.PRODUCTS), "the goods drifted from the engine"
    assert M.CROPS == tuple(K.CROPS)
    assert M.ANIMALS == tuple(K.ANIMALS)
    assert M.MOVEMENT == tuple(K.FARMER_MOVES), "movement ops are the engine's table"
    assert M.SHED_ACCESS == tuple((int(x), int(y))
                                 for x, y in K._shed_access_tiles(10))
    assert M.SELLABLE == M.GOODS, "the sellable set is the engine's product list"


def test_the_action_vocabulary_is_extracted_not_typed() -> None:
    """`ACTIONS` is what the engine's handlers accept, read out of their source."""
    assert M.ACTIONS == M.engine_action_names(), "the extraction is not what it says"
    for op in ("PASS", "PLANT", "WATER", "HARVEST", "PICKUP", "DROP", "PLACE",
               "FEED", "CARE", "COLLECT_FERTILIZER", "FERTILIZE", "DIG",
               "BUILD_COOP", "BUILD_PASTURE", "SELL", "BUY_SEED", "BUY_PRODUCT",
               "BUY_ANIMAL", "HIRE", "BUY_LAND"):
        assert op in M.ACTIONS, f"the engine accepts {op} and the model lost it"
    for move in K.FARMER_MOVES:
        assert move in M.ACTIONS


def test_every_chain_op_compiles_to_an_engine_action() -> None:
    """The compile table is total over the chain vocabulary, and lands on ACTIONS."""
    for op in sorted(M.CHAIN_OPS):
        entity = {"PLANT": "WHEAT", "BUILD": "COW"}.get(op, "COW" if op in M.PLACING_OPS
                                                       else None)
        action = M.compile_op(op, entity)
        assert action[0] in M.ACTIONS, f"{op} compiles to {action}, not an engine action"
    for bad, why in ((("NOT_AN_OP",), "unknown op"),
                     (("PLANT", "COW"), "a crop op naming an animal"),
                     (("BUILD", "WHEAT"), "a build naming a crop")):
        try:
            M.compile_chain(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} was accepted: {bad}")


def test_every_registry_chain_compiles() -> None:
    """Every chain the DP can choose has a worker-day the engine accepts."""
    seen = 0
    for ops in chains.CHAIN_NAMES:
        for entity in ("WHEAT", "COW", None):
            try:
                compiled = M.compile_chain(ops, entity)
            except ValueError:
                continue                      # this entity does not fit this chain
            seen += 1
            for action in compiled:
                assert action[0] in M.ACTIONS, (ops, entity, action)
            break
    assert seen >= len(chains.CHAIN_NAMES), (seen, len(chains.CHAIN_NAMES))


def test_the_dp_vocabulary_agrees_with_the_model() -> None:
    """The DP's resources and market ops are the model's; the one extra op is named."""
    assert tuple(chains.RESOURCE_NAMES) == M.RESOURCES, (
        "the DP's 18 columns drifted from the model's order (the shipped graph "
        "stores them by index, so this is a data-compatibility guard)")
    assert set(chains.MARKET_OPS) == set(M.MARKET_OPS)
    extra = set(chains.WORKER_OPS) - set(M.WORKER_OPS)
    assert extra == {"PLACE_ANIMAL"}, (
        f"the DP's worker vocabulary diverged by more than the known alias: {extra}")
    assert set(chains.ALL_OPS) <= set(M.CHAIN_OPS) | {"PLACE_ANIMAL"}


def test_the_legacy_secretary_vocabulary_maps_onto_the_model() -> None:
    """The WSR's enums name real engine objects — and its two bugs stay pinned.

    Pinning a divergence on purpose: `PRODUCT_ITEMS` is missing every crop and
    `MOVEMENT_ACTIONS` carries `PASS`. Both are fixed by the migration, and the
    guard fails the moment either changes, so the change is deliberate.
    """
    engine_names = set(M.GOODS) | set(M.CROPS) | set(M.ANIMALS)
    for item in wsr.Item:
        assert item.value.upper() in engine_names, (
            f"{item.value!r} is not an engine good/crop/species: the WSR invented a name")
    for action in wsr.MinorActionType:
        assert action.value in M.ACTIONS, f"{action.value} is not an engine action"
    assert {i.value.upper() for i in wsr.PRODUCT_ITEMS} == {"MILK", "WOOL", "EGG"}, (
        "PRODUCT_ITEMS changed: if it now equals the sellable set, delete this pin "
        "and the enum (ARCHITECTURE §5 step 3)")
    assert len(M.SELLABLE) == 9 and len(wsr.PRODUCT_ITEMS) == 3, (
        "the WSR's product set is a strict subset of what the market sells")
    assert {a.value for a in wsr.MOVEMENT_ACTIONS} == set(M.MOVEMENT) | {"PASS"}


def test_no_module_may_grow_its_own_good_names() -> None:
    """The name debt is a ratchet: every module may only shrink."""
    measured: dict[str, int] = {}
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if any(rel.startswith(prefix) or rel == prefix for prefix in EXEMPT):
            continue
        if "__pycache__" in rel:
            continue
        count = _spelled_names(path)
        if count:
            measured[rel] = count
    grew = {f: (n, NAME_DEBT.get(f, 0)) for f, n in measured.items()
            if n > NAME_DEBT.get(f, 0)}
    assert not grew, (
        "these modules spell good names by hand beyond their pin "
        f"(file: measured vs pin): {grew}. Import them from world/model.py "
        "instead, or lower the pin if you removed some.")
    assert set(measured) <= set(NAME_DEBT), (
        f"new modules with hand-spelled names: {sorted(set(measured) - set(NAME_DEBT))}")


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
        print(f"FAIL model: {failures}/{checks} guard(s) failed")
        return 1
    print(f"PASS model: {checks} guards (engine-sourced names, extracted action "
          f"vocabulary, the compile table total over {len(M.CHAIN_OPS)} chain ops "
          f"and {len(chains.CHAIN_NAMES)} registry chains, the DP and WSR "
          f"vocabularies pinned, and a name-debt ratchet over "
          f"{len(NAME_DEBT)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
