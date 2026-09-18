"""The canonical names: enums, hard-coded, for every layer to read.

`docs/ARCHITECTURE.md` §1. These are the engine's names, pinned as members so an
agent can read a definition instead of calling a function. The values were
extracted from the engine once (`kaggriculture`'s tables, `FARMER_MOVES`, and the
action literals in `_apply_unit_action` / `_commit_unit` / `_parse_order`) and
`tests/test_model.py` re-extracts them on every run, so drift is caught there
rather than computed here.

Two vectors are read over the same 18 columns (`Vector`):

- the **price vector**, the DP's input — `p` for what a produced unit is worth and
  `w` for what a consumed unit costs;
- the **result vector**, the tile's day — `cost` for what it consumed and
  `produce` for what it made.

Wheat and fertiliser are NOT duplicated: they are one column, non-zero on both
sides of the result pair (a harvest makes wheat, FEED eats it; COLLECT_FERTILIZER
makes fertiliser, FERTILIZE eats it). What *is* two numbers is their market price:
the engine quotes a buy at `price(I-1)` and pays a sale at `price(I)` (F033), and
that spread belongs to the market layer, not to this column space.

The division of labour that answers "how does the layer above know": the **vector
is the ledger** (what a day costs and makes, which is what the DP prices), the
**chain's op list is the schedule** (which op must have what in the bag, and in
what order — `CARRIES`/`PRODUCES`), and the **chain id is the bridge** between
them. A same-day produce-then-consume of one good (harvest wheat, then FEED it) is
where the ledger alone cannot decide whether a shed trip is needed: the op ORDER
decides, and the compiler is the reader that knows it.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

# --- what exists ------------------------------------------------------------ #

#: The 9 goods the market trades and the farm sells.
GOODS: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                          "EGG", "MILK", "WOOL", "FERTILIZER")
PRODUCTS: tuple[str, ...] = GOODS
CROPS: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS: tuple[str, ...] = ("GOOSE", "COW", "SHEEP")

#: Wheat and fertiliser are both: the farm buys them and the market sells them.
DUAL: tuple[str, ...] = ("WHEAT", "FERTILIZER")

#: Resources: what the farm buys or consumes. 11 — labour, fertiliser, wheat, the
#: five seeds, the three animals. Crops and animal products are products only.
RESOURCES: tuple[str, ...] = ("LABOR_HOURS", "FERTILIZER", "WHEAT",
                              "SEED_WHEAT", "SEED_CARROT", "SEED_TOMATO",
                              "SEED_STRAWBERRY", "SEED_MELON",
                              "ANIMAL_GOOSE", "ANIMAL_COW", "ANIMAL_SHEEP")

#: The DP's 18 columns = resources + products, in the order the shipped graph
#: stores them (11 + 9 − 2 shared = 18). Both vectors are read over this space.
VECTOR: tuple[str, ...] = ("LABOR_HOURS", "FERTILIZER", "WHEAT",
                           "SEED_WHEAT", "SEED_CARROT", "SEED_TOMATO",
                           "SEED_STRAWBERRY", "SEED_MELON",
                           "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                           "EGG", "MILK", "WOOL",
                           "ANIMAL_GOOSE", "ANIMAL_COW", "ANIMAL_SHEEP")

#: Anything the farm holds or trades: the 9 products plus the 3 species.
ITEMS: tuple[str, ...] = PRODUCTS + ("GOOSE", "COW", "SHEEP")

#: The two vectors, named. `PRICE` is the DP's input, `RESULT` its output.
PRICE_VECTORS: tuple[str, ...] = ("PRODUCE", "INPUT")     # p, w
RESULT_VECTORS: tuple[str, ...] = ("COST", "PRODUCE")     # per_day_cost, per_day_produce

#: What a tile's `kind` can be; a bare tile is None, which no member expresses.
TILE_KINDS: tuple[str, ...] = ("PLANT", "COOP", "PASTURE", "WEED")

#: The engine's action vocabulary: what its handlers act on. Anything else is a
#: silent no-op (F047).
ACTIONS: tuple[str, ...] = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "FEED", "CARE",
    "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE",
    "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND",
)

#: One tile per op.
MOVEMENT: tuple[str, ...] = ("NORTH", "SOUTH", "EAST", "WEST")
MOVE_DELTA: Mapping[str, tuple[int, int]] = {
    "NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

#: The market's own actions, never a worker's. `SELL`, `BUY_LAND` and `HIRE` are
#: the market layer's decisions; the three `BUY_*` are also what a chain names
#: when it needs an input.
MARKET_ACTIONS: tuple[str, ...] = ("SELL", "BUY_SEED", "BUY_PRODUCT",
                                   "BUY_ANIMAL", "HIRE", "BUY_LAND")

#: Worker ops: what a unit spends a turn on. No market op appears here — the
#: contractor and the WSR never touch the market.
WORKER_OPS: tuple[str, ...] = ("PLANT", "WATER", "FERTILIZE", "HARVEST", "DIG",
                               "BUILD", "PLACE", "FEED", "CARE",
                               "COLLECT_FERTILIZER")

#: What a chain may name: worker ops + the market buys it needs + the day pass.
CHAIN_OPS: tuple[str, ...] = WORKER_OPS + ("BUY_SEED", "BUY_PRODUCT",
                                           "BUY_ANIMAL", "NO_ACT")

#: The two directions an upper layer reads a good in.
#:
#: `CARRIES` is what an op eats: a chain naming `FEED` consumes WHEAT, so the
#: compiler must have it in the shed, buy it, and schedule the PICKUP trip.
#: `PRODUCES` is what an op yields: a chain naming `HARVEST` takes the tile's own
#: crop off it (`{crop}` is the chain's entity), so the compiler schedules the DROP
#: that makes it sellable. A good can be in both maps — that is the point of the
#: shared column: the SIDE says whether this tile-day consumes it or makes it.
CARRIES: Mapping[str, str] = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}
PRODUCES: Mapping[str, str] = {"HARVEST": "{crop}", "COLLECT_FERTILIZER": "FERTILIZER"}
COLLECT_ITEM = "FERTILIZER"

#: Ops that carry the entity itself, need a seed, or name what they construct.
PLACING_OPS: tuple[str, ...] = ("PLACE",)
SEED_OPS: tuple[str, ...] = ("PLANT",)
ENTITY_OPS: tuple[str, ...] = ("PLANT", "BUILD", "PLACE")

#: Species -> the structure it lives in.
ANIMAL_STRUCTURE: Mapping[str, str] = {"GOOSE": "COOP", "COW": "PASTURE",
                                       "SHEEP": "PASTURE"}

#: The shed's four access tiles, in the engine's NWSE order. PICKUP and DROP work
#: nowhere else.
SHED_ACCESS_ORDERED: tuple[tuple[int, int], ...] = ((4, 4), (5, 4), (4, 5), (5, 5))
SHED_ACCESS: frozenset[tuple[int, int]] = frozenset(SHED_ACCESS_ORDERED)

# --- named views ------------------------------------------------------------ #

def _names(cls_name: str, values) -> Enum:
    """A `str` enum whose member names and values are the engine's own strings."""
    return Enum(cls_name, {str(v): str(v) for v in values}, type=str, module=__name__)


Good = _names("Good", GOODS)
Crop = _names("Crop", CROPS)
Species = _names("Species", ANIMALS)
Resource = _names("Resource", RESOURCES)
Product = _names("Product", PRODUCTS)
Item = _names("Item", ITEMS)
Vector = _names("Vector", VECTOR)
Price = _names("Price", PRICE_VECTORS)
Result = _names("Result", RESULT_VECTORS)
TileKind = _names("TileKind", TILE_KINDS)
Action = _names("Action", ACTIONS)
WorkerOp = _names("WorkerOp", WORKER_OPS)
MarketAction = _names("MarketAction", MARKET_ACTIONS)
ChainOp = _names("ChainOp", CHAIN_OPS)
Move = _names("Move", MOVEMENT)

# --- the compile table ------------------------------------------------------ #

#: A chain op -> the engine action it means. `{entity}` is filled from the chain's
#: own entity; an op absent from this table means "the op is the action".
COMPILE: Mapping[str, tuple[str, ...]] = {
    "PLANT": ("PLANT", "{entity}"),
    "PLACE": ("PLACE", "{entity}"),
    "BUILD": ("BUILD_{structure}",),
    "NO_ACT": ("PASS",),
}


def compile_op(op: str, entity: str | None = None) -> tuple[str, ...]:
    """One chain op -> the engine action it means. Raises on an unknown op."""
    if op not in CHAIN_OPS:
        raise ValueError(f"{op!r} is not a chain op: {list(CHAIN_OPS)}")
    if op == "BUILD":
        if entity not in ANIMAL_STRUCTURE:
            raise ValueError(f"BUILD {entity!r} is not an animal")
        return (f"BUILD_{ANIMAL_STRUCTURE[entity]}",)
    if op in ("PLANT", "PLACE"):
        known = CROPS if op == "PLANT" else ANIMALS
        if entity not in known:
            raise ValueError(f"{op} {entity!r} is not a {'crop' if op == 'PLANT' else 'species'}")
        return (op, str(entity))
    if op == "NO_ACT":
        return ("PASS",)
    return (str(op),)


def compile_chain(ops: tuple[str, ...], entity: str | None = None
                  ) -> list[tuple[str, ...]]:
    """A chain -> the worker actions of one day, one per turn. The only expansion."""
    if "NO_ACT" in ops and len(ops) > 1:
        raise ValueError(f"NO_ACT is a whole-chain op, got {ops}")
    return [compile_op(op, entity) for op in ops]
