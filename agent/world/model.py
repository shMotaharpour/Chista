"""The names, as the engine spells them — declared as enums, for a reader.

Source of truth: `kaggle_environments/envs/kaggriculture/kaggriculture.py` (the
interpreter) and the rules the environment ships in its own `README.md`. Every member
below is copied from one of them, and the citation is in the comment; nothing here is
derived from our own code, and nothing here imports it.

The engine's own vocabulary, in its own words:

- a **product** is one of the 9 the market trades (`Product`).
- a **crop** is one of the 5 you plant (`Crop`); its **seed** is bought by the crop's
  name and never enters the shed or a unit's inventory (`PLANT` consumes it from
  `private["seeds"]` directly).
- an **animal** is one of the 3 you place (`Animal`); it lives in a **structure**
  (`Structure`), either a coop or a pasture, and makes one product.
- a **tile** is empty, locked, a plant, a weed, or a structure that may hold an animal
  (`TileKind`).
- a **unit** is the main farmer or a hired hand. Both act every turn, both carry an
  inventory, and both are cleared at the end of the day.
- the **market** has its own orders (`MarketOrder`), executed by its own code, never by
  a unit.

Each enum is written out rather than generated, so a reader (and a type checker) sees
the members; the tuple constants are then derived from the enums, so there is still
one source for each name.
"""

from __future__ import annotations

from enum import Enum


class Product(str, Enum):
    """The 9 goods the market trades. kaggriculture.py:25"""

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    FERTILIZER = "FERTILIZER"


class Crop(str, Enum):
    """The 5 crops you can plant; the keys of CROPS. kaggriculture.py:11"""

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"


class Animal(str, Enum):
    """The 3 species you can place; the keys of ANIMALS. kaggriculture.py:19"""

    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"


class Structure(str, Enum):
    """What an animal lives in: ANIMALS[.]["structure"]. kaggriculture.py:20-22"""

    COOP = "COOP"
    PASTURE = "PASTURE"


class TileKind(str, Enum):
    """What a tile can be.

    Empty and locked are the engine's `None` and `"LOCKED"`; a plant, a weed, and a
    structure — whose kind is the structure's own name, with an optional animal on it
    (kaggriculture.py:157-158, 215-241, 493-503).
    """

    EMPTY = "EMPTY"
    LOCKED = "LOCKED"
    PLANT = "PLANT"
    WEED = "WEED"
    COOP = "COOP"
    PASTURE = "PASTURE"


class UnitAction(str, Enum):
    """What a unit can be told to do in a turn: the ops `_apply_unit_action` handles.

    kaggriculture.py:312-530 — the four moves, the shed trips, the plant and animal
    ops, and PASS. 18 of them.
    """

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    PASS = "PASS"
    PICKUP = "PICKUP"
    DROP = "DROP"
    PLACE = "PLACE"
    PLANT = "PLANT"
    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    DIG = "DIG"
    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"
    FEED = "FEED"
    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
    CARE = "CARE"


class MarketOrder(str, Enum):
    """What the market's own code handles: `_parse_order`. kaggriculture.py:631-649.

    None of these is a unit action, and no unit action is a market order.
    """

    SELL = "SELL"
    BUY_SEED = "BUY_SEED"
    BUY_PRODUCT = "BUY_PRODUCT"
    BUY_ANIMAL = "BUY_ANIMAL"
    HIRE = "HIRE"
    BUY_LAND = "BUY_LAND"


class Move(str, Enum):
    """The four moves. Their (dx, dy) is in `MOVES`; y grows downward (:88-93)."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class Column(str, Enum):
    """The 18 things a farm acquires or sells, and so the columns a plan is priced over.

    One unit of labour (bought with HIRE), 5 seeds, 3 animals, 9 products. Wheat and
    fertiliser appear once, as products — they are the two the market also buys back,
    which is why they are the only columns that can sit on both sides of a plan's
    ledger (:598).
    """

    LABOR = "LABOR"
    SEED_WHEAT = "SEED_WHEAT"
    SEED_CARROT = "SEED_CARROT"
    SEED_TOMATO = "SEED_TOMATO"
    SEED_STRAWBERRY = "SEED_STRAWBERRY"
    SEED_MELON = "SEED_MELON"
    ANIMAL_GOOSE = "ANIMAL_GOOSE"
    ANIMAL_COW = "ANIMAL_COW"
    ANIMAL_SHEEP = "ANIMAL_SHEEP"
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    FERTILIZER = "FERTILIZER"


# --- the same names as tuples, derived from the enums ------------------------ #

PRODUCTS: tuple[str, ...] = tuple(m.value for m in Product)
CROPS: tuple[str, ...] = tuple(m.value for m in Crop)
ANIMALS: tuple[str, ...] = tuple(m.value for m in Animal)
STRUCTURES: tuple[str, ...] = tuple(m.value for m in Structure)
TILE_KINDS: tuple[str, ...] = tuple(m.value for m in TileKind)
UNIT_ACTIONS: tuple[str, ...] = tuple(m.value for m in UnitAction)
MARKET_ORDERS: tuple[str, ...] = tuple(m.value for m in MarketOrder)
COLUMNS: tuple[str, ...] = tuple(m.value for m in Column)

#: ANIMALS[.]["product"] — one product per species (kaggriculture.py:20-22).
ANIMAL_PRODUCTS: tuple[str, ...] = ("EGG", "MILK", "WOOL")

#: The two things the farm eats that the market also sells: the engine's BUY_PRODUCT
#: accepts these two and nothing else (kaggriculture.py:598).
DUAL: tuple[str, ...] = ("WHEAT", "FERTILIZER")

#: private["shed"] is keyed by PRODUCTS + ANIMALS (kaggriculture.py:171).
SHED_ITEMS: tuple[str, ...] = PRODUCTS + ANIMALS

#: A unit's inventory holds shed items. Seeds never appear here: they live in
#: private["seeds"], keyed by crop, and PLANT consumes them directly
#: (kaggriculture.py:367-368, 425-427).
INVENTORY_ITEMS: tuple[str, ...] = SHED_ITEMS

#: The four moves and their (dx, dy); y grows downward (kaggriculture.py:88-93).
MOVES: dict[str, tuple[int, int]] = {
    "NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
