"""The names, as the engine spells them.

Source of truth: `kaggle_environments/envs/kaggriculture/kaggriculture.py` (the
interpreter) and the rules the environment ships in its own `README.md`. Every set
below is copied from one of them, and the citation is in the comment; nothing here
is derived from our own code, and nothing here imports it.

The engine's own vocabulary, in its own words:

- a **product** is one of the 9 the market trades (`PRODUCTS`).
- a **crop** is one of the 5 you plant (`CROPS`); its **seed** is bought by the
  crop's name and never enters the shed or a unit's inventory (`PLANT` consumes it
  from `private["seeds"]` directly).
- an **animal** is one of the 3 you place; it lives in a **structure**, either a
  coop or a pasture, and makes one product.
- a **unit** is the main farmer or a hired hand. Both act every turn, both carry an
  inventory, and both are cleared at the end of the day.
- the **market** has its own orders, executed by its own code, never by a unit.
"""

from __future__ import annotations

from enum import Enum

# --- what the engine has ----------------------------------------------------- #

#: kaggriculture.py:25
PRODUCTS: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "EGG", "MILK", "WOOL", "FERTILIZER")

#: The keys of CROPS, kaggriculture.py:11
CROPS: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

#: The keys of ANIMALS, kaggriculture.py:19
ANIMALS: tuple[str, ...] = ("GOOSE", "COW", "SHEEP")

#: ANIMALS[.]["structure"], kaggriculture.py:20-22
STRUCTURES: tuple[str, ...] = ("COOP", "PASTURE")

#: ANIMALS[.]["product"]
ANIMAL_PRODUCTS: tuple[str, ...] = ("EGG", "MILK", "WOOL")

#: The two things the farm eats that the market also sells: the engine's
#: BUY_PRODUCT accepts these two and nothing else (kaggriculture.py:598).
DUAL: tuple[str, ...] = ("WHEAT", "FERTILIZER")

#: private["shed"] is keyed by PRODUCTS + ANIMALS (kaggriculture.py:171).
SHED_ITEMS: tuple[str, ...] = PRODUCTS + ANIMALS

#: A unit's inventory holds shed items. Seeds never appear here: they live in
#: private["seeds"], keyed by crop, and PLANT consumes them directly
#: (kaggriculture.py:367-368, 425-427).
INVENTORY_ITEMS: tuple[str, ...] = SHED_ITEMS

#: The 18 things a farm acquires or sells, and so the columns a plan can be priced
#: over: 1 unit of labour (bought with HIRE), 5 seeds, 3 animals, 9 products.
#: Wheat and fertiliser appear once, as products — they are the two the market also
#: buys back, which is why they are the only columns that can be on both sides of a
#: plan's ledger.
COLUMNS: tuple[str, ...] = (("LABOR",)
                            + tuple(f"SEED_{crop}" for crop in CROPS)
                            + tuple(f"ANIMAL_{animal}" for animal in ANIMALS)
                            + PRODUCTS)

#: A tile is one of: empty (None), LOCKED, a plant, a weed, or a structure — and a
#: structure may hold an animal (kaggriculture.py:157-158, 215-241, 496-503).
TILE_STATES: tuple[str, ...] = ("EMPTY", "LOCKED", "PLANT", "WEED", "COOP",
                                "PASTURE")

#: What a unit can be told to do in a turn: the ops `_apply_unit_action` handles
#: (kaggriculture.py:312-530). 18 of them.
UNIT_ACTIONS: tuple[str, ...] = (
    "NORTH", "SOUTH", "EAST", "WEST",              # FARMER_MOVES
    "PASS",
    "PICKUP", "DROP", "PLACE",                     # shed, from an access tile
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE",
    "FEED", "COLLECT_FERTILIZER", "CARE",          # animal care
)

#: What the market's own code handles: `_parse_order` (kaggriculture.py:631-649).
#: None of these is a unit action, and no unit action is a market order.
MARKET_ORDERS: tuple[str, ...] = ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
                                  "HIRE", "BUY_LAND")

#: The four moves, and their (dx, dy); y grows downward (kaggriculture.py:88-93).
MOVES: dict[str, tuple[int, int]] = {
    "NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# --- named views, so a reader can say the word ------------------------------- #

def _names(cls_name: str, values) -> Enum:
    """A `str` enum whose member names and values are the engine's own strings."""
    return Enum(cls_name, {str(v): str(v) for v in values}, type=str, module=__name__)


Product = _names("Product", PRODUCTS)
Crop = _names("Crop", CROPS)
Animal = _names("Animal", ANIMALS)
Structure = _names("Structure", STRUCTURES)
TileState = _names("TileState", TILE_STATES)
UnitAction = _names("UnitAction", UNIT_ACTIONS)
MarketOrder = _names("MarketOrder", MARKET_ORDERS)
Column = _names("Column", COLUMNS)
Move = _names("Move", tuple(MOVES))
