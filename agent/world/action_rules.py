"""What each action needs and what it does — the engine's own checks.

Read from `_apply_unit_action` (kaggriculture.py:312-530), `_parse_order` (:631-649)
and `_commit_unit` (:652-687). The point of writing it down is that a unit action
and a market order are executed by different code: a unit action is applied to the
tile the unit stands on, a market order is applied to the market, and neither can do
the other's job. An action whose precondition does not hold is a **silent no-op** —
the engine returns without a word (:313).

(`offline/actions.py` is a different file: the action-shape validator the pool
guard uses. This one is the reference.)
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.world.model import Animal, Crop, Product


@dataclass(frozen=True)
class ActionRule:
    """One action: its preconditions and its effects, both in the engine's words."""

    op: str
    #: What must hold before it does anything. `()` means "nothing beyond standing".
    needs: tuple[str, ...]
    #: What changes when it runs.
    does: tuple[str, ...]
    #: kaggriculture.py lines
    cite: str


#: The unit actions, kaggriculture.py:312-530.
UNIT_ACTIONS: dict[str, ActionRule] = {
    "NORTH": ActionRule("NORTH", ("in bounds after the move",),
                        ("unit moves one tile; LOCKED tiles are passable",), "323-332"),
    "SOUTH": ActionRule("SOUTH", ("in bounds after the move",),
                        ("unit moves one tile; LOCKED tiles are passable",), "323-332"),
    "EAST": ActionRule("EAST", ("in bounds after the move",),
                       ("unit moves one tile; LOCKED tiles are passable",), "323-332"),
    "WEST": ActionRule("WEST", ("in bounds after the move",),
                       ("unit moves one tile; LOCKED tiles are passable",), "323-332"),
    "PASS": ActionRule("PASS", (), ("nothing",), "334-335"),

    "DROP": ActionRule("DROP", ("unit stands on a shed-access tile",),
                       ("every item in the unit's inventory moves to the shed, up to"
                        " SHED_CAPACITY; the overflow is lost",), "343-356"),
    "PICKUP": ActionRule("PICKUP", ("unit stands on a shed-access tile",
                                    "the named item is in the shed",
                                    "n >= 1"),
                         ("min(n, shed) of the item moves to the unit's inventory;"
                          " seeds are NOT in the shed and cannot be picked up",),
                         "358-375"),
    "PLACE": ActionRule("PLACE", ("the unit holds the animal, and stands on a matching"
                                  " structure that holds none — or stands on a"
                                  " shed-access tile and holds the item",),
                        ("the animal moves from the inventory onto the structure, or"
                         " min(n, held) of the item moves to the shed",), "377-410"),

    "PLANT": ActionRule("PLANT", ("the tile is empty and unlocked",
                                  "the seed is held in private['seeds']"),
                        ("one seed is spent and the tile becomes a plant, born"
                         " unwatered with consecutive_unwatered = 1",), "417-429"),
    "WATER": ActionRule("WATER", ("the tile is a plant",
                                  "the plant has not been watered today"),
                        ("watered_today = True; inside the yield window a one-time"
                         " crop gains 1 (2 while fertilised), capped at max_yield",),
                        "431-444"),
    "HARVEST": ActionRule("HARVEST", ("the tile is a plant or an animal",
                                      "yield_units > 0",
                                      "for a crop: day - planted_day >= first_yield_day"),
                          ("the tile's yield_units move to the inventory and reset to"
                           " 0; a one-time crop's tile becomes empty",), "446-473"),
    "FERTILIZE": ActionRule("FERTILIZE", ("the tile is a plant",
                                          "the unit holds 1 FERTILIZER"),
                            ("1 FERTILIZER is spent; fertilized_until_day = day + 2",),
                            "475-482"),
    "DIG": ActionRule("DIG", ("the tile is not empty and not LOCKED",
                              "the tile holds no animal"),
                      ("the tile becomes empty (a plant, a weed, or an empty"
                       " structure)",), "484-491"),
    "BUILD_COOP": ActionRule("BUILD_COOP", ("the tile is empty and unlocked",),
                             ("the tile becomes an empty coop",), "493-497"),
    "BUILD_PASTURE": ActionRule("BUILD_PASTURE", ("the tile is empty and unlocked",),
                                ("the tile becomes an empty pasture",), "499-503"),
    "FEED": ActionRule("FEED", ("the tile holds an animal",
                                "the animal has not been fed today",
                                "the unit holds 1 WHEAT"),
                       ("1 WHEAT is spent; fed_today = True",), "505-513"),
    "COLLECT_FERTILIZER": ActionRule("COLLECT_FERTILIZER",
                                     ("the tile holds an animal",
                                      "fertilizer_available is True"),
                                     ("fertilizer_available = False; 1 FERTILIZER"
                                      " moves to the inventory",), "515-522"),
    "CARE": ActionRule("CARE", ("the tile holds an animal",
                                "the animal has not been cared for today"),
                       ("cared_today = True; a fed-and-cared day banks a bonus for"
                        " the next production",), "524-530"),
}

#: The market's orders, kaggriculture.py:631-687.
MARKET_ORDERS: dict[str, ActionRule] = {
    "SELL": ActionRule("SELL", ("the item is a product",
                                "the shed holds the item",
                                "n >= 1"),
                       ("1 unit per commit leaves the shed, money += the price quoted"
                        " at the pre-sale inventory; a sale at $1 does not add supply",),
                       "653-661"),
    "BUY_PRODUCT": ActionRule("BUY_PRODUCT", ("the item is WHEAT or FERTILIZER",
                                              "money >= the quoted price",
                                              "the shed has room",
                                              "n >= 1"),
                              ("1 unit lands in the shed, money -= the price quoted at"
                               " the post-buy inventory, market inventory -= 1",),
                              "662-672"),
    "BUY_SEED": ActionRule("BUY_SEED", ("the item is a crop",
                                        "money >= the crop's seed price",
                                        "n >= 1"),
                           ("money -= the crop's seed price; one seed is added to"
                            " private['seeds']",), "673-678"),
    "BUY_ANIMAL": ActionRule("BUY_ANIMAL", ("the item is an animal",
                                            "money >= the animal's cost",
                                            "the shed has room",
                                            "n >= 1"),
                             ("1 animal lands in the shed; money -= its cost",),
                             "679-686"),
    "HIRE": ActionRule("HIRE", ("money >= the hire cost for the n-th hire today",),
                       ("money -= fib(n); a hand spawns on the least-occupied"
                        " shed-access tile and hires_today += 1",), "576-577, 702-709"),
    "BUY_LAND": ActionRule("BUY_LAND", ("money >= the next quadrant's price",
                                        "fewer than three quadrants bought so far"),
                           ("money -= the price; the next quadrant in LAND_ORDER is"
                            " unlocked",), "579-581, 712-725"),
}

#: What an op's arguments are: does it name an item, does it take a count. Read from
#: the engine's handlers — `_apply_unit_action` (:312-530) for the unit ops, where
#: PICKUP and PLACE take `[item, n]`, PLANT takes a crop, and the rest take nothing;
#: `_parse_order` (:631-649) for the market's, where SELL and the three BUY_* need
#: `[item, n]` and HIRE and BUY_LAND need neither. An op not listed takes neither.
SIGNATURE: dict[str, tuple[bool, bool]] = {
    "PICKUP": (True, True), "PLACE": (True, True), "PLANT": (True, False),
    "SELL": (True, True), "BUY_SEED": (True, True), "BUY_PRODUCT": (True, True),
    "BUY_ANIMAL": (True, True),
}

#: The item vocabulary each op accepts, from the same handlers: PLANT a crop, PLACE an
#: animal, PICKUP anything the shed holds (products and animals), SELL a product,
#: BUY_SEED a crop, BUY_PRODUCT a product (only WHEAT and FERTILIZER survive the
#: engine's own check, :598), BUY_ANIMAL an animal.
ITEM_OF: dict[str, tuple[type, ...]] = {
    "PICKUP": (Product, Animal), "PLACE": (Animal,), "PLANT": (Crop,),
    "SELL": (Product,), "BUY_SEED": (Crop,), "BUY_PRODUCT": (Product,),
    "BUY_ANIMAL": (Animal,),
}

#: The ops that only work from one of the four shed-access tiles (:343-410). They use
#: the tile as a standing position and never change it, which is why they work even
#: on a LOCKED tile.
SHED_OPS: tuple[str, ...] = ("PICKUP", "DROP", "PLACE")

#: An op that cannot run without something in the unit's inventory, and what it is.
#: The rest of the inventory's contents come from HARVEST and COLLECT_FERTILIZER.
CARRIES: dict[str, str] = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}
#: ... and `PLACE` carries the animal itself, named by the action's argument.

#: The ops that put something in the inventory, and where it comes from.
YIELDS: dict[str, str] = {"HARVEST": "the tile's yield_units",
                          "COLLECT_FERTILIZER": "FERTILIZER",
                          "PICKUP": "the shed"}
