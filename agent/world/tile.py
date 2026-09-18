"""A tile, in our own terms: the modelled day-start state, for any layer to read.

This is the world's view of a tile, and it is general on purpose — the DP, the WSR and
the rival analysis all read the same object. The fields are the ones we argued for the
DP's `TileState`, because they are how we see a tile, not how the engine stores it:

    crops:    (crop, age, consec, fert_left, yield_units)
    animals:  (animal, age, unfed, care_bank, yield_units)
    kinds:    NONE | WEED | PLANT | ANIMAL | EMPTY_COOP | EMPTY_PASTURE

Two engine fields are deliberately NOT here. `planted_day` / `placed_day` are replaced
by `age`, the day-invariant lifecycle day (the conventions are below, and they are
ours). `max_lifespan_step` is dropped: the engine stamps it to say when a plant starts
dying, and that day is derivable from the crop's own table (`crop_weed_age`), so
keeping the stamp would only add a second way to say the same thing. `TURNS_PER_DAY`
is not needed either, which is why nothing here depends on the run's configuration.

`LOCKED` is the one kind the engine stores that our model does not argue about: it is
a farm-level fact (which quadrant is bought), not a state a unit works on. It is kept
so a reader can say what it sees — the rival's farm, the WSR's map.

Day-start, not mid-day. The engine's `watered_today`, `fed_today`, `cared_today` and
`fertilizer_available` are intra-day flags; what a day-start reader needs from them is
`consec`, `unfed` and `fert_left`, and a layer that needs the live flags reads the
engine's own tile dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.world.model import Animal, Crop, Product, Structure, TileKind
from agent.world.rules import ANIMAL_RULES, CROP_RULES

#: The kinds that hold nothing a unit can work on.
HOLDS_NOTHING: frozenset[TileKind] = frozenset(
    {TileKind.NONE, TileKind.LOCKED, TileKind.WEED})

#: The two kinds that are an empty structure: what can be placed in each is what makes
#: them different kinds at all (a COOP takes a GOOSE, a PASTURE a COW or a SHEEP).
EMPTY_STRUCTURE: dict[Structure, TileKind] = {Structure.COOP: TileKind.EMPTY_COOP,
                                              Structure.PASTURE: TileKind.EMPTY_PASTURE}
STRUCTURE_OF_KIND: dict[TileKind, Structure] = {v: k for k, v in EMPTY_STRUCTURE.items()}

#: `ANIMALS[species]["product"]`, read from the engine's table (kaggriculture.py:19).
ANIMAL_PRODUCT: dict[Animal, Product] = {
    Animal(name): Product(spec["product"]) for name, spec in ANIMAL_RULES.items()}

#: The engine's limit on consecutive missed days, for a plant and for an animal: the
#: second one turns the plant into a WEED or lets the animal escape
#: (kaggriculture.py:783, :817). A fresh plant is born at 1 (the planting day counts,
#: :222) and a fresh animal at 0 (:236).
DRY_LIMIT = 2

#: A FERTILIZE dose covers its own day plus two more (:481), so `fert_left` runs 0..2.
FERT_DAYS = 3


# --- our age conventions ----------------------------------------------------- #

def crop_age_origin(crop: Crop | str) -> int:
    """The crop day whose age is 0 (Hossein's convention, 2026-09-14).

    one-shot: the START OF THE GOLDEN WINDOW, `(max_yield_day + 1) // 2` — the first
    day watering is worth anything (:439-443).
    ongoing:  `max_yield_day`, the day of its first scheduled production (:789-802).
    """
    spec = CROP_RULES[crop]
    if spec["ongoing"]:
        return int(spec["max_yield_day"])
    return (int(spec["max_yield_day"]) + 1) // 2


def crop_last_day(crop: Crop | str) -> int:
    """The last day on which the crop is still a PLANT, counted from its planting.

    one-shot: the harvest deadline itself; the night after it the tile is a WEED
    (:224). ongoing: its last production day, `max_yield_day + (max_yield - 1) *
    interval` (:789-802).
    """
    spec = CROP_RULES[crop]
    if spec["ongoing"]:
        return (int(spec["max_yield_day"])
                + (int(spec["max_yield"]) - 1) * max(1, int(spec["interval"])))
    return int(spec["max_yield_day"])


def crop_weed_age(crop: Crop | str) -> int:
    """The age at which the tile decodes as WEED, `crop_last_day + 1` in age terms."""
    return crop_last_day(crop) - crop_age_origin(crop) + 1


def animal_cycle_age(animal: Animal | str, day: int, placed_day: int) -> int:
    """An animal's age: growing up (negative), or the production phase (0..interval-1).

    The placement day is intra-day, so the negative range is `1 - first_yield_day ..
    -1`; from `first_yield_day` on, the age is the day's position in the production
    cycle, which wraps (:822-823).
    """
    spec = ANIMAL_RULES[animal]
    age = int(day) - int(placed_day) - int(spec["first_yield_day"])
    if age >= 0:
        age %= max(1, int(spec["interval"]))
    return age


@dataclass(frozen=True)
class Tile:
    """One tile at a day start, in our terms. `kind` is a `TileKind`."""

    kind: TileKind
    #: PLANT only: the crop growing here.
    crop: Crop | None = None
    #: ANIMAL only: the species on the structure.
    animal: Animal | None = None
    #: The structure on the tile: COOP or PASTURE for an animal or an empty one.
    structure: Structure | None = None
    #: The lifecycle day, our convention: `crop_age_origin` / the production cycle.
    age: int = 0
    #: Plants: 0 | 1 — was it watered on the day before this one.
    consec: int = 0
    #: Animals: 0 | 1 — was it fed on the day before this one.
    unfed: int = 0
    #: Plants: 0..2 — days a FERTILIZE dose still covers, this one included.
    fert_left: int = 0
    #: Animals: care banked on fed-and-cared days, paid on the next production.
    care_bank: int = 0
    #: Harvestable units on the tile, capped by the crop's `max_yield` or the
    #: animal's `max_held`.
    yield_units: int = 0

    # --- construction ------------------------------------------------------- #

    @classmethod
    def decode(cls, tile: Any, day: int) -> "Tile":
        """The engine's `tiles[y][x]` at a day start, as our `Tile`.

        `day` is the day the tile is being read at; it is what turns the engine's
        absolute `planted_day` / `placed_day` into `age`.
        """
        if tile is None:
            return cls(TileKind.NONE)
        if tile == "LOCKED":
            return cls(TileKind.LOCKED)
        if not isinstance(tile, dict):
            raise ValueError(f"unsupported tile {tile!r}")
        kind = tile.get("kind")
        if kind == "WEED":
            return cls(TileKind.WEED)

        if kind == "PLANT":
            crop = Crop(tile["crop"])
            spec = CROP_RULES[crop]
            consec = int(tile.get("consecutive_unwatered", 0))
            age = int(day) - (int(tile.get("planted_day", day)) + crop_age_origin(crop))
            # The engine's own destroy paths are the answer here, not a re-derivation:
            # two consecutive dry nights, or a plant past its last day, is a WEED
            # (:783-784, :224, :801-802).
            if consec >= DRY_LIMIT or age >= crop_weed_age(crop):
                return cls(TileKind.WEED)
            return cls(kind=TileKind.PLANT, crop=crop, age=age, consec=consec,
                       fert_left=max(0, int(tile.get("fertilized_until_day", -1))
                                     - int(day) + 1),
                       yield_units=int(tile.get("yield_units", 0)))

        if kind in ("COOP", "PASTURE"):
            structure = Structure(kind)
            animal = tile.get("animal")
            if animal is None:
                return cls(EMPTY_STRUCTURE[structure], structure=structure)
            species = Animal(animal)
            spec = ANIMAL_RULES[species]
            unfed = int(tile.get("consecutive_unfed", 0))
            if unfed >= DRY_LIMIT:      # the animal escaped; the structure stays (:817)
                return cls(EMPTY_STRUCTURE[structure], structure=structure)
            # `care_bank` is capped at `max_held` on purpose (Hossein, probed
            # 2026-09-14): the engine can bank more, but the first production consumes
            # min(max_held, yield + 1 + bank), so any larger bank has the same future.
            bank = min(int(tile.get("pending_care_bonus", 0)), int(spec["max_held"]))
            return cls(kind=TileKind.ANIMAL, animal=species, structure=structure,
                       age=animal_cycle_age(species, day, int(tile.get("placed_day", day))),
                       unfed=unfed, care_bank=bank,
                       yield_units=int(tile.get("yield_units", 0)))

        raise ValueError(f"unsupported tile kind {kind!r}")

    # --- what it is --------------------------------------------------------- #

    @property
    def is_none(self) -> bool:
        return self.kind is TileKind.NONE

    @property
    def is_locked(self) -> bool:
        return self.kind is TileKind.LOCKED

    @property
    def is_weed(self) -> bool:
        return self.kind is TileKind.WEED

    @property
    def is_plant(self) -> bool:
        return self.kind is TileKind.PLANT

    @property
    def is_animal(self) -> bool:
        return self.kind is TileKind.ANIMAL

    @property
    def is_empty_structure(self) -> bool:
        """A coop or pasture with no animal: PLACE can still fill it (:384-392)."""
        return self.kind in EMPTY_STRUCTURE.values()

    @property
    def holds_nothing(self) -> bool:
        return self.kind in HOLDS_NOTHING

    # --- what it produces --------------------------------------------------- #

    @property
    def product(self) -> Product | None:
        """What a HARVEST here yields: the crop, or the animal's product (:466-472)."""
        if self.is_plant:
            return Product(self.crop) if self.crop is not None else None
        if self.animal is not None:
            return ANIMAL_PRODUCT[self.animal]
        return None

    @property
    def yield_cap(self) -> int:
        """The cap on `yield_units`: the crop's `max_yield`, the animal's `max_held`."""
        if self.is_plant and self.crop is not None:
            return int(CROP_RULES[self.crop]["max_yield"])
        if self.animal is not None:
            return int(ANIMAL_RULES[self.animal]["max_held"])
        return 0

    @property
    def fertilised(self) -> bool:
        """Whether a FERTILIZE dose still covers today (:481)."""
        return self.fert_left > 0

    def describe(self) -> str:
        """One line, for a message or a report."""
        if self.kind is TileKind.NONE:
            return "NONE"
        if self.kind is TileKind.LOCKED:
            return "LOCKED"
        if self.kind is TileKind.WEED:
            return "WEED"
        if self.is_empty_structure:
            return f"EMPTY {self.structure}"
        if self.is_animal:
            return (f"{self.animal} age={self.age} unfed={self.unfed} "
                    f"bank={self.care_bank} y={self.yield_units}")
        return (f"{self.crop} age={self.age} consec={self.consec} "
                f"fert_left={self.fert_left} y={self.yield_units}")
