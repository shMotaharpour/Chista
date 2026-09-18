"""A tile: the engine's own fields, decoded, for any layer to read.

This is the world's view of a tile, and it is general on purpose — the DP, the WSR
and the rival analysis all read the same object, and none of them needs the others'
conventions to exist. It holds what the engine holds (kaggriculture.py:215-241,
:446-530) and derives only what is arithmetic on those fields (age, product, cap,
whether a dose is still active). It deliberately does NOT decide legality: the
engine's handlers are the arbiter of what an op does on a tile (R003), and a table of
"which ops work here" would be a second implementation of the game.

Every name in it comes from `agent.world.model` — `TileKind`, `Crop`, `Animal`,
`Structure`, `Product` — so a reader never sees a bare string where the world has a
member.

Day-anchored, not normalized. `born_day`, `max_lifespan_step` and
`fertilized_until_day` are absolute engine days, exactly as the engine stores them. A
layer that wants a day-invariant key (the DP's node) does its own normalizing on top
of this; `at_day_start()` gives the one normalization every layer needs — the flags
the end-of-day refresh clears (:875-891).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent.world.model import (Animal, Crop, Product, Structure, TileKind)
from agent.world.rules import ANIMAL_RULES, CROP_RULES

#: The kinds that hold nothing a unit can work on.
HOLDS_NOTHING: frozenset[TileKind] = frozenset(
    {TileKind.EMPTY, TileKind.LOCKED, TileKind.WEED})

#: The two kinds that are a structure, and so can hold an animal.
STRUCTURE_KINDS: frozenset[TileKind] = frozenset({TileKind.COOP, TileKind.PASTURE})

#: `ANIMALS[species]["product"]`, read from the engine's table (kaggriculture.py:19).
ANIMAL_PRODUCT: dict[Animal, Product] = {
    Animal(name): Product(spec["product"]) for name, spec in ANIMAL_RULES.items()}


@dataclass(frozen=True)
class Tile:
    """One tile, decoded. `kind` is a `TileKind`."""

    kind: TileKind
    #: PLANT only: the crop growing here.
    crop: Crop | None = None
    #: A structure holding an animal: the species.
    animal: Animal | None = None
    #: PLANT: `planted_day`; an animal: `placed_day`. Absolute engine day.
    born_day: int | None = None
    watered_today: bool = False
    fed_today: bool = False
    cared_today: bool = False
    #: Consecutive missed days: 2 turns a plant into a WEED, or lets an animal
    #: escape (kaggriculture.py:783, :817). A fresh plant starts at 1 — the
    #: planting day counts as missed (:222) — and a fresh animal at 0 (:236).
    consecutive_unwatered: int = 0
    consecutive_unfed: int = 0
    #: Harvestable units on the tile, capped by the crop's `max_yield` or the
    #: animal's `max_held`.
    yield_units: int = 0
    #: FERTILIZE sets this to `day + 2`: active on `day`, `day+1`, `day+2` (:481).
    fertilized_until_day: int = -1
    #: Animals only: an animal makes one available every night it survives (:831).
    fertilizer_available: bool = False
    #: Animals only: care banked on fed-and-cared days, paid on the next
    #: production day (:826-830).
    pending_care_bonus: int = 0
    #: Plants only: the step decay starts at, `-1` for ongoing crops until their
    #: production cap is reached (:224, :801-802).
    max_lifespan_step: int = -1

    # --- construction ------------------------------------------------------- #

    @classmethod
    def decode(cls, tile: Any) -> "Tile":
        """The engine's `tiles[y][x]` as a `Tile`: `None`, `"LOCKED"`, or a dict."""
        if tile is None:
            return cls(TileKind.EMPTY)
        if tile == TileKind.LOCKED:
            return cls(TileKind.LOCKED)
        if not isinstance(tile, dict):
            raise TypeError(f"not an engine tile: {tile!r}")
        raw_kind = tile.get("kind")
        if raw_kind is None:
            raise ValueError(f"an engine tile dict always carries 'kind': {tile!r}")
        kind = TileKind(raw_kind)
        animal = Animal(tile["animal"]) if tile.get("animal") else None
        crop = Crop(tile["crop"]) if tile.get("crop") else None
        return cls(
            kind=kind,
            crop=crop,
            animal=animal,
            born_day=tile.get("placed_day" if animal else "planted_day",
                              tile.get("born_day")),
            watered_today=bool(tile.get("watered_today", False)),
            fed_today=bool(tile.get("fed_today", False)),
            cared_today=bool(tile.get("cared_today", False)),
            consecutive_unwatered=int(tile.get("consecutive_unwatered", 0)),
            consecutive_unfed=int(tile.get("consecutive_unfed", 0)),
            yield_units=int(tile.get("yield_units", 0)),
            fertilized_until_day=int(tile.get("fertilized_until_day", -1)),
            fertilizer_available=bool(tile.get("fertilizer_available", False)),
            pending_care_bonus=int(tile.get("pending_care_bonus", 0)),
            max_lifespan_step=int(tile.get("max_lifespan_step", -1)),
        )

    def at_day_start(self) -> "Tile":
        """The same tile after an end-of-day refresh: the daily flags are cleared.

        The engine clears `watered_today` (:782), and `fed_today` / `cared_today`
        (:832-833), at the night between two days. Everything else is carried.
        """
        return replace(self, watered_today=False, fed_today=False, cared_today=False)

    # --- what it is --------------------------------------------------------- #

    @property
    def is_empty(self) -> bool:
        return self.kind is TileKind.EMPTY

    @property
    def is_locked(self) -> bool:
        return self.kind is TileKind.LOCKED

    @property
    def is_plant(self) -> bool:
        return self.kind is TileKind.PLANT

    @property
    def is_weed(self) -> bool:
        return self.kind is TileKind.WEED

    @property
    def holds_nothing(self) -> bool:
        return self.kind in HOLDS_NOTHING

    @property
    def structure(self) -> Structure | None:
        """The structure on the tile: its own kind, or where its animal lives."""
        return Structure(self.kind) if self.kind in STRUCTURE_KINDS else None

    @property
    def holds_animal(self) -> bool:
        return self.animal is not None

    @property
    def is_empty_structure(self) -> bool:
        """A coop or pasture with no animal on it — PLACE can still fill it."""
        return self.structure is not None and self.animal is None

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

    def age(self, day: int) -> int | None:
        """Days since the plant was planted or the animal placed; None if neither."""
        return None if self.born_day is None else int(day) - self.born_day

    def fertilised(self, day: int) -> bool:
        """Whether a FERTILIZE dose is still active today (:481)."""
        return self.fertilized_until_day >= int(day)

    def describe(self) -> str:
        """One line, for a message or a report."""
        bits = [str(self.kind)]
        if self.crop:
            bits.append(str(self.crop))
        if self.animal:
            bits.append(str(self.animal))
        if self.born_day is not None:
            bits.append(f"born={self.born_day}")
        if self.yield_units:
            bits.append(f"yield={self.yield_units}")
        if self.fertilized_until_day >= 0:
            bits.append(f"fert_until={self.fertilized_until_day}")
        if self.consecutive_unwatered:
            bits.append(f"dry={self.consecutive_unwatered}")
        if self.consecutive_unfed:
            bits.append(f"unfed={self.consecutive_unfed}")
        if self.pending_care_bonus:
            bits.append(f"care={self.pending_care_bonus}")
        return " ".join(bits)
