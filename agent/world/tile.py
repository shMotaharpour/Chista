"""A tile, in our own terms: the modelled state at a day start, and inside a day.

This is the world's view of a tile, and it is general on purpose — the DP, the WSR and
the rival analysis all read the same object.

`TileHourZero` is the tile as a day begins (hour 0). Its fields are the ones we argued
for the DP's `TileState`, because they are how we see a tile, not how the engine stores
it:

    crops:    (crop, age, consec, fert_left, yield_units)
    animals:  (animal, age, unfed, care_bank, yield_units)
    kinds:    NONE | LOCKED | WEED | PLANT | ANIMAL | EMPTY_COOP | EMPTY_PASTURE

Two engine fields are deliberately NOT there. `planted_day` / `placed_day` are replaced
by `age`, the day-invariant lifecycle day (the conventions are below, and they are
ours). `max_lifespan_step` is dropped: the engine stamps it to say when a plant starts
dying, and that day is derivable from the crop's own table (`crop_weed_age`), so
keeping the stamp would only add a second way to say the same thing. `TURNS_PER_DAY` is
not needed either, which is why nothing here depends on the run's configuration.

`TileInDay` is the same tile part-way through a day: the hour, plus the facts that only
an hour can change — was it watered today, was a dose given today, has the animal's
fertiliser been taken today, was it fed and cared for today. It inherits the day-start
state, because inside a day nothing about the tile's life changes (age, `consec`,
`unfed`, `care_bank`, the kind itself): those move at the night.

`delta(later, earlier)` reads one hour of a worker's work off a tile: it takes two
`TileInDay` one hour apart and returns the actions consistent with the change. That is
how the rival's public tiles can be read without knowing what the rival sent.

`LOCKED` is the one kind the engine stores that our model does not argue about: it is a
farm-level fact (which quadrant is bought), not a state a unit works on. It is kept so
a reader can say what it sees — the rival's farm, the WSR's map.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent.world.action import WorkerAction
from agent.world.model import Animal, Crop, Product, Structure, TileKind, UnitAction
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


# --- the tile at a day start ------------------------------------------------- #

@dataclass(frozen=True)
class TileHourZero:
    """One tile as a day begins, in our terms. `kind` is a `TileKind`."""

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

    @classmethod
    def decode(cls, tile: Any, day: int) -> "TileHourZero":
        """The engine's `tiles[y][x]` at a day start, as our tile.

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


# --- the tile inside a day ---------------------------------------------------- #

@dataclass(frozen=True)
class TileInDay(TileHourZero):
    """The same tile part-way through a day: the hour, and today's facts.

    Everything that changes at the night (the kind, `age`, `consec`, `unfed`,
    `care_bank`) comes from `TileHourZero` and is read at the day start; what a day can
    change is here.
    """

    #: The hour of the day, 0..`TURNS_PER_DAY - 1` (kaggriculture.py:911).
    hour: int = 0
    #: WATER, once a day (:434-436).
    watered_today: bool = False
    #: FEED, once a day (:508-509).
    fed_today: bool = False
    #: CARE, once a day (:527-528).
    cared_today: bool = False
    #: FERTILIZE given today. The engine stores only `fertilized_until_day` (:481), so
    #: "today" is read off `fert_left`: a dose given today covers today and two more
    #: days, which is the only way `fert_left` reaches `FERT_DAYS`.
    fertilized_today: bool = False
    #: COLLECT_FERTILIZER taken today. An animal makes one available every night it
    #: survives (:831), so a day start with none available means it was taken.
    fertilizer_collected_today: bool = False

    @classmethod
    def decode_at(cls, tile: Any, day: int, hour: int) -> "TileInDay":
        """The engine's `tiles[y][x]` at any hour, as our tile.

        Named differently from `TileHourZero.decode` on purpose: an hour is not a
        day-start state, and a caller should have to say which one it wants.
        """
        start = TileHourZero.decode(tile, day)
        raw = tile if isinstance(tile, dict) else {}
        return cls(
            **{f: getattr(start, f) for f in TileHourZero.__dataclass_fields__},
            hour=int(hour),
            watered_today=bool(raw.get("watered_today", False)),
            fed_today=bool(raw.get("fed_today", False)),
            cared_today=bool(raw.get("cared_today", False)),
            fertilized_today=start.fert_left == FERT_DAYS,
            fertilizer_collected_today=(start.is_animal
                                        and not raw.get("fertilizer_available", True)),
        )

    def describe(self) -> str:
        """One line: the day-start state, plus what has happened today."""
        today = [name for name, on in (("watered", self.watered_today),
                                       ("fertilized", self.fertilized_today),
                                       ("fed", self.fed_today),
                                       ("cared", self.cared_today),
                                       ("fert_taken", self.fertilizer_collected_today))
                 if on]
        tail = f" h{self.hour}" + (" " + ",".join(today) if today else "")
        return super().describe() + tail


# --- reading one hour of work off a tile -------------------------------------- #

#: The actions a tile can show, and what each change means.
def delta(later: TileInDay, earlier: TileInDay) -> tuple[WorkerAction, ...]:
    """The actions consistent with one hour of change on one tile.

    `later` and `earlier` are the same tile one hour apart, `later.hour ==
    earlier.hour + 1`. While the tile keeps its kind, nothing that moves at the night
    may have moved (the age, `consec`, `unfed`, `care_bank`), and `yield_units` may
    only have grown on a watering. The two crops differ here: a ONE-SHOT crop watered
    inside its golden window is paid at once (:439-443), while an ONGOING crop's
    production - and an animal's - lands at the night, one day after the watering that
    earned it (:789-802, :822-831). When the kind DID change, an action must explain it, or this raises rather
    than quietly reporting nothing.

    A one-shot crop that is now empty was HARVESTed if it had yield on it and DUG if it
    had none (the owner's rule; a plan does not dig a crop it could harvest). The
    remaining cases each name exactly one action, so the return value is a tuple only
    because a single hour can hold two units' work on one tile - one watered it while
    the other fertilised it.
    """
    if later.hour != earlier.hour + 1:
        raise ValueError(f"not one hour apart: {earlier.hour} -> {later.hour}")
    same_kind = later.kind is earlier.kind
    if same_kind:
        for field in ("age", "consec", "unfed", "care_bank"):
            if getattr(later, field) != getattr(earlier, field):
                raise ValueError(f"{field} moved between hour {earlier.hour} and "
                                 f"{later.hour} while the tile kept its kind: the two "
                                 "are not one hour apart")
        if (later.yield_units > earlier.yield_units
                and not (later.watered_today and not earlier.watered_today)):
            raise ValueError("yield_units grew inside a day with no watering: an "
                             "ongoing crop and an animal produce at the night, so the "
                             "two are not one hour apart")

    acts: list[WorkerAction] = []
    if later.watered_today and not earlier.watered_today:
        acts.append(WorkerAction(UnitAction.WATER))
    if later.fertilized_today and not earlier.fertilized_today:
        acts.append(WorkerAction(UnitAction.FERTILIZE))
    if later.fertilizer_collected_today and not earlier.fertilizer_collected_today:
        acts.append(WorkerAction(UnitAction.COLLECT_FERTILIZER))
    if later.fed_today and not earlier.fed_today:
        acts.append(WorkerAction(UnitAction.FEED))
    if later.cared_today and not earlier.cared_today:
        acts.append(WorkerAction(UnitAction.CARE))

    if not same_kind:
        explained = _kind_change_actions(later, earlier)
        if not explained:
            raise ValueError(f"the tile changed from {earlier.kind} to {later.kind} "
                             f"between hour {earlier.hour} and {later.hour}, and no "
                             "action explains it")
        acts.extend(explained)
    elif later.yield_units < earlier.yield_units:
        acts.append(WorkerAction(UnitAction.HARVEST))
    return tuple(acts)


def _kind_change_actions(later: TileInDay, earlier: TileInDay) -> tuple[WorkerAction, ...]:
    """What the kind change alone allows: the constructive and destructive ops.

    DIG empties a plant, a weed or an empty structure (:484-491); HARVEST does it to a
    one-shot crop that has something on it (:464-468); PLANT, BUILD_* and PLACE each
    make one kind out of another (:417-429, :493-503, :384-392).
    """
    if earlier.is_none and later.is_plant:
        return (WorkerAction(UnitAction.PLANT, later.crop),)
    if earlier.is_none and later.is_empty_structure:
        return (WorkerAction(UnitAction.BUILD_COOP if later.structure is Structure.COOP
                       else UnitAction.BUILD_PASTURE),)
    if earlier.is_empty_structure and later.is_animal:
        return (WorkerAction(UnitAction.PLACE, later.animal),)
    if later.is_none and earlier.is_weed:
        return (WorkerAction(UnitAction.DIG),)
    if later.is_none and earlier.is_plant:
        # A one-shot crop: it had something on it, so it was HARVESTed; it had nothing,
        # so it was DUG. (The engine would let a DIG remove a plant that has yield too
        # (:484-491), but a plan does not dig a crop it could harvest - the owner's
        # rule, and the only reading a tile can support.)
        return ((WorkerAction(UnitAction.HARVEST),) if earlier.yield_units > 0
                else (WorkerAction(UnitAction.DIG),))
    if later.is_none and earlier.is_empty_structure:
        return (WorkerAction(UnitAction.DIG),)
    return ()
