"""TileState: the tile at day start (hour 0), generalized for crops and
animals. Independent state dims (no coupling — engine-verified).

Crops: (crop, age, consec, fert_left, yield)
Animals: (animal, age, unfed, care_bank, yield)
Kinds: NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE (coop/pasture with
no animal — PLACE a new animal without DIG).

Age origins (Hossein's convention, 2026-09-14):
- one-shot crop: age 0 = START OF THE GOLDEN WINDOW = (max_yield_day + 1) // 2;
  the last planned day is max_yield_day, and the day the plant starts
  turning into a weed (engine max_lifespan_step) is decoded as WEED.
- ongoing crop: age 0 = max_yield_day; the last planned day is the last
  production day (max_yield_day + (max_yield - 1) * interval); the next
  day (the weed day) is decoded as WEED.
- animal: age 0 = first_yield_day (placement day itself is intra-day, so
  the negative range is 1 - first_yield_day .. -1) and the positive range
  is the production phase 0 .. interval - 1 (it wraps: the calendar age is
  not a decision variable).
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture import kaggriculture as K

KIND_NONE = "NONE"
KIND_WEED = "WEED"
KIND_PLANT = "PLANT"
KIND_ANIMAL = "ANIMAL"
KIND_EMPTY_STRUCTURE = "EMPTY_STRUCTURE"

CROP_NAMES: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                               "MELON")
ANIMAL_NAMES: tuple[str, ...] = ("GOOSE", "COW", "SHEEP")

# Engine step granularity: one day is 24 steps. The single definition in
# tile_dp: graph.py imports TURNS_PER_DAY from here (v16).
TURNS_PER_DAY = 24

# Fixed vocabulary of the name fields in `pack` (crops, then animals, then the
# two structures); the bit field stores index + 1 so 0 means "no name".
_VOCAB: tuple[str, ...] = CROP_NAMES + ANIMAL_NAMES + ("COOP", "PASTURE")


def crop_age_origin(spec: dict) -> int:
    """Crop day whose age is 0 (Hossein's convention).

    one-shot: START OF GOLDEN WINDOW = (max_yield_day + 1) // 2.
    ongoing:  max_yield_day.
    """
    if spec.get("ongoing"):
        return int(spec["max_yield_day"])
    return (int(spec["max_yield_day"]) + 1) // 2


@dataclass(frozen=True)
class TileState:
    """Day-start state of ONE tile (any crop/animal v2)."""

    kind: str                     # NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE
    crop: str | None              # plant crop name
    animal: str | None            # animal species name
    structure: str | None         # COOP | PASTURE (for animal/empty)
    age: int                      # lifecycle day (crop) / cycle age (animal)
    consec: int                   # crops: 0|1 yesterday watered/dry
    unfed: int                    # animals: 0|1 yesterday fed/hungry
    fert_left: int                # crops: 0..2 covered days incl today
    care_bank: int                # animals: 0..max_held banked care
    yield_units: int              # 0..cap

    def pack(self) -> int:
        kind_code = {"NONE": 0, "WEED": 1, "PLANT": 2, "ANIMAL": 3,
                     "EMPTY_STRUCTURE": 4}[self.kind]
        crop_code = _name_code(self.crop)
        animal_code = _name_code(self.animal)
        struct_code = _name_code(self.structure)
        # Bit layout (50 of 64 bits used): yield_units 0..7, fert_left 8..10,
        # consec 11..12, unfed 13..14, care_bank 15..20, age+64 21..28,
        # crop 29..33, animal 34..38, structure 39..44, kind 47..49.
        # KNOWN LIMITATION (documented, owner 2026-09-14: guard not wanted yet):
        # no range check - a value wider than its field spills silently into the
        # next one, so two different states could share a key; only the engine's
        # own ranges keep this safe today.
        return (int(self.yield_units)
                | (self.fert_left << 8)
                | (self.consec << 11)
                | (self.unfed << 13)
                | (self.care_bank << 15)
                | ((self.age + 64) << 21)
                | (crop_code << 29)
                | (animal_code << 34)
                | (struct_code << 39)
                | (kind_code << 47))

    @classmethod
    def unpack(cls, key: int) -> "TileState":
        y = key & 0xFF
        fert = (key >> 8) & 0x07
        consec = (key >> 11) & 0x03
        unfed = (key >> 13) & 0x03
        bank = (key >> 15) & 0x3F
        age = ((key >> 21) & 0xFF) - 64
        crop = _name_from_code((key >> 29) & 0x1F)
        animal = _name_from_code((key >> 34) & 0x1F)
        structure = _name_from_code((key >> 39) & 0x3F)
        kind_code = (key >> 47) & 0x07
        kind = {0: KIND_NONE, 1: KIND_WEED, 2: KIND_PLANT, 3: KIND_ANIMAL,
                4: KIND_EMPTY_STRUCTURE}[kind_code]
        return cls(kind, crop, animal, structure, age, consec, unfed,
                   fert, bank, y)

    def describe(self) -> str:
        if self.kind == KIND_NONE:
            return "NONE"
        if self.kind == KIND_WEED:
            return "WEED"
        if self.kind == KIND_EMPTY_STRUCTURE:
            return f"EMPTY {self.structure}"
        if self.kind == KIND_ANIMAL:
            return (f"{self.animal} age={self.age} unfed={self.unfed} "
                    f"bank={self.care_bank} y={self.yield_units}")
        return (f"{self.crop} age={self.age} consec={self.consec} "
                f"fert_left={self.fert_left} y={self.yield_units}")


def _name_code(name: str | None) -> int:
    """Code of a name in the fixed vocabulary order (0 = none / unknown)."""
    if name is None or name not in _VOCAB:
        return 0
    return _VOCAB.index(name) + 1


def _name_from_code(code: int) -> str | None:
    if code == 0:
        return None
    return _VOCAB[code - 1]


def decode_tile(tile: object, day: int) -> TileState:
    """Engine tile value at day start -> TileState."""
    if tile is None:
        return TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0)
    if isinstance(tile, str):
        if tile == "WEED":
            return TileState(KIND_WEED, None, None, None, 0, 0, 0, 0, 0, 0)
        raise ValueError(f"tile {tile!r} has no day-start state")
    if not isinstance(tile, dict):
        raise ValueError(f"unsupported tile {tile!r}")

    kind = tile.get("kind")
    if kind == "WEED":
        return TileState(KIND_WEED, None, None, None, 0, 0, 0, 0, 0, 0)

    if kind == "PLANT":
        crop = tile["crop"]
        spec = K.CROPS[crop]
        # The day the plant STARTS turning into a weed (the engine stamps
        # max_lifespan_step for exactly that day: once at plant time for
        # one-shot crops, on the night the last unit is produced for
        # ongoing ones) is decoded as WEED: the project never plans on it.
        mls = int(tile.get("max_lifespan_step", -1) or -1)
        # MAGIC NUMBER (engine, kaggriculture.py:226/800): TODO - pin it with a
        # probe test against the engine instead of trusting this copy.
        if mls > 0 and day * TURNS_PER_DAY >= mls:
            return TileState(KIND_WEED, None, None, None, 0, 0, 0, 0, 0, 0)
        age = day - (tile.get("planted_day", day) + crop_age_origin(spec))
        fert_left = max(0, tile.get("fertilized_until_day", -1) - day + 1)
        consec = int(tile.get("consecutive_unwatered", 0))
        # MAGIC NUMBER (engine, kaggriculture.py:783): two dry days turn the
        # plant into a weed. TODO - pin it with a probe test (owner 2026-09-14).
        if consec >= 2:
            return TileState(KIND_WEED, None, None, None, 0, 0, 0, 0, 0, 0)
        return TileState(KIND_PLANT, crop, None, None, age, consec, 0,
                         fert_left, 0, int(tile.get("yield_units", 0)))

    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        if animal is None:
            return TileState(KIND_EMPTY_STRUCTURE, None, None, kind, 0,
                             0, 0, 0, 0, 0)
        spec = K.ANIMALS[animal]
        placed = tile.get("placed_day", day)
        # age < 0: growing up (placement day is intra-day, so the range is
        # 1 - first_yield_day .. -1); age >= 0: the production phase, which
        # wraps inside 0 .. interval - 1.
        cycle_age = day - placed - spec["first_yield_day"]
        if cycle_age >= 0:
            cycle_age %= int(spec["interval"])
        # care_bank is a contract cap (Hossein): the engine can bank more
        # than max_held before the first production, the model caps it.
        bank = min(int(tile.get("pending_care_bonus", 0)),
                   int(spec["max_held"]))
        return TileState(KIND_ANIMAL, None, animal, kind, cycle_age, 0,
                         int(tile.get("consecutive_unfed", 0)), 0,
                         bank,
                         int(tile.get("yield_units", 0)))

    raise ValueError(f"unsupported tile kind {kind!r}")
