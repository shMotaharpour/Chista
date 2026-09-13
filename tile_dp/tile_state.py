"""TileState: the tile at day start (hour 0), generalized for crops and
animals. Independent state dims (no coupling — engine-verified).

Crops: (crop, age, consec, fert_left, yield)
Animals: (animal, cycle_age, unfed, care_bank, yield)
Kinds: NONE | WEED | PLANT | ANIMAL | EMPTY_STRUCTURE (coop/pasture with
no animal — PLACE a new animal without DIG).
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

_SEED_RES: dict[str, str] = {
    "WHEAT": "SEED_WHEAT", "CARROT": "SEED_CARROT",
    "TOMATO": "SEED_TOMATO", "STRAWBERRY": "SEED_STRAWBERRY",
    "MELON": "SEED_MELON",
}


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
    if name is None:
        return 0
    # stable small code from a fixed vocabulary order
    vocab = list(CROP_NAMES) + list(ANIMAL_NAMES) + ["COOP", "PASTURE"]
    return vocab.index(name) + 1 if name in vocab else 0


def _name_from_code(code: int) -> str | None:
    if code == 0:
        return None
    vocab = list(CROP_NAMES) + list(ANIMAL_NAMES) + ["COOP", "PASTURE"]
    return vocab[code - 1]


def decode_tile(tile: object, day: int, placed_day: int | None = None
                ) -> TileState:
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
        # age origin = FIRST HARVEST DAY (Hossein's convention):
        # age = day - planted_day - (first_yield_day - 1)... precisely:
        # the first harvest day (planted_day + first_yield_day) is age 0,
        # so age = day - (planted_day + first_yield_day).
        age = day - (tile.get("planted_day", day)
                     + spec["first_yield_day"])
        fert_left = max(0, tile.get("fertilized_until_day", -1) - day + 1)
        consec = int(tile.get("consecutive_unwatered", 0))
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
        cycle_age = day - placed - spec["first_yield_day"]
        return TileState(KIND_ANIMAL, None, animal, kind, cycle_age, 0,
                         int(tile.get("consecutive_unfed", 0)), 0,
                         int(tile.get("pending_care_bonus", 0)),
                         int(tile.get("yield_units", 0)))

    raise ValueError(f"unsupported tile kind {kind!r}")
