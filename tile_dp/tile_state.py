"""TileState: the tile AT DAY START (hour 0), carrot v1 (+ NONE / WEED).

Fields (engine-relevant projection, all verified against the engine):
  kind    : "NONE" | "WEED" | "PLANT"
  crop    : "CARROT" (v1)
  age     : days since first harvest day (origin = first harvest day;
            age -1 = the day before it; age 2 = last living day for carrot)
  consec  : consecutive-unwatered YESTERDAY (0|1; 2 => weed, never a
            day-start state — F002)
  fert_left: remaining fertilizer-covered days INCLUDING today (0|1|2)
  yield   : units sitting on the plant

The packed int64 key is the numpy-friendly identity; ids are assigned at
graph build from these keys.
"""

from __future__ import annotations

from dataclasses import dataclass

# Engine constants (R002: imported, never transcribed)
from kaggle_environments.envs.kaggriculture import kaggriculture as K

KIND_NONE = "NONE"
KIND_WEED = "WEED"
KIND_PLANT = "PLANT"


@dataclass(frozen=True)
class TileState:
    """Day-start state of ONE tile (carrot v1)."""

    kind: str             # NONE | WEED | PLANT
    crop: str | None      # "CARROT" for plants, None otherwise
    age: int              # 0 = first harvest day (plants only)
    consec: int           # 0 = watered yesterday, 1 = dry yesterday
    fert_left: int        # 0..2 covered days including today (plants only)
    yield_units: int      # 0..engine cap (plants only)

    def pack(self) -> int:
        crop_code = 1 if self.kind == KIND_PLANT else 0
        age_code = self.age + 32
        return (int(self.yield_units)
                | (int(self.fert_left) << 7)
                | (int(self.consec) << 10)
                | (age_code << 13)
                | (crop_code << 21))

    @classmethod
    def unpack(cls, key: int) -> "TileState":
        yield_units = key & 0x7F
        fert = (key >> 7) & 0x07
        consec = (key >> 10) & 0x07
        age = ((key >> 13) & 0xFF) - 32
        is_plant = (key >> 21) & 0x01
        if is_plant:
            return cls(KIND_PLANT, "CARROT", age, consec, fert, yield_units)
        return cls(KIND_NONE if age == 0 else KIND_WEED, None, 0, 0, 0, 0)

    def describe(self) -> str:
        if self.kind == KIND_NONE:
            return "NONE"
        if self.kind == KIND_WEED:
            return "WEED"
        return (f"CARROT age={self.age} consec={self.consec} "
                f"fert_left={self.fert_left} yield={self.yield_units}")


def decode_tile(tile: object, day: int) -> TileState:
    """Engine tile value (None, "LOCKED", "WEED", plant dict) -> TileState
    at day start. Age uses the first-harvest-day origin."""
    if tile is None:
        return TileState(KIND_NONE, None, 0, 0, 0, 0)
    if isinstance(tile, str):
        if tile == "WEED":
            return TileState(KIND_WEED, None, 0, 0, 0, 0)
        raise ValueError(f"tile {tile!r} has no day-start state")
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return TileState(KIND_WEED, None, 0, 0, 0, 0)
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        raise ValueError(f"unsupported tile {tile!r}")
    crop = tile["crop"]
    if crop != "CARROT":
        raise ValueError(f"crop {crop!r} not supported (v1: CARROT only)")
    spec = K.CROPS[crop]
    age = day - spec["first_yield_day"]
    fert_left = max(0, tile.get("fertilized_until_day", -1) - day + 1)
    consec = int(tile.get("consecutive_unwatered", 0))
    if consec >= 2:  # engine weed conversion (F002) — decode defensively
        return TileState(KIND_WEED, None, 0, 0, 0, 0)
    if consec == 1:
        # yesterday was dry → the fertilizer day cannot have been watered
        # yesterday → at most 1 covered day can remain today (F004 + the
        # watered-fert-day coupling verified on the engine)
        fert_left = min(fert_left, 1)
    return TileState(KIND_PLANT, crop, age, consec, fert_left,
                     int(tile.get("yield_units", 0)))
