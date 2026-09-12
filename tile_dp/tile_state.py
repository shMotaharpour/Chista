"""TileState: the day-start state of ONE tile, as the DP sees it.

Fields are the engine-relevant projection of the tile dict (verified against
the engine — see DESIGN.md "State space"): crop, age (origin = first harvest
day, per Hossein's convention), consec_unwatered (0|1 — 2 became a weed
overnight, F002), fert_left (0|2 days of remaining fertilizer coverage),
yield_units.

The packed int64 key is the dict-free identity used inside numpy arrays:
(crop_id, age, consec, fert, yield) -> 6*8*3*64*4 + ... — big enough for all
crops' ranges; ids are assigned at graph build from these keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Engine constants (R002: imported, never transcribed)
from kaggle_environments.envs.kaggriculture import kaggriculture as K

CROP_NAMES: tuple[str, ...] = ("WHEAT", "CARROT")  # v1
CROP_ID: dict[str, int] = {name: i for i, name in enumerate(CROP_NAMES)}

# Special crop codes for the two non-plant states (negative ids)
CROP_NONE = -1
CROP_WEED = -2

_KIND_TO_CROP_ID = {
    "WHEAT": CROP_ID["WHEAT"],
    "CARROT": CROP_ID["CARROT"],
}


@dataclass(frozen=True)
class TileState:
    """Day-start state of one tile (v1: wheat/carrot plants + None/Weed)."""

    crop_id: int          # CROP_ID value, or CROP_NONE / CROP_WEED
    age: int              # 0 = first harvest day; negative = pre-harvest
    consec_unwatered: int  # 0 | 1 (2 => weed, never a day-start state)
    fert_left: int        # 0 | 1 | 2 remaining fertilizer-covered days
    yield_units: int      # 0..engine cap

    def pack(self) -> int:
        return pack_key(self.crop_id, self.age, self.consec_unwatered,
                        self.fert_left, self.yield_units)

    @classmethod
    def unpack(cls, key: int) -> "TileState":
        return TileState(*unpack_key(key))

    def is_plant(self) -> bool:
        return self.crop_id >= 0

    def describe(self) -> str:
        """Boundary-only human form (never used in the hot path)."""
        if self.crop_id == CROP_NONE:
            return "NONE"
        if self.crop_id == CROP_WEED:
            return "WEED"
        name = CROP_NAMES[self.crop_id]
        return (f"{name} age={self.age} consec={self.consec_unwatered} "
                f"fert_left={self.fert_left} yield={self.yield_units}")


def pack_key(crop_id: int, age: int, consec: int, fert: int,
             yield_units: int) -> int:
    """Pack 5 small ints into one int64 (field widths chosen for all v1 crops).

    Layout (low -> high): yield(7 bits) | fert(3) | consec(3) | age(8, biased
    by +32 to allow negatives) | crop(8, biased by +2 for NONE/WEED).
    """
    return (int(yield_units)
            | (int(fert) << 7)
            | (int(consec) << 10)
            | ((int(age) + 32) << 13)
            | ((int(crop_id) + 2) << 21))


def unpack_key(key: int) -> tuple[int, int, int, int, int]:
    yield_units = key & 0x7F
    fert = (key >> 7) & 0x07
    consec = (key >> 10) & 0x07
    age = ((key >> 13) & 0xFF) - 32
    crop_id = ((key >> 21) & 0xFF) - 2
    return crop_id, age, consec, fert, yield_units


def decode_tile(tile: object, day: int) -> TileState:
    """Engine tile dict -> TileState at day-start (boundary function).

    `tile` is the raw tile value from the observation (None, "LOCKED",
    "WEED", or a plant dict). Age uses Hossein's origin: first harvest day
    of the crop = age 0, so age = day - first_yield_day (negative inside the
    pre-harvest golden window).
    """
    if tile is None:
        return TileState(CROP_NONE, 0, 0, 0, 0)
    if isinstance(tile, str):
        if tile == "WEED":
            return TileState(CROP_WEED, 0, 0, 0, 0)
        raise ValueError(f"tile {tile!r} has no day-start DP state in v1")
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return TileState(CROP_WEED, 0, 0, 0, 0)
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        raise ValueError(f"unsupported tile {tile!r} for tile_dp v1")

    crop = tile["crop"]
    if crop not in _KIND_TO_CROP_ID:
        raise ValueError(f"crop {crop!r} not supported in v1 (wheat+carrot)")
    crop_id = _KIND_TO_CROP_ID[crop]
    spec = K.CROPS[crop]
    age = day - spec["first_yield_day"]

    fert_until = tile.get("fertilized_until_day", -1)
    fert_left = max(0, fert_until - day + 1)  # coverage includes today
    consec = int(tile.get("consecutive_unwatered", 0))
    if consec >= 2:
        # Engine turns the tile into a weed at nightfall (F002); a day-start
        # state with consec>=2 must not occur — decode defensively as weed.
        return TileState(CROP_WEED, 0, 0, 0, 0)

    return TileState(crop_id, age, consec, fert_left,
                     int(tile.get("yield_units", 0)))


def state_age_bounds(crop_id: int) -> tuple[int, int]:
    """Reachable age range for a plant crop (origin = first harvest day).

    Ages run from -(first_yield_day) (planting day) to the last *living*
    day: the engine's lifespan is (planted + max_yield_day + 1) days (F008),
    so the plant is still standing (decaying) at age
    max_yield_day + 1 - first_yield_day; the weed conversion happens the
    next night.
    """
    name = CROP_NAMES[crop_id]
    spec = K.CROPS[name]
    low = -spec["first_yield_day"]
    high = spec["max_yield_day"] + 1 - spec["first_yield_day"] + 1
    return low, high


def all_start_states(crop_id: int) -> list[TileState]:
    """Every *reachable* day-start state for a crop (pruned, not the naive
    cross product): yield is capped by what watering/fertilizer can achieve
    by that age (F005/F006: +1 per watered day in the window, +2 when
    fertilized), fert coverage is bounded by age, and consec_unwatered=1
    states exist only from age 1 (planting day itself counts unwatered)."""
    name = CROP_NAMES[crop_id]
    spec = K.CROPS[name]
    gws_age = (spec["max_yield_day"] + 1) // 2 - spec["first_yield_day"]
    last_age = spec["max_yield_day"] - spec["first_yield_day"] + 1
    cap = spec["max_yield"]

    states: list[TileState] = []
    # last *living* age per the engine lifespan formula (F008) — the weed
    # conversion happens the following night; one extra decay-tail age is
    # kept and any decode beyond the table clamps to WEED (defensive).
    last_age = spec["max_yield_day"] + 1 - spec["first_yield_day"] + 1
    for age in range(-spec["first_yield_day"], last_age + 1):
        days_alive = age + spec["first_yield_day"]  # 0 on planting day
        # yield upper bound: 1 base + 2 per golden-window day so far
        window_days = max(0, min(age, last_age) - max(gws_age, 0) + 1) \
            if age >= gws_age else 0
        y_max = min(cap, 1 + 2 * window_days)
        # fert coverage can only exist if some past day could have fertilized
        fert_max = min(2, max(0, days_alive))
        for fert in range(0, fert_max + 1):
            for consec in (0, 1):
                # on planting day consec is 1 (F002); later days can be 0 or 1
                if days_alive == 0 and consec != 1:
                    continue
                # a consec=1 state must still be watered today => yield stays
                # whatever it is; both consec values are reachable otherwise
                for y in range(0, y_max + 1):
                    states.append(TileState(crop_id, age, consec, fert, y))
    return states
