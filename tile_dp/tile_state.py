"""TileState: the tile at day start (hour 0), generalized for crops and
animals. Independent state dims (no coupling — engine-verified).

Crops: (crop, age, consec, fert_left, yield)
Animals: (animal, age, unfed, care_bank, yield)
Kinds: NONE | WEED | PLANT | ANIMAL | EMPTY_COOP | EMPTY_PASTURE (a structure
with no animal - PLACE a new animal without DIG). The two empty structures are
separate kinds (2026-09-14): only a COOP holds a GOOSE and only a PASTURE holds
a COW/SHEEP, and BUILD_COOP / BUILD_PASTURE need a NONE tile (they refuse any
other tile) while DIG turns a plant, a weed or an empty structure back into
NONE (kaggriculture.py:484-503).

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
# An empty structure is its own kind per structure type (decision 2026-09-14):
# a COOP with no animal and a PASTURE with no animal differ in what they can
# PLACE (GOOSE vs COW/SHEEP), so they are separate graph nodes.
KIND_EMPTY_COOP = "EMPTY_COOP"
KIND_EMPTY_PASTURE = "EMPTY_PASTURE"
EMPTY_KINDS: tuple[str, ...] = (KIND_EMPTY_COOP, KIND_EMPTY_PASTURE)
EMPTY_KIND_OF_STRUCTURE: dict[str, str] = {"COOP": KIND_EMPTY_COOP,
                                           "PASTURE": KIND_EMPTY_PASTURE}
KIND_CODES: tuple[str, ...] = (KIND_NONE, KIND_WEED, KIND_PLANT, KIND_ANIMAL,
                               KIND_EMPTY_COOP, KIND_EMPTY_PASTURE)

# Name vocabulary, read from the engine tables themselves (2026-09-14): a crop
# or animal added to kaggriculture.py widens the packed key by itself instead
# of silently colliding with an existing name code.
CROP_NAMES: tuple[str, ...] = tuple(K.CROPS)
ANIMAL_NAMES: tuple[str, ...] = tuple(K.ANIMALS)

# Engine step granularity: `turns_per_day` of the interpreter, whose own
# default is 24 (kaggriculture.py:864, `get(cfg, "turnsPerDay", 24)`); the
# engine stamps `max_lifespan_step` with it, so the decoder below compares
# engine steps against it. A run that overrides `turnsPerDay` breaks this
# constant (TODO: read the run's own configuration).
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


def crop_last_day(spec: dict) -> int:
    """Last day of life on which a crop is still a PLANT (engine table).

    one-shot: the harvest deadline itself, the night after it the tile is WEED
    (kaggriculture.py:224). ongoing: the last production day, max_yield_day +
    (max_yield - 1) * interval (kaggriculture.py:789-801).
    """
    if spec["ongoing"]:
        return (int(spec["max_yield_day"])
                + (int(spec["max_yield"]) - 1) * max(1, int(spec["interval"])))
    return int(spec["max_yield_day"])


# --- packed tile key (2026-09-14) ---
# Widths COMPUTED from the table or constant that defines the range, never
# typed in; shifts are cumulative and pack() range-checks every field, so a
# too-wide value raises instead of spilling into its neighbour.
_DRY_LIMIT = 2      # engine: a second dry night makes the tile WEED; a second
                    # unfed night makes the animal escape (lines 783, 817);
                    # verify_engine_constants() re-checks both live
_FERT_DAYS = 3      # engine: a dose covers its own day plus two more
_CROP_SPECS = tuple(K.CROPS.values())
_AGE_LO = -max(crop_age_origin(s) for s in _CROP_SPECS)
_AGE_HI = max(crop_last_day(s) - crop_age_origin(s) for s in _CROP_SPECS)
_YIELD_MAX = max([int(s["max_yield"]) for s in _CROP_SPECS]
                 + [int(s["max_held"]) for s in K.ANIMALS.values()])
_CARE_MAX = max(int(s["max_held"]) for s in K.ANIMALS.values())
_NAME_MAX = len(_VOCAB)
_KIND_MAX = len(KIND_CODES) - 1
AGE_BIAS = -_AGE_LO


def _bits(hi: int) -> int:
    """Bits needed for the range 0..hi (at least one)."""
    return max(1, int(hi).bit_length())


KEY_FIELDS: tuple[tuple[str, int], ...] = (
    ("yield_units", _bits(_YIELD_MAX)),
    ("fert_left", _bits(_FERT_DAYS - 1)),
    ("consec", _bits(_DRY_LIMIT - 1)),
    ("unfed", _bits(_DRY_LIMIT - 1)),
    ("care_bank", _bits(_CARE_MAX)),
    ("age", _bits(_AGE_HI + AGE_BIAS)),
    ("crop_code", _bits(_NAME_MAX)),
    ("animal_code", _bits(_NAME_MAX)),
    ("structure_code", _bits(_NAME_MAX)),
    ("kind_code", _bits(_KIND_MAX)),
)
KEY_BITS = sum(width for _, width in KEY_FIELDS)


@dataclass(frozen=True)
class TileState:
    """Day-start state of ONE tile (any crop/animal v2)."""

    kind: str                     # NONE | WEED | PLANT | ANIMAL | EMPTY_COOP
                                  # | EMPTY_PASTURE
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
        # Layout and widths live in KEY_FIELDS: the shift of every field is
        # derived from the widths, never hand-written, so fields cannot overlap.
        key = 0
        shift = 0
        for name, width in KEY_FIELDS:
            value = _field_value(self, name)
            if not 0 <= value < (1 << width):
                raise ValueError(
                    f"tile field {name}={value} does not fit in {width} bits "
                    f"({self.describe()})")
            key |= value << shift
            shift += width
        return key

    @classmethod
    def unpack(cls, key: int) -> "TileState":
        """Inverse of `pack` (a key outside the layout raises)."""
        if not 0 <= key < (1 << KEY_BITS):
            raise ValueError(f"tile key {key} is outside the {KEY_BITS}-bit "
                             "layout")
        field: dict[str, int] = {}
        shift = 0
        for name, width in KEY_FIELDS:
            field[name] = (key >> shift) & ((1 << width) - 1)
            shift += width
        kind_code = field["kind_code"]
        if kind_code >= len(KIND_CODES):
            raise ValueError(f"packed key {key} names kind code {kind_code}, "
                             f"but only {len(KIND_CODES)} kinds exist")
        return cls(KIND_CODES[kind_code],
                   _name_from_code(field["crop_code"]),
                   _name_from_code(field["animal_code"]),
                   _name_from_code(field["structure_code"]),
                   field["age"] - AGE_BIAS, field["consec"], field["unfed"],
                   field["fert_left"], field["care_bank"], field["yield_units"])

    def describe(self) -> str:
        if self.kind == KIND_NONE:
            return "NONE"
        if self.kind == KIND_WEED:
            return "WEED"
        if self.kind in EMPTY_KINDS:
            return f"EMPTY {self.structure}"
        if self.kind == KIND_ANIMAL:
            return (f"{self.animal} age={self.age} unfed={self.unfed} "
                    f"bank={self.care_bank} y={self.yield_units}")
        return (f"{self.crop} age={self.age} consec={self.consec} "
                f"fert_left={self.fert_left} y={self.yield_units}")


def _name_code(name: str | None) -> int:
    """Code of a name in the fixed vocabulary order (0 = none).

    An unknown name RAISES instead of packing as "none" (owner, 2026-09-14):
    the old version returned 0 for a typo, so a misspelled crop or structure
    silently shared the "empty" code with no name at all and no one noticed.
    """
    if name is None:
        return 0
    if name not in _VOCAB:
        raise ValueError(
            f"unknown name {name!r} is not in the tile vocabulary {_VOCAB}; a "
            "name that is not in the vocabulary must never pack as 'none'")
    return _VOCAB.index(name) + 1


def _name_from_code(code: int) -> str | None:
    """Inverse of `_name_code` (0 = none; an out-of-range code raises)."""
    if code == 0:
        return None
    if not 0 < code <= len(_VOCAB):
        raise ValueError(f"vocabulary code {code} is out of range 0.."
                         f"{len(_VOCAB)}")
    return _VOCAB[code - 1]


def _field_value(state: "TileState", name: str) -> int:
    """Non-negative value of one packed field (KEY_FIELDS layout)."""
    if name == "kind_code":
        if state.kind not in KIND_CODES:
            raise ValueError(f"unknown kind {state.kind!r}: not one of "
                             f"{KIND_CODES}")
        return KIND_CODES.index(state.kind)
    if name == "age":
        return state.age + AGE_BIAS
    if name == "crop_code":
        return _name_code(state.crop)
    if name == "animal_code":
        return _name_code(state.animal)
    if name == "structure_code":
        return _name_code(state.structure)
    return int(getattr(state, name))


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
        if consec >= _DRY_LIMIT:      # engine: a second dry night = WEED
            return TileState(KIND_WEED, None, None, None, 0, 0, 0, 0, 0, 0)
        return TileState(KIND_PLANT, crop, None, None, age, consec, 0,
                         fert_left, 0, int(tile.get("yield_units", 0)))

    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        if animal is None:
            return TileState(EMPTY_KIND_OF_STRUCTURE[kind], None, None, kind,
                             0, 0, 0, 0, 0, 0)
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
