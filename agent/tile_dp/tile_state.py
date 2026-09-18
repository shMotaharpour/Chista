"""The DP's view of a tile: world's `TileHourZero`, plus the node key.

The definition of a tile — its fields, its decode, the age conventions — lives in
`agent.world.tile`. What is here is the one thing the DP needs on top of it: a node has
to be an int, so the state is packed into a bit layout. The layout is the only thing
this file owns, and its widths are computed from the engine's own tables (`CROPS`,
`ANIMALS`) rather than typed in, so a crop or an animal added to the game widens the key
by itself instead of silently colliding with an existing name code.

Everything else is re-exported, so a reader of the graph code finds the same names it
always did, now defined once in world.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from agent.world.model import ANIMALS, CROPS, STRUCTURES, Animal, Crop, Structure, TileKind
from agent.world.rules import ANIMAL_RULES, CROP_RULES
from agent.world.tile import (DRY_LIMIT, FERT_DAYS, TileHourZero, animal_cycle_age,
                              crop_age_origin, crop_last_day, crop_weed_age)

# --- the names, as world spells them ----------------------------------------- #

KIND_NONE = TileKind.NONE
KIND_WEED = TileKind.WEED
KIND_PLANT = TileKind.PLANT
KIND_ANIMAL = TileKind.ANIMAL
KIND_EMPTY_COOP = TileKind.EMPTY_COOP
KIND_EMPTY_PASTURE = TileKind.EMPTY_PASTURE
EMPTY_KINDS: tuple[TileKind, ...] = (KIND_EMPTY_COOP, KIND_EMPTY_PASTURE)
EMPTY_KIND_OF_STRUCTURE: dict[Structure, TileKind] = {
    Structure.COOP: KIND_EMPTY_COOP, Structure.PASTURE: KIND_EMPTY_PASTURE}
#: The kinds a DP node can be: world's six working kinds. LOCKED is not one — a locked
#: tile is a farm-level fact, not a state a plan works on.
KIND_CODES: tuple[TileKind, ...] = (KIND_NONE, KIND_WEED, KIND_PLANT, KIND_ANIMAL,
                                    KIND_EMPTY_COOP, KIND_EMPTY_PASTURE)

CROP_NAMES: tuple[str, ...] = CROPS
ANIMAL_NAMES: tuple[str, ...] = ANIMALS

#: Engine step granularity, kept for the readers that compare engine steps
#: (kaggriculture.py:864). The decode in world does not need it.
TURNS_PER_DAY = 24

# --- the packed node key ------------------------------------------------------ #

#: Fixed vocabulary of the name fields in `pack` (crops, then animals, then the two
#: structures); the bit field stores index + 1 so 0 means "no name".
_VOCAB: tuple[str, ...] = CROP_NAMES + ANIMAL_NAMES + STRUCTURES

_AGE_LO = -max(crop_age_origin(c) for c in CROPS)
_AGE_HI = max(crop_last_day(c) - crop_age_origin(c) for c in CROPS)
_YIELD_MAX = max([int(CROP_RULES[c]["max_yield"]) for c in CROPS]
                 + [int(s["max_held"]) for s in ANIMAL_RULES.values()])
_CARE_MAX = max(int(s["max_held"]) for s in ANIMAL_RULES.values())
_NAME_MAX = len(_VOCAB)
_KIND_MAX = len(KIND_CODES) - 1
AGE_BIAS = -_AGE_LO


def _bits(hi: int) -> int:
    """Bits needed for the range 0..hi (at least one)."""
    return max(1, int(hi).bit_length())


KEY_FIELDS: tuple[tuple[str, int], ...] = (
    ("yield_units", _bits(_YIELD_MAX)),
    ("fert_left", _bits(FERT_DAYS - 1)),
    ("consec", _bits(DRY_LIMIT - 1)),
    ("unfed", _bits(DRY_LIMIT - 1)),
    ("care_bank", _bits(_CARE_MAX)),
    ("age", _bits(_AGE_HI + AGE_BIAS)),
    ("crop_code", _bits(_NAME_MAX)),
    ("animal_code", _bits(_NAME_MAX)),
    ("structure_code", _bits(_NAME_MAX)),
    ("kind_code", _bits(_KIND_MAX)),
)
KEY_BITS = sum(width for _, width in KEY_FIELDS)


@dataclass(frozen=True)
class TileState(TileHourZero):
    """A day-start tile that can be packed into the DP's node key."""

    def pack(self) -> int:
        # Layout and widths live in KEY_FIELDS: the shift of every field is derived from
        # the widths, never hand-written, so fields cannot overlap.
        key = 0
        shift = 0
        for name, width in KEY_FIELDS:
            value = _field_value(self, name)
            if not 0 <= value < (1 << width):
                raise ValueError(f"tile field {name}={value} does not fit in {width} "
                                 f"bits ({self.describe()})")
            key |= value << shift
            shift += width
        return key

    @classmethod
    def unpack(cls, key: int) -> "TileState":
        """Inverse of `pack` (a key outside the layout raises)."""
        if not 0 <= key < (1 << KEY_BITS):
            raise ValueError(f"tile key {key} is outside the {KEY_BITS}-bit layout")
        field: dict[str, int] = {}
        shift = 0
        for name, width in KEY_FIELDS:
            field[name] = (key >> shift) & ((1 << width) - 1)
            shift += width
        kind_code = field["kind_code"]
        if kind_code >= len(KIND_CODES):
            raise ValueError(f"packed key {key} names kind code {kind_code}, but only "
                             f"{len(KIND_CODES)} kinds exist")
        crop = _name_from_code(field["crop_code"])
        animal = _name_from_code(field["animal_code"])
        structure = _name_from_code(field["structure_code"])
        return cls(KIND_CODES[kind_code],
                   Crop(crop) if crop else None,
                   Animal(animal) if animal else None,
                   Structure(structure) if structure else None,
                   field["age"] - AGE_BIAS, field["consec"], field["unfed"],
                   field["fert_left"], field["care_bank"], field["yield_units"])


def decode_tile(tile: object, day: int) -> TileState:
    """Engine tile value at day start -> TileState (world's decode, as this class)."""
    start = TileHourZero.decode(tile, day)
    return TileState(**{f.name: getattr(start, f.name) for f in fields(TileHourZero)})


def _name_code(name: str | None) -> int:
    """Code of a name in the fixed vocabulary order (0 = none).

    An unknown name RAISES instead of packing as "none" (owner, 2026-09-14): the old
    version returned 0 for a typo, so a misspelled crop or structure silently shared the
    "empty" code with no name at all and no one noticed.
    """
    if name is None:
        return 0
    if name not in _VOCAB:
        raise ValueError(f"unknown name {name!r} is not in the tile vocabulary "
                         f"{_VOCAB}; a name that is not in the vocabulary must never "
                         "pack as 'none'")
    return _VOCAB.index(name) + 1


def _name_from_code(code: int) -> str | None:
    """Inverse of `_name_code` (0 = none; an out-of-range code raises)."""
    if code == 0:
        return None
    if not 0 < code <= len(_VOCAB):
        raise ValueError(f"vocabulary code {code} is out of range 0..{len(_VOCAB)}")
    return _VOCAB[code - 1]


def _field_value(state: "TileState", name: str) -> int:
    """Non-negative value of one packed field (KEY_FIELDS layout)."""
    if name == "kind_code":
        if state.kind not in KIND_CODES:
            raise ValueError(f"unknown kind {state.kind!r}: not one of {KIND_CODES}")
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
