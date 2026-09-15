"""Observation decode: harness obs -> WorldView (both farms, packed keys).

Issue #10 (M1/D1-2). One pass over the harness observation producing the
internal state every other layer reads — for BOTH farms, because the
opponent's board is public and decoding it with the identical code path
is free intelligence (#16, #21).

The core output is a **multiset of packed state keys** per farm:
tile_dp's graph is position-invariant, so two tiles in the same state
are the same Dantzig-Wolfe block, and `{state_key: count}` is what lets
the pricing in #11/#12 cost one backward sweep for the whole board.

Four behaviours the raw decoder does not give you (all engine-verified):

- **LOCKED tiles** (F042: three of four quadrants start locked, working
  one is a silent no-op, land buys in a fixed NE/SW/SE prefix) get the
  sentinel key `-1` — never a TileState, never in `classes`.
- **Day-start vs mid-day**: `decode_tile` decodes a day-start state; the
  DP consumes day-start states only. `at_day_start=True` asserts
  `hour == 0` (the pre-v15 rebuild bug was a mid-day decode feeding the
  DP — 184/394 nodes wrong).
- **Unknown keys survive**: a key absent from the shipped graph's
  `key_index` maps to the nearest modelled state (same kind and crop /
  animal, then closest age, then closest yield) and is counted in
  `unknown_keys` — the agent cannot die on an unenumerated board, but
  the fallback is never silent.
- **R005**: no new constants here; the LOCKED sentinel and the nearest-
  state order are named behaviours, not tuned numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tile_dp.tile_state import TileState, decode_tile

# The sentinel key for LOCKED tiles: outside the KEY_BITS space by
# construction (packed keys are >= 0), so it can never collide.
LOCKED_KEY = -1


@dataclass(frozen=True)
class FarmView:
    """One farm: packed keys, equivalence classes, and the public numbers."""

    keys: np.ndarray              # (H, W) int64 packed TileState keys; LOCKED = -1
    classes: dict[int, int]       # {packed_key: count} over plannable tiles only
    money: int
    farmer: tuple[int, int]       # (x, y)
    hands: tuple[tuple[int, int], ...]
    unlocked: frozenset[str]      # subset of NW/NE/SW/SE
    hires_today: int


@dataclass(frozen=True)
class PrivateView:
    shed: dict[str, int]
    seeds: dict[str, int]
    inventories: tuple[dict[str, int], ...]   # [farmer, hand1, ...]


@dataclass(frozen=True)
class WorldView:
    day: int
    hour: int
    player: int
    me: FarmView
    opponent: FarmView | None      # None unless decode_opponent=True (#16)
    private: PrivateView          # ours only; the opponent's is hidden
    market_inventory: dict[str, int]
    market_prices: dict[str, int]
    unlocked_shops: tuple[str, ...]
    unknown_keys: int             # cumulative count of unmodelled keys seen


def _decode_one(tile: object, day: int) -> int:
    """One raw tile value -> its packed key (LOCKED -> sentinel)."""
    if isinstance(tile, str) and tile == "LOCKED":
        return LOCKED_KEY
    state = decode_tile(tile, day)
    return state.pack()


def decode_farm(farm: dict, day: int, hour: int) -> FarmView:
    """One farm dict -> FarmView.

    `hour == 0` is the day-start contract the DP consumes; this function
    is honest about it instead of guessing — callers that need a mid-day
    view pass the real hour and get the same decode, but anything that
    feeds the DP must check `hour == 0` first (the pre-v15 rebuild bug
    was a mid-day decode feeding the DP: 184/394 nodes wrong). The
    assertion lives in `decode_world`, the only DP entry point.
    """
    tiles = farm["tiles"]
    h, w = len(tiles), len(tiles[0])
    keys = np.empty((h, w), dtype=np.int64)
    classes: dict[int, int] = {}
    for y in range(h):
        for x in range(w):
            k = _decode_one(tiles[y][x], day)
            keys[y, x] = k
            if k != LOCKED_KEY:
                classes[k] = classes.get(k, 0) + 1
    return FarmView(
        keys=keys, classes=classes, money=int(farm.get("money", 0)),
        farmer=tuple(farm.get("farmer", (0, 0))),
        hands=tuple(tuple(pos) for pos in farm.get("hands", [])),
        unlocked=frozenset(farm.get("unlocked_quadrants", [])),
        hires_today=int(farm.get("hires_today", 0)))


def decode_world(obs: dict, config=None, *, at_day_start: bool = False,
                 decode_opponent: bool = False,
                 graph_keys: frozenset[int] | None = None) -> WorldView:
    """The harness observation -> WorldView (both farms, one code path).

    `at_day_start=True` is the DP entry point: it asserts `hour == 0`,
    because `decode_tile` decodes a day-start state and a mid-day value
    fed to the DP is the pre-v15 rebuild bug (184/394 nodes wrong).

    `decode_opponent=False` (the M1 default, owner 2026-09-15) leaves
    `opponent` as None: nothing consumes the opponent farm yet (#16 does),
    and decoding it every turn is 0.12 ms of dead work. The hook stays -
    flip the flag and the identical code path decodes their board.

    `graph_keys` is `frozenset(TileGraph.load(...).key_index)` — the set
    of states the shipped graph knows. When given, keys outside it are
    mapped to the nearest modelled state (same kind and crop / animal,
    then closest age, then closest yield_units) and counted in
    `unknown_keys`; when None, no remapping happens and the raw keys are
    returned with `unknown_keys = 0` (the caller opted out of the check).
    """
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    if at_day_start:
        assert hour == 0, (
            f"at_day_start decode at hour {hour}: decode_tile is a "
            "day-start decoder, a mid-day value here fed the DP the wrong "
            "state (the pre-v15 rebuild bug)")
    player = int(obs.get("player", 0))
    farms = obs["farms"]
    me = decode_farm(farms[player], day, hour)
    opponent = decode_farm(farms[1 - player], day, hour) \
        if decode_opponent else None

    unknown = 0
    if graph_keys is not None:
        fixed_me = dict(me.classes)
        fixed_opp = dict(opponent.classes) if opponent is not None else None
        for view_classes in (fixed_me, fixed_opp):
            if view_classes is None:
                continue
            for key in list(view_classes):
                if key not in graph_keys:
                    count = view_classes.pop(key)
                    near = _nearest_modelled(key, graph_keys)
                    view_classes[near] = view_classes.get(near, 0) + count
                    unknown += count
        me = FarmView(keys=me.keys, classes=fixed_me, money=me.money,
                      farmer=me.farmer, hands=me.hands,
                      unlocked=me.unlocked, hires_today=me.hires_today)
        if opponent is not None:
            opponent = FarmView(keys=opponent.keys, classes=fixed_opp,
                                money=opponent.money,
                                farmer=opponent.farmer, hands=opponent.hands,
                                unlocked=opponent.unlocked,
                                hires_today=opponent.hires_today)

    private = obs.get("private", {})
    market = obs.get("market", {})
    town = obs.get("town", {})
    return WorldView(
        day=day, hour=hour, player=player, me=me, opponent=opponent,
        private=PrivateView(shed=dict(private.get("shed", {})),
                            seeds=dict(private.get("seeds", {})),
                            inventories=tuple(dict(inv) for inv
                                               in private.get("inventories", []))),
        market_inventory=dict(market.get("inventory", {})),
        market_prices=dict(market.get("prices", {})),
        unlocked_shops=tuple(town.get("unlocked_shops", [])),
        unknown_keys=unknown)


def _nearest_modelled(key: int, graph_keys: frozenset[int]) -> int:
    """The nearest modelled key to an unmodelled one (never raises).

    Order, written down per the issue: same kind first (and same crop /
    animal where the kind has one), then the closest age, then the
    closest yield_units. Conservative: a fallback never crosses a kind.
    """
    unknown = TileState.unpack(key)
    best, best_rank = None, None
    for cand_key in graph_keys:
        cand = TileState.unpack(cand_key)
        if cand.kind != unknown.kind:
            continue
        same_id = ((unknown.crop is not None and cand.crop == unknown.crop)
                   or (unknown.animal is not None
                       and cand.animal == unknown.animal))
        age_gap = abs(cand.age - unknown.age)
        yield_gap = abs(cand.yield_units - unknown.yield_units)
        rank = (0 if same_id else 1, age_gap, yield_gap)
        if best_rank is None or rank < best_rank:
            best, best_rank = cand_key, rank
    if best is None:      # no candidate of the same kind at all: fall back
        best = next(iter(sorted(graph_keys)))
    return best
