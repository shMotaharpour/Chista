"""The rival's harvest calendar (#16): their board is PUBLIC — `farms[i]`
tiles carry crop, age, yield — so the day each of their planted tiles
pays out is derivable, and their future supply is a dated forecast, not a
mystery.

`decode_world(obs, decode_opponent=True)` (the hook #16 left) reads their
25 unlocked tiles through the same decode our own board gets. This module
turns those packed tile states into (day, good, units) supply events and
a per-good supply curve the forecast can consume as the rival half of
#110's plan-supply conditioning.

Scope note: this prices their PLANTED tiles' scheduled payouts. It cannot
see tiles they will plant later (that gap is #22's planting prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.obs import decode_world
from agent.tile_dp.tile_state import TileState, TileZeroCode
from agent.world.model import Crop, TileKind
from agent.world.tile import crop_age_origin, crop_last_day


@dataclass(frozen=True)
class RivalHarvestEvent:
    """One planted rival tile's scheduled payout."""

    day: int          # season day the yield becomes collectable
    good: str
    units: int        # the tile's accumulated yield_units
    x: int
    y: int


def _unpack(key: int) -> TileState:
    return TileState.unpack(TileZeroCode(key))


def harvest_events(obs: dict, horizon: int = 30) -> list[RivalHarvestEvent]:
    """The rival board's scheduled payouts, from today's decode.

    A planted tile's yield is collectable on `crop_last_day(crop)` from
    its planting; its packed age is counted from `crop_age_origin`, so
    the event day is `last_day - (age - origin)` in season days — read
    off `obs["day"]` so the calendar is dated, not relative. Ongoing
    crops (animals, multi-harvest) appear once per remaining cycle with
    their current accumulated yield; the tracker's residual is the
    authority on realised amounts, this is the SCHEDULE.
    """
    view = decode_world(obs, decode_opponent=True)
    if view.opponent is None:
        return []
    day = int(view.day)
    keys = view.opponent.keys
    tiles = (obs.get("farms", []) or [])[1 - int(view.player)].get("tiles", [])
    events: list[RivalHarvestEvent] = []
    for y in range(keys.shape[0]):
        for x in range(keys.shape[1]):
            key = int(keys[y][x])
            if key == -1:                       # LOCKED sentinel
                continue
            st = _unpack(key)
            if st.kind is not TileKind.PLANT or st.crop is None:
                continue
            crop = st.crop.value if hasattr(st.crop, "value") else str(st.crop)
            last = crop_last_day(st.crop)
            origin = crop_age_origin(st.crop)
            # the packed age counts from `origin` (the golden window's
            # start); the tile's REAL age is `age - origin`, so its
            # planting day was `day - real_age` and its payout lands on
            # `planting + last` (crop_last_day counts from planting).
            real_age = st.age - origin
            event_day = day - real_age + last
            if event_day < day:
                continue                    # window already past
            raw = tiles[y][x] if tiles else None
            units = int(st.yield_units)
            if isinstance(raw, dict):
                units = max(units, int(raw.get("yield_units", 0)))
            if units > 0 and 0 <= event_day < horizon:
                events.append(RivalHarvestEvent(
                    day=event_day, good=crop, units=units, x=x, y=y))
    return events


def supply_curve(obs: dict, goods: tuple[str, ...], horizon: int = 30
                 ) -> np.ndarray:
    """`(horizon, n_goods)` units/day the rival's planted tiles schedule.

    This is the rival half of the plan-supply conditioning (#110): the
    price path that assumes no rival supply is a path that ignores half
    the board.
    """
    out = np.zeros((max(1, int(horizon)), len(goods)), dtype=np.float64)
    ix = {g: i for i, g in enumerate(goods)}
    for ev in harvest_events(obs, horizon=horizon):
        if ev.good in ix and ev.day < out.shape[0]:
            out[ev.day, ix[ev.good]] += ev.units
    return out
