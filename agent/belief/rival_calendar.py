"""The rival's harvest calendar (#16): their board is PUBLIC — `farms[i]`
tiles carry crop, age, yield — so the days each of their planted tiles
pays out are derivable, and their future supply is a dated forecast, not a
mystery.

`decode_world(obs, decode_opponent=True)` (the hook #16 left) reads their
25 unlocked tiles through the same decode our own board gets. This module
turns those packed tile states into (day, good, units) supply events and
a per-good supply curve the forecast can consume as the rival half of
#110's plan-supply conditioning.

The SCHEDULE is a FLOOR, deliberately (owner, 2026-09-27): each event sits
on the FIRST day a production can be collected, and its units are the
minimum that will exist by then — what the tile already carries plus what
accrues by itself. Nobody can harvest before the floor, so "we have at
least this long and at least this many units" is a guarantee. Fertiliser
(and, for animals, care) is the owner's decision, not ours: it can only
ADD units on top, so the floor holds either way. `units_hint` on an event
carries the fertilised expectation, read off their PUBLIC tile
(`fertilized_until_day`), for whoever wants the estimate rather than the
bound.

Scope note: this prices their PLANTED tiles' scheduled payouts. It cannot
see tiles they will plant later (that gap is #22's planting prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.obs import decode_world
from agent.tile_dp.tile_state import TileState, TileZeroCode
from agent.world.model import TileKind
from agent.world.rules import CROP_RULES
from agent.world.tile import crop_age_origin, crop_last_day


@dataclass(frozen=True)
class RivalHarvestEvent:
    """One planted rival tile's scheduled payout."""

    day: int          # season day the yield becomes collectable
    good: str
    units: int        # the floor: what exists by `day` with no further action
    x: int
    y: int
    units_hint: int = 0   # the fertilised expectation (public tile read); >= units


def _unpack(key: int) -> TileState:
    return TileState.unpack(TileZeroCode(key))


def _plant_schedule(crop: str) -> list[int]:
    """Ages (days from planting) at which the crop's yield is collectable.

    One-shot: the WATER window's START, `(max_yield_day + 1) // 2` — the
    first day a watering can add a unit, so the earliest day anything can
    be harvested; from then the tile's units only grow, and
    `crop_last_day` is the WEED deadline, not a payout day (measured:
    melon is harvested at age 10, wheat from age 2 — the replay's own
    histogram, `offline_lab/rival_calendar_accuracy.py`).
    Ongoing: every interval step from `max_yield_day` on, `max_yield`
    productions (kaggriculture.py:792-802).
    """
    spec = CROP_RULES[crop]
    if spec["ongoing"]:
        step = max(1, int(spec["interval"]))
        return [int(spec["max_yield_day"]) + k * step
                for k in range(int(spec["max_yield"]))]
    return [(int(spec["max_yield_day"]) + 1) // 2]


def harvest_events(obs: dict, horizon: int = 30) -> list[RivalHarvestEvent]:
    """The rival board's scheduled payouts, from today's decode.

    The packed age is `day - planted - origin` (world/tile.py:164), so the
    tile's REAL age in days-from-planting is `st.age + origin` — the
    pre-fix module subtracted the origin instead and dated every payout
    `2*origin` days late. Events land on `_plant_schedule`'s remaining
    ages, dated off `obs["day"]`; ages already past carry no event — the
    units they produced are already IN `yield_units` and belong to today's
    floor.

    `units` is the floor by the event's day: the tile's current
    `yield_units` plus what accrues unwatered-by-nobody is NOT claimed —
    the floor claims only what exists now, and each remaining scheduled
    production adds at least 1 (2 with their fertiliser, which is their
    choice and only raises the number — `units_hint`).
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
            origin = crop_age_origin(st.crop)
            # REAL age from planting: the decode built the packed age as
            # `day - planted - origin`, so adding the origin back gives
            # `day - planted`. The pre-fix `st.age - origin` read negative
            # for growing tiles and dated every payout 2*origin days late.
            real_age = st.age + origin
            raw = tiles[y][x] if tiles else None
            # The floor takes what exists NOW. Their PUBLIC tile names the
            # fertiliser state (`fertilized_until_day`), so the expected
            # accrual per remaining watering is readable, not guessed.
            fert_until = (int(raw.get("fertilized_until_day", -1))
                          if isinstance(raw, dict) else -1)
            base = int(st.yield_units)
            if isinstance(raw, dict):
                base = max(base, int(raw.get("yield_units", 0)))
            scheduled = _plant_schedule(crop)
            spec = CROP_RULES[crop]
            if spec["ongoing"]:
                steps = [(age, 1) for age in scheduled if age >= real_age]
                # The carried units ride the FIRST remaining step (or today
                # when every step is past and the tile still holds them).
                first = steps[0][0] if steps else (scheduled[-1] + 1)
                carried_at = steps[0][0] if steps else None
            else:
                # One-shot: the window's start is the first collectable day.
                # Everything the tile carries is collectable from there on,
                # today included — the event fires at max(start, today) and
                # carries the base; nothing else is guaranteed (an
                # unwatered tile yields 0, so future accrual is a hint).
                start = scheduled[0]
                carried_age = max(start, real_age)
                steps = [(carried_age, base)]
                carried_at = carried_age if base > 0 else None
            for age, units in steps:
                event_day = day + (age - real_age)
                if not (0 <= event_day < horizon):
                    continue
                # A production day accrues 1 unwatered; their fertiliser
                # (a public fact, if active over that day) makes it 2.
                hint = 2 if fert_until >= event_day else 1
                if age == carried_at:
                    units = base
                    if spec["ongoing"]:
                        # The estimate adds the step's own fert-aware accrual
                        # ON TOP of the carried units: a scheduled production
                        # lands that day and coexists with what is held.
                        hint = base + hint
                    # else one-shot: growth is watering-driven, not scheduled
                    # — the past waterings are already IN `base`, and future
                    # ones are the owner's choice. hint stays >= base via
                    # `max(units, hint)` below.
                events.append(RivalHarvestEvent(
                    day=event_day, good=crop, units=units, x=x, y=y,
                    units_hint=max(units, hint)))
            if spec["ongoing"] and not steps and base > 0:
                # An ongoing tile past its last scheduled production still
                # holds its carried units: collectable from today.
                events.append(RivalHarvestEvent(
                    day=day, good=crop, units=base, x=x, y=y,
                    units_hint=base))
            # `crop_last_day` is the tile's death, not a payout: the
            # calendar's authority ends there, nothing accrues past the
            # schedule — the floor stays valid only while the tile lives.
            _ = crop_last_day(st.crop)
    return events


def supply_curve(obs: dict, goods: tuple[str, ...], horizon: int = 30,
                 *, expected: bool = False) -> np.ndarray:
    """`(horizon, n_goods)` units/day the rival's planted tiles schedule.

    The FLOOR by default (`expected=False`): a guarantee, usable as "how
    long do we have / how many units at least". `expected=True` sums the
    fertilised hints instead — the estimate their own board implies, not
    a bound.
    """
    out = np.zeros((max(1, int(horizon)), len(goods)), dtype=np.float64)
    ix = {g: i for i, g in enumerate(goods)}
    for ev in harvest_events(obs, horizon=horizon):
        if ev.good in ix and ev.day < out.shape[0]:
            out[ev.day, ix[ev.good]] += (ev.units_hint if expected
                                         else ev.units)
    return out
