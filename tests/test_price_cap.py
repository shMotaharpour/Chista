"""Guards for the two things the sale price path does: the offset and the ceiling.

Both changes sit behind their own gate, so each guard is written as the claim the
change makes and not as a value it happened to produce: the offset's quote is never
ABOVE the raw forecast (it prices the un-drained market), and the archive's ceiling
never raises a price, only lowers one. R007: break either claim and the guard goes
red -- the drills are in the PR that added this file.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from agent.belief.market import forecast
from agent.config import Config
from agent.planner.colgen import RESOURCE_ID
from agent.planner.master import _archive_day_caps, _product_price_path
from offline_lab.fast_sim import FastSim

#: The horizon these guards price: the artifact covers the season, this only needs to
#: reach the days where a drain-only path and the offset disagree.
DAYS = 8

#: A tolerance for float arithmetic, not for a claim.
EPS = 1e-9


@pytest.fixture(scope="module")
def board():
    """A real day-0 board and its forecast, once for the module."""
    sim = FastSim({"episodeSteps": 24 * DAYS + 6, "seed": 33, "farmHandCostMult": 1})
    obs = sim.observations(copy_state=False)[0]
    return obs, forecast(obs, days=DAYS), np.zeros((DAYS, 18))


def _path(board, **kwargs):
    obs, fc, flat = board
    cfg = replace(Config(), **kwargs)
    path, _why = _product_price_path(obs, DAYS, flat, forecast_obj=fc, cfg=cfg)
    return path


def test_the_offset_is_the_quote_the_plan_is_priced_with(board):
    """The gate must change the path, and never upward.

    A drain-only path rises because the town eats the stock; the offset prices the
    market our own sale lands in, so it can only take the price down. R007: delete the
    line that hands the block's quote to the path and this goes red on the first
    assertion; make the offset price above the raw forecast and it goes red on the
    second.
    """
    raw = _path(board, price_supply_rounds=0)
    off = _path(board, price_supply_rounds=1, sell_lot_default=15.0)
    moved = [item for item, rid in RESOURCE_ID.items()
             if rid in range(raw.shape[1]) and np.any(np.abs(off[:, rid] - raw[:, rid]) > EPS)]
    assert moved, "the offset gate changed nothing: its quote is being thrown away"
    assert np.all(off <= raw + EPS), (
        "the offset priced above the raw forecast: " +
        ", ".join(item for item, rid in RESOURCE_ID.items()
                  if rid in range(raw.shape[1]) and np.any(off[:, rid] > raw[:, rid] + EPS)))


def test_the_archive_ceiling_never_raises_a_price(board):
    """The cap lowers the far days to what the engine really paid, and nothing else.

    R007: drop the `min(...)` from the cap block and this goes red.
    """
    plain = _path(board, price_cap_from_archive=False)
    capped = _path(board, price_cap_from_archive=True)
    caps = _archive_day_caps()
    assert np.all(capped <= plain + EPS), "the cap raised a price"
    lowered = []
    for item, rid in RESOURCE_ID.items():
        if item not in caps or rid not in range(capped.shape[1]):
            continue
        ceiling = np.asarray(caps[item][:DAYS], dtype=np.float64)
        finite = np.isfinite(ceiling)
        assert np.all(capped[:DAYS, rid][finite] <= ceiling[finite] + EPS), (
            f"{item} priced above its own day's ceiling")
        if np.any(np.abs(capped[:, rid] - plain[:, rid]) > EPS):
            lowered.append(item)
    assert lowered, "the cap changed nothing: it is not being applied"


def test_the_cap_reads_no_artifact_when_the_gate_is_off(board, monkeypatch):
    """Off means untouched, not merely equal: the loader must not be consulted.

    The block lives inside the path's own try/except, so a loader that RAISES would
    only make it degrade to the flat stand-in and the path would still look finite --
    the guard has to make the loader's ANSWER visible instead. It answers with a
    ceiling of zero for every good, which would flatten the whole path; with the gate
    off the path must not move at all.

    R007: read the archive outside the gate and this goes red.
    """
    import agent.planner.master as master

    monkeypatch.setattr(master, "_archive_day_caps",
                        lambda: {item: np.zeros(DAYS) for item in RESOURCE_ID})
    plain = _path(board, price_cap_from_archive=False)
    assert np.isfinite(plain).all() and float(np.max(plain)) > 0.0, (
        "a ceiling was applied with the gate off: the archive was read anyway")
