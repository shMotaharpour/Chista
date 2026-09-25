"""Guards for the land plan's pure functions (agent/planner/land_plan.py).

R007: each guard was broken and seen red before it was trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner.land_plan import cash_cost, round_plan, smooth, tiles_open


def test_a_single_spike_is_the_hard_plan_buy_on_that_day() -> None:
    """The hard plan 'buy on day d' opens the 25 tiles ON day d — the engine settles
    BUY_LAND atomically at the buy turn (kaggriculture.py:712), so the tiles are ordinary
    empty tiles from that day on."""
    s = np.zeros(10)
    s[4] = 1.0
    open_tiles = tiles_open(s)
    assert open_tiles[:4].sum() == 0.0, "tiles open before the purchase"
    assert open_tiles[4] == 25.0, "the purchase's own day must have the tiles open"
    assert (open_tiles[4:] == 25.0).all()


def test_the_purse_pays_on_the_purchase_day() -> None:
    """cash_cost lands the whole price on the day the buy settles."""
    s = np.zeros(10)
    s[4] = 1.0
    pay = cash_cost(s, 1000)
    assert pay[:4].sum() == 0.0
    assert pay[4] == 1000.0
    assert pay[5:].sum() == 0.0


def test_a_gradual_curve_is_a_gradual_opening_and_never_over_mints() -> None:
    """The soft curve splits the buy across days, and can never mint more than 25 tiles
    even when `s` sums past 1."""
    s = np.full(10, 0.4)          # sums to 4.0 — far past one quadrant
    open_tiles = tiles_open(s)
    assert (open_tiles <= 25.0 + 1e-12).all(), "more tiles than the quadrant has"
    assert open_tiles[-1] == 25.0
    assert open_tiles[0] == 10.0   # 0.4 bought by day 0's planning point


def test_smoothing_is_monotone_and_saturates() -> None:
    """A larger raw value never means a smaller buy, and extremes saturate cleanly."""
    z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    out = smooth(z)
    assert (np.diff(out) > 0).all(), "not monotone"
    assert abs(out[2] - 0.5) < 1e-9        # sigmoid's centre
    assert out[0] < 0.05 and out[-1] > 0.95


def test_rounding_names_the_peak_day_or_nothing() -> None:
    """The rounded plan is the peak day when the peak is meaningful, else no buy."""
    s = np.zeros(10)
    s[6] = 0.9
    assert round_plan(s) == 6
    assert round_plan(np.full(10, 0.3)) is None, "a flat weak curve is no decision"
    assert round_plan(np.zeros(3)) is None


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all land_plan guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
