"""Guards for the land plan's pure functions (agent/planner/land_plan.py).

R007: each guard was broken and seen red before it was trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner.land_plan import (FREE_QUADRANT, cash_cost, cash_rhs_reduction,
                                     cells_open_by_day, held_fraction, purchase_order,
                                     quadrant_cells, round_plan, smooth, tiles_open)


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


def test_the_quadrants_tile_the_board_and_match_the_forced_order() -> None:
    """The four quadrants are disjoint and cover a 10x10 board, each with 25 cells, and the
    free one is NW — the engine's own layout (rules.py:110)."""
    size = 10
    seen: set[tuple[int, int]] = set()
    for q in (FREE_QUADRANT, *("NE", "SW", "SE")):
        cells = quadrant_cells(size, q)
        assert len(cells) == 25, f"{q}: {len(cells)} cells"
        assert not (set(cells) & seen), f"{q} overlaps another quadrant"
        seen |= set(cells)
    assert len(seen) == size * size, "the quadrants do not cover the board"
    # NW is the top-left half-block, NE the top-right one: cells are (x, y).
    assert (0, 0) in quadrant_cells(size, FREE_QUADRANT)
    assert (size - 1, 0) in quadrant_cells(size, "NE")
    assert (0, size - 1) in quadrant_cells(size, "SW")
    assert (size - 1, size - 1) in quadrant_cells(size, "SE")


def test_an_odd_board_or_unknown_quadrant_is_refused() -> None:
    """A wrong board size would silently price the wrong tiles, so it raises."""
    for bad in (9, 11):
        try:
            quadrant_cells(bad, "NE")
        except ValueError:
            pass
        else:
            raise AssertionError(f"board_size {bad} was accepted")
    try:
        quadrant_cells(10, "NOPE")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown quadrant was accepted")


def test_the_next_purchase_follows_the_forced_order_and_ends() -> None:
    """F042: NE, SW, SE at 1000, 2000, 4000 — and a fourth purchase does not exist."""
    assert purchase_order(0) == ("NE", 1000)
    assert purchase_order(1) == ("SW", 2000)
    assert purchase_order(2) == ("SE", 4000)
    assert purchase_order(3) is None
    try:
        purchase_order(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative purchase count was accepted")


def test_the_cash_ceiling_drops_on_the_buy_day_and_stays_down() -> None:
    """The cash rows are cumulative, so the reduction is cumulative too: zero before the buy,
    the whole price on the buy day, and never less than the price afterwards."""
    s = np.zeros(6)
    s[2] = 1.0
    drop = cash_rhs_reduction(s, 1000)
    assert drop[:2].sum() == 0.0, "the ceiling dropped before the purchase"
    assert drop[2] == 1000.0, "the ceiling must drop on the buy day"
    assert (drop[2:] == 1000.0).all(), "the ceiling came back up"


def test_cells_open_in_the_forced_order_and_on_the_buy_day() -> None:
    """The cells handed over are the quadrant's own, they open on the buy day, and a soft
    curve opens them in board order — never more than the quadrant holds."""
    cells = quadrant_cells(10, "NE")
    s = np.zeros(5)
    s[1] = 1.0
    per_day = list(cells_open_by_day(s, cells))
    assert len(per_day) == 5
    assert per_day[0] == [], "cells open before the purchase"
    assert per_day[1] == cells, "the buy day must have the whole quadrant open"
    assert per_day[4] == cells

    half = np.zeros(5)
    half[0] = 0.5
    opening = list(cells_open_by_day(half, cells))
    assert len(opening[0]) == 13 or len(opening[0]) == 12, len(opening[0])
    assert len(opening[0]) < len(cells), "half a buy opened everything"


def test_held_fraction_measures_what_was_paid_for() -> None:
    """Any hard plan holds the whole quadrant; a weak curve holds only its part."""
    hard = np.zeros(8)
    hard[3] = 1.0
    assert held_fraction(hard) == 1.0
    assert abs(held_fraction(np.full(8, 0.25)) - 1.0) < 1e-12   # 8 * 0.25 = 2 -> clipped
    assert abs(held_fraction(np.full(8, 0.05)) - 0.4) < 1e-12


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
