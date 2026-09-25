"""The land plan: how a purchase enters the manager's own planning.

The owner's design (this session): the manager sees the land purchase inside its own model, as
`tiles per day` — a soft decision per day, smoothed so a search can move it, rounded at the end
to "buy on day d". The engine settles `BUY_LAND` atomically at the buy turn and turns those
cells from "LOCKED" to None on the spot (kaggriculture.py:712-726), so the tiles are usable
from the hour after the purchase, i.e. the purchase's own day.

Part one — the soft buy curve and its two effects:

    tiles_open(s)     25 * clip(cumsum(s), 0, 1): a single spike at d is the hard plan
    cash_cost(s)      the price lands on the day the buy settles
    smooth(s_raw)     sigmoid squash to [0, 1], monotone, for the search
    round_plan(s)     the peak day, or None when the peak is not meaningful

Part two — the two objects a purchase changes in the master:

    cash_rhs_reduction(s, price)  the cash rows are CUMULATIVE against one purse
                                  (`Σ_{d'<=d} spend[d'] − Σ_{d'<d} earn[d'] ≤ money`,
                                  colgen.py:373), so a price paid on day d lowers the
                                  right-hand side of every row from d onward
    quadrant_cells(size, q)       the 25 cells a quadrant hands over
    cells_open_by_day(s, cells)   per day, the cells already open by that day's planning point
    purchase_order(bought)        which quadrant the next purchase buys, and its price (F042)

The forced order of quadrants stays an OUTER loop: one quadrant per plan, never a variable.
Board convention: `tiles[y][x]`, quadrants are the four half-blocks, `NW` is free from the start
and `LAND_ORDER` is ("NE", "SW", "SE") at `LAND_PRICES` (rules.py:110-114).

Everything here is a pure function, testable without the engine.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from agent.world.rules import LAND_ORDER, LAND_PRICES

#: The quadrant owned from the start (rules.py:110).
FREE_QUADRANT = "NW"


def tiles_open(s: np.ndarray) -> np.ndarray:
    """Tiles of the next quadrant open on each day, from the soft buy curve `s`.

    `s` is (days,) in [0, 1]: the buy intensity per day. The running sum is the fraction
    bought by that day's planning point, and the quadrant has 25 tiles. Values above 1 (or
    below 0) are clipped, so a sloppy curve cannot invent tiles.

    A single-spike `s` (1 at day d, 0 elsewhere) is the hard plan "buy on day d": the tiles
    are open from day d itself, matching the engine's atomic settlement.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    bought = np.clip(np.cumsum(s), 0.0, 1.0)
    return 25.0 * bought


def cash_cost(s: np.ndarray, price: int) -> np.ndarray:
    """What the purse pays on each day for this quadrant, row-ready for `cash_out`.

    The payment lands on the day the buy settles — the day `s` steps up. With the hard plan
    (a single spike at d) the whole price sits on day d.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return float(price) * s


def smooth(s_raw: np.ndarray) -> np.ndarray:
    """The soft version of a raw buy curve, for the search.

    A sigmoid squash into [0, 1]: monotone, so a larger raw value never means a smaller buy,
    and near-zero when the raw value is very negative (the search can pull a day's buy down
    without it vanishing).
    """
    z = np.asarray(s_raw, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def round_plan(s: np.ndarray) -> int | None:
    """The day to buy, from the soft curve — or None when the model says do not buy.

    The soft curve is a search device; the plan it names is the single day with the most buy
    in it, and only if that peak is meaningful (>= 0.5). Day 0 means "buy at the season's
    first planning point".
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    if s.size == 0 or float(s.max()) < 0.5:
        return None
    return int(np.argmax(s))


def quadrant_cells(board_size: int, quadrant: str) -> list[tuple[int, int]]:
    """The (x, y) cells of one quadrant, in board order (y outer, x inner).

    Raises for an odd board size, where the halves would not be equal, and for an unknown
    quadrant name: a silent wrong answer here prices the wrong 25 tiles.
    """
    if board_size % 2:
        raise ValueError(f"board_size {board_size} is not even: quadrants are half-blocks")
    if quadrant not in (*LAND_ORDER, FREE_QUADRANT):
        raise ValueError(f"unknown quadrant {quadrant!r}")
    half = board_size // 2
    x0 = 0 if quadrant in (FREE_QUADRANT, LAND_ORDER[1]) else half      # NW, SW | NE, SE
    y0 = 0 if quadrant in (FREE_QUADRANT, LAND_ORDER[0]) else half      # NW, NE | SW, SE
    return [(x0 + i, y0 + j) for j in range(half) for i in range(half)]


def purchase_order(bought: int) -> tuple[str, int] | None:
    """The quadrant the next purchase buys and its price, given how many are already bought.

    F042: quadrants open as a fixed prefix of `LAND_ORDER` at `LAND_PRICES`, never a gap. The
    fourth purchase does not exist, and `None` says so rather than inventing a price.
    """
    if bought < 0:
        raise ValueError(f"bought {bought} cannot be negative")
    if bought >= len(LAND_ORDER):
        return None
    return LAND_ORDER[bought], int(LAND_PRICES[bought])


def cash_rhs_reduction(s: np.ndarray, price: int) -> np.ndarray:
    """Per-day reduction of the cash rows' right-hand side, from a soft buy curve.

    The rows are cumulative against one purse, so the day the price is paid is the day the
    ceiling drops for every later row. Day d of the result is the price paid by the end of day
    d — the cumulative cost, not the per-day cost.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return np.cumsum(float(price) * s)


def cells_open_by_day(s: np.ndarray, cells: list[tuple[int, int]]) -> Iterator[list]:
    """For each day, the cells already open by that day's planning point.

    Yields one entry per day of `s`. A hard plan (one spike at d) opens all the cells on day d
    itself, matching the engine: the purchase settles at its turn and the cells are ordinary
    empty tiles from then on, so the same day can already work them.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    bought = np.clip(np.cumsum(s), 0.0, 1.0)
    total = len(cells)
    for day in range(s.size):
        k = int(round(bought[day] * total))
        yield cells[:k]


def held_fraction(s: np.ndarray) -> float:
    """How much of the quadrant the curve buys in total, clipped to [0, 1].

    The honest measure of "did this plan buy the land": 1.0 for any hard plan; for a soft curve
    it is the part actually paid for, which is what a search must push to a boundary before the
    plan means anything.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return float(min(1.0, float(s.sum())))


def with_quadrant_open(obs: dict, quadrant: str, price: int, player: int = 0) -> dict:
    """The observation as it would be if this farm had just bought `quadrant`.

    Deep-copied and edited at the level the day plan actually reads: that quadrant's "LOCKED"
    cells become None (empty), the quadrant joins `unlocked_quadrants`, and the price leaves
    the purse. Nothing else moves — so the plans compared against it differ in exactly the two
    things a purchase changes, which is what makes the comparison a valuation of the land
    rather than of a different board.

    Raises when the observation does not carry what this needs, instead of returning a board
    that would be silently wrong.
    """
    import copy

    farms = (obs or {}).get("farms") or []
    if len(farms) <= player:
        raise ValueError(f"observation has no farm for player {player}")
    out = copy.deepcopy(obs)
    farm = out["farms"][player]
    tiles = farm.get("tiles") or []
    if not tiles:
        raise ValueError("farm has no tiles to open")
    board = len(tiles)
    for x, y in quadrant_cells(board, quadrant):
        if tiles[y][x] == "LOCKED":
            tiles[y][x] = None
    unlocked = list(farm.get("unlocked_quadrants", []) or [])
    if quadrant not in unlocked:
        unlocked.append(quadrant)
    farm["unlocked_quadrants"] = unlocked
    farm["money"] = float(farm.get("money", 0.0)) - float(price)
    return out


def quadrant_is_locked(obs: dict, quadrant: str, player: int = 0) -> bool:
    """Whether this farm still has `quadrant` locked — the precondition for buying it.

    Reads the board rather than the quadrant list: the list is the engine's bookkeeping, and
    the cells are what the plan works.
    """
    farm = ((obs or {}).get("farms") or [{}])[player]
    tiles = farm.get("tiles") or []
    if not tiles:
        return False
    board = len(tiles)
    return all(tiles[y][x] == "LOCKED" for x, y in quadrant_cells(board, quadrant))


def quadrants_bought(obs: dict, player: int = 0) -> int:
    """How many quadrants this farm has bought: NW is free from the start (F042).

    Read from the observation rather than tracked in the manager, so the count survives a
    restart and cannot drift from the board the engine shows.
    """
    farms = (obs or {}).get("farms") or []
    farm = farms[player] if len(farms) > player else {}
    return max(0, len(farm.get("unlocked_quadrants", []) or []) - 1)
