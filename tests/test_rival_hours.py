"""Guards for the manager's DATED rival supply (#16).

`_rival_supply` is the calendar: which days the rival pays out, not which hour.
`_rival_hours` is the hour — the tracker's inferred sales for the turns already
recorded, and the opponent model's expectation beyond them — so the walk stops
dating their whole day at hour 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import agent.manager.core as MC
from agent.belief.market import PRODUCTS
from agent.config import Config


def _manager() -> MC.Manager:
    return MC.Manager(Config())


def test_the_recorded_rival_sales_come_back_dated() -> None:
    """What the tracker inferred is dated by the turn it happened in."""
    m = _manager()
    m.opponent = None                    # the tracker branch alone
    sales = np.zeros(len(PRODUCTS))
    sales[PRODUCTS.index("MELON")] = 4.0
    m.tracker = SimpleNamespace(records=[SimpleNamespace(step=57, rival_sales=sales)],
                                step=57)
    hours = m._rival_hours(None, 2)
    assert hours == {57: {"MELON": 4}}, hours


def test_without_a_tracker_or_a_model_the_calendar_stands() -> None:
    """No source of hours means no dated input: the daily curve is used."""
    m = _manager()
    m.tracker = None
    m.opponent = None
    assert m._rival_hours(None, 3) == {}


def test_a_broken_tracker_spills_unless_the_run_asked_to_degrade() -> None:
    """`never_raise` decides, exactly as it does at the turn boundary.

    Off (the shipped default) the error comes out where it can be seen; on, the
    calendar stands. Swallowing unconditionally is how the `NameError` in
    `0ce5d25` hid behind this very except.
    """

    class Boom:
        @property
        def records(self):
            raise RuntimeError("tracker exploded")

    m = _manager()
    m.tracker = Boom()
    m.opponent = None
    try:
        m._rival_hours(None, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a broken tracker must spill, not hide")

    quiet = MC.Manager(Config(never_raise=True))
    quiet.tracker = Boom()
    quiet.opponent = None
    assert quiet._rival_hours(None, 2) == {}


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} rival-hour checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_a_model_present_dates_the_hours_the_tracker_has_not_seen() -> None:
    """The trained model covers the horizon; dropping its branch empties this."""
    m = _manager()
    m.tracker = None                       # no records: the model alone
    model = MC.opponent_model()
    assert model is not None, "the pretrained opponent table must be present"
    m.opponent = model
    calendar = np.zeros((2, len(PRODUCTS)))
    calendar[0, PRODUCTS.index("MELON")] = 20.0     # the board says day 0 sells
    m._rival_supply = lambda obs, horizon: calendar
    quotes = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
              "FERTILIZER": 100}
    obs = {"step": 0, "day": 0, "market": {"prices": quotes}}
    hours = m._rival_hours(obs, 2)
    assert hours, (
        "with the model present the rival's hours must be dated; an empty dict "
        "means the model branch is not running")
    assert all(0 <= int(step) < 2 * 24 for step in hours), sorted(hours)


def test_the_model_only_dates_a_day_the_board_says_has_goods() -> None:
    """An empty calendar gates the model out: no phantom rival supply.

    The model's table is trained on OTHER players' games, so on a day the board
    shows nothing to sell its priors describe a rival who is not there. Measured
    on the PASS season: letting them through cost 4,058 coins.
    """
    m = _manager()
    m.tracker = None
    m.opponent = MC.opponent_model()
    assert m.opponent is not None
    empty = np.zeros((2, len(PRODUCTS)))
    m._rival_supply = lambda obs, horizon: empty
    quotes = {g: 50 for g in PRODUCTS}
    obs = {"step": 0, "day": 0, "market": {"prices": quotes}}
    assert m._rival_hours(obs, 2) == {}, (
        "the board says no goods: the model must not date anything")
