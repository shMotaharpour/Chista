"""The day's market queue: what it buys, when, and in what order.

Every failure this suite catches is silent on the board. The engine refuses a
bad market order without a word (F047), so a day that buys the wrong thing, or
the right thing one turn too late, reports success and plants nothing.

R007: each guard was broken and seen red before it was trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.planner import market as K
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.world.rules import TURNS_PER_DAY

QUOTES = {"WHEAT": 25, "FERTILIZER": 100}
FEED_CHAIN = ("BUILD_PASTURE", "PLACE", "FEED", "CARE")
PLANT_CHAIN = ("PLANT", "WATER")


def _chain(cell, ops, entity):
    return (cell, chain_ops(chain_id_of(ops)), entity)


def test_the_settle_order_has_exactly_one_definition():
    """F032 is a design decision, and a second copy is a second decision.

    It was written four times: `planner/market.py`'s `QUEUE_RANK`,
    `planner/repair.py`'s `_QUEUE_RANK`, a fourth inline dict in
    `market_layer.py` — a fourth, retired with that env-switched
    circuit — whose own docstring asked to delegate back once the planner
    imported cleanly — and the belief hook it is passed through. The
    engine settles strictly by queue index, so two tables can disagree and one
    of them would be wrong without saying so.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    table = re.compile(r"""["']BUY_LAND["']\s*:\s*0""")
    copies = [str(path.relative_to(root))
              for path in (root / "agent").rglob("*.py")
              if path != root / "agent/world/action_rules.py"
              and table.search(path.read_text())]
    assert not copies, f"a second copy of F032's order table: {copies}"

    from agent.world.action_rules import SETTLE_RANK, SETTLE_RANK_DEFAULT
    assert SETTLE_RANK == {"BUY_LAND": 0, "SELL": 1, "HIRE": 2}
    assert SETTLE_RANK_DEFAULT == 3


def test_wheat_the_seed_and_wheat_the_product_are_different_purchases():
    """One name, two goods, two orders — and buying the wrong one is silent.

    A PLANT of wheat needs `BUY_SEED WHEAT` at the rule table's 10. A FEED
    needs `BUY_PRODUCT WHEAT` at the market's quote. Counting by the good
    alone emitted one BUY_SEED for both, so the FEED picked up a product that
    was never bought and the engine dropped the pickup without a word (F047).
    """
    chains = (_chain((3, 3), FEED_CHAIN, "COW"),
              _chain((1, 1), PLANT_CHAIN, "WHEAT"))
    assert K.needs(chains) == {("COW", "ANIMAL"): 1, ("WHEAT", "PRODUCT"): 1,
                               ("WHEAT", "SEED"): 1}
    orders, bill = K.buy_orders(chains, {}, {}, QUOTES)
    kinds = {(o[0], o[1]) for o in orders}
    assert ("BUY_SEED", "WHEAT") in kinds and ("BUY_PRODUCT", "WHEAT") in kinds
    assert bill == 400 + 25 + 10


def test_what_the_farm_already_holds_is_not_bought_again():
    """Seeds come out of the purse, animals and products out of the shed."""
    chains = (_chain((1, 1), PLANT_CHAIN, "WHEAT"),)
    assert K.buy_orders(chains, {"WHEAT": 1}, {}, QUOTES) == ([], 0)
    # ...and the shed does not satisfy a seed: they are different stores (F001).
    orders, _bill = K.buy_orders(chains, {}, {"WHEAT": 9}, QUOTES)
    assert orders == [["BUY_SEED", "WHEAT", 1]]


def test_a_good_the_farm_must_buy_is_not_in_the_shed_until_hour_one():
    """Units act before market inside a turn (F030), so a turn-0 buy lands at 1.

    A good left out of `available` is available at hour 0 as far as wsr is
    concerned. That is how a day came back with `PICKUP COW` in turn 0 beside
    the `BUY_ANIMAL COW` that pays for it.
    """
    chains = (_chain((3, 3), FEED_CHAIN, "COW"),)
    assert K.availability(chains, {}, {}) == {"COW": 1, "WHEAT": 1}
    # Already in the shed: there from the first hour.
    assert K.availability(chains, {}, {"COW": 1, "WHEAT": 4}) == \
        {"COW": 0, "WHEAT": 0}


def test_what_an_op_eats_is_named_even_though_it_is_not_the_entity():
    """FEED eats wheat and FERTILIZE eats fertilizer; neither is the entity."""
    assert K.OP_CONSUMES == {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}
    chains = (_chain((3, 3), FEED_CHAIN, "COW"),)
    assert "WHEAT" in K.availability(chains, {}, {})


def test_the_queue_settles_in_the_engines_order_and_respects_the_cap():
    """F032: land, sells, hires, purchases. F031: ten per TURN, not per day."""
    # A HIRE inside the sell rows, deliberately out of order: only a row that
    # is actually SORTED puts the sells in front of it.
    # The HIRE sits in the MIDDLE on purpose: reversing this row also puts a
    # SELL first, so a fixture with it at either end cannot tell a sort from a
    # reverse. Sorted it is SELL, SELL, HIRE; reversed it is SELL, HIRE, SELL.
    sells = [[["SELL", "MILK", 2], ["HIRE"], ["SELL", "EGG", 1]]] + [[]] * 23
    hires = [["HIRE"]] * 3
    buys = [["BUY_SEED", "WHEAT", 1]] * 9
    rows, dropped = K.merge(sells, hires, buys, cap=10)

    assert len(rows) == TURNS_PER_DAY
    assert [o[0] for o in rows[0]][:3] == ["SELL", "SELL", "HIRE"], (
        f"the row is not in the engine's settle order: {rows[0]}")
    assert all(len(r) <= 10 for r in rows), "a row went over the per-turn cap"
    placed = sum(1 for r in rows for o in r if o[0] == "BUY_SEED")
    assert placed == len(buys) - len(dropped)


def test_an_order_with_no_room_is_reported_not_forgotten():
    """A queue that quietly drops an order is a plan that silently does less."""
    buys = [["BUY_SEED", "WHEAT", 1]] * 5
    rows, dropped = K.merge([], [], buys, cap=2, turns=2)
    assert sum(len(r) for r in rows) == 4
    assert len(dropped) == 1 and dropped[0][0] == "BUY_SEED"


def test_an_animal_in_the_shed_does_not_take_the_sell_queue_down():
    """#77: `sellable()` returns shed items, `ladder._IX` knows products only.

    Ten raises in one season and the seat stopped at day 20 before the day
    layer started handing belief an input it can price.
    """
    from offline_lab.kaggle_env import new_environment

    env = new_environment()
    env.reset(2)
    obs = dict(env.state[0].observation)
    obs["private"] = dict(obs.get("private", {}))
    obs["private"]["shed"] = dict(obs["private"].get("shed", {}))
    obs["private"]["shed"]["GOOSE"] = 1
    obs["private"]["shed"]["MILK"] = 3

    rows = K.sell_rows(obs, harvest_expected=0, cash_needed=0.0,
                       config=env.configuration)
    assert isinstance(rows, list)

    # A shed that holds only products is handed over untouched — the trim is
    # a guard for an input belief cannot price, not a rewrite of every one.
    clean = {"private": {"shed": {"MILK": 2, "EGG": 1}}}
    assert K._sellable_obs(clean) is clean

    # And the real observation always takes the trim: the engine lists every
    # shed item it COULD hold, animals included, at zero. So the guard is on
    # every path, not only on a farm that owns livestock — which is why the
    # KeyError was not rarer than it was.
    real = env.state[0].observation
    assert K._sellable_obs(real) is not real
    # ...and the products are still there to be sold, not trimmed with them.
    # The shed carries a zero for every product it could hold, so the check is
    # which KEYS survive, not which counts.
    from agent.world.model import ANIMALS, PRODUCTS
    trimmed = K._sellable_obs(obs)["private"]["shed"]
    assert set(trimmed) == set(PRODUCTS), sorted(trimmed)
    assert not set(trimmed) & set(ANIMALS)
    assert trimmed["MILK"] == 3, "the trim ate a product it was meant to keep"


def test_master_lp_sells_are_emitted_in_market_orders():
    """Sales decided by the master LP are emitted in the market queue (#147)."""
    from offline_lab.kaggle_env import new_environment
    env = new_environment()
    env.reset(2)
    obs = dict(env.state[0].observation)
    obs["private"] = dict(obs.get("private", {}))
    obs["private"]["shed"] = {"CARROT": 10}

    # Without master_sells: baseline shed policy hoards (0 CARROT sales)
    rows_no_master = K.sell_rows(obs, harvest_expected=0, cash_needed=0.0)
    sells_no_master = sum(o[2] for r in rows_no_master for o in r if o and o[0] == "SELL" and o[1] == "CARROT")
    assert sells_no_master == 0, f"baseline should have hoarded, but sold: {sells_no_master}"

    # With master_sells: the LP's decision to sell 5 carrots is executed
    rows_with_master = K.sell_rows(obs, harvest_expected=0, cash_needed=0.0,
                                   master_sells={"CARROT": 5})
    sells_with_master = sum(o[2] for r in rows_with_master for o in r if o and o[0] == "SELL" and o[1] == "CARROT")
    assert sells_with_master == 5, f"expected 5 carrots sold, got {sells_with_master}"

