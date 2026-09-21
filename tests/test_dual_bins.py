"""Guards for the dual goods' buy bin (#65 follow-up): a dual good's last
bin holds the seat's net BUY counts, not a sell — so a sell-volume query
must never read it, a buy-volume query must never read the sell bins, and
a sell-only good has no buy side at all.

The defect this pins: `expected_sell('WHEAT', …, activity=1)` answered
5.48 units/turn for a rival whose row was 99% BUY_PRODUCT counts — a
silent-on-sells, buying-feed rival (the PASS shape) looked like a heavy
seller. Run:  .venv/bin/python -m tests.test_dual_bins   (also pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.opponent import OpponentModel
from agent.world.model import DUAL


def _model() -> OpponentModel:
    return OpponentModel(pretrained=True)


def test_a_dual_goods_buy_bin_is_not_a_sell_volume() -> None:
    """The artifact's (WHEAT, day=1, bucket=0, act=1) row is 99% buy counts;
    expected_sell must answer ~0 there, expected_buy carries them."""
    m = _model()
    sell = m.expected_sell("WHEAT", 240, 30, activity=1)
    buy = m.expected_buy("WHEAT", 240, 30, activity=1)
    assert sell < 1.0, (
        f"expected_sell(silent WHEAT) = {sell:.3f}: the buy bin leaked into "
        "the sell volume — a silent-but-buying rival looks like a seller")
    assert buy > 0.0, "expected_buy found nothing in the buy bin"


def test_the_two_volumes_are_different_numbers() -> None:
    """`expected_sell` and `expected_buy` must not be the same query."""
    m = _model()
    for act in (None, 1, 3):
        s = m.expected_sell("WHEAT", 240, 30, activity=act)
        b = m.expected_buy("WHEAT", 240, 30, activity=act)
        assert s != b, f"sell == buy ({s}) at activity={act}: no routing"


def test_a_sell_only_good_has_no_buy_side() -> None:
    """MILK's bins are all sales; expected_buy is 0 by construction."""
    m = _model()
    assert "MILK" not in DUAL
    assert m.expected_buy("MILK", 240, 30) == 0.0
    assert m.expected_sell("MILK", 240, 30) > 0.0


def test_the_mean_volume_router_writes_the_bins_it_names() -> None:
    """Unit shape: the router zeroes exactly the bins it says it does."""
    m = _model()
    counts = np.array([0.0, 83.0, 188.0, 33702.0, 500.0])
    qty = np.array([0.0, 83.0, 564.0, 300000.0, 4000.0])
    sell_mean = m._mean_volume(counts, qty, "WHEAT", buy=False)
    buy_mean = m._mean_volume(counts, qty, "WHEAT", buy=True)
    assert sell_mean[m.BUY_BIN] == 0.0 and sell_mean[:m.BUY_BIN].sum() > 0
    assert buy_mean[:m.BUY_BIN].sum() == 0.0 and buy_mean[m.BUY_BIN] > 0.0
    milk_sell = m._mean_volume(counts, qty, "MILK", buy=False)
    assert (milk_sell == np.where(counts > 0, qty / np.maximum(counts, 1),
                                  0.0)).all()
    assert m._mean_volume(counts, qty, "MILK", buy=True).sum() == 0.0


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
    print(f"{len(tests) - failures}/{len(tests)} dual-bin checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
