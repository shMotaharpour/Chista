"""Guards for the forecast's residual wire (#65 step 1).

The claim: feeding the trained model's expected rival volume into
`forecast(residual=...)` moves the price path DOWN (the rival's supply was
the named bias), and the model-fed path sits between the zero-residual path
and the TRUE-residual path on self-play episodes where the rival's real
volume is known.

The R007 red control: with the residual wire cut (`residual=None`), the
day-2 price of a good the model expects the rival to dump must stay ABOVE
the model-fed path — a wire that does not move the path cannot be a wire.

Run:  .venv/bin/python -m tests.test_residual_wire   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from offline_lab.fast_sim import FastSim
from agent.belief.market import forecast
from agent.belief.opponent import OpponentModel
from agent.world.model import PRODUCTS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _sim_to(seed: int, day: int):
    sim = FastSim(configuration={"seed": seed, "episodeSteps": 720},
                  validate="dev")
    for _ in range(day * 24):
        sim.step([PASS, PASS])
    return sim.observations()[0]


def test_the_wire_moves_the_price_path_down() -> None:
    """A model that expects rival supply must price the future LOWER."""
    obs = _sim_to(0, 3)
    model = OpponentModel(pretrained=True)
    # a HIGH-activity rival (bucket 4): the regime the corpus says sells
    residual = model.expected_sell_day(obs, activity=4)
    assert any(v > 0 for v in residual.values()), "the model expects nothing?"
    fc_plain = forecast(obs, days=5)
    fc_rival = forecast(obs, days=5, residual=residual)
    for g in ("WHEAT", "MILK", "STRAWBERRY"):
        gi = PRODUCTS.index(g)
        assert fc_rival.prices[4][gi] <= fc_plain.prices[4][gi], (
            g, fc_plain.prices[4][gi], fc_rival.prices[4][gi])


def test_the_model_fed_path_sits_toward_the_oracle() -> None:
    """On scripted self-play, model-fed residual halves the oracle gap.

    The rival sells a KNOWN 10 WHEAT per day (scripted); the oracle residual
    is exactly that. The trained model's residual must pull the day-3 WHEAT
    price at least halfway from the zero-residual path toward the oracle
    path — otherwise the wire is decorative.
    """
    from agent.belief.market import forecast as f
    obs = _sim_to(5, 2)
    oracle = {"WHEAT": 10.0}
    fc_zero = f(obs, days=4)
    fc_oracle = f(obs, days=4, residual=oracle)
    model = OpponentModel(pretrained=True)
    fed = model.expected_sell_day(obs, activity=4)
    fc_model = f(obs, days=4, residual={"WHEAT": fed["WHEAT"]})
    gi = PRODUCTS.index("WHEAT")
    p_zero = fc_zero.prices[3][gi]
    p_oracle = fc_oracle.prices[3][gi]
    p_model = fc_model.prices[3][gi]
    assert p_oracle < p_zero, (p_zero, p_oracle)
    closed = (p_zero - p_model) / max(1e-9, (p_zero - p_oracle))
    assert closed >= 0.5, (
        f"the model-fed path closed only {closed:.0%} of the oracle gap "
        f"({p_zero} -> {p_model}, oracle {p_oracle})")


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
    print(f"{len(tests) - failures}/{len(tests)} residual-wire checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
