"""Guards for the Wentges smoothing knob (#87 follow-up sweep).

The smoothing parameter changes HOW FAST the certificate arrives, never
WHAT it certifies — pinned here as: the same objective and bound at every
alpha, and the default alpha=0.7 cutting rounds on the cold day-0 board.

Run:  .venv/bin/python -m tests.test_smoothing   (also under pytest)
"""

from __future__ import annotations

from offline_lab.kaggle_env import new_environment
from agent.planner import master as M
from agent.planner.inputs import load_contractor


def test_smoothing_never_changes_the_certified_answer() -> None:
    """Every alpha certifies the same objective and bound — smoothing is
    a speed knob, not an answer knob."""
    env = new_environment()
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    supply = M.supply_from_obs(obs)
    answers = {}
    for alpha in (0.0, 0.5, 0.7):
        r = M.equilibrate(object(), obs, contractor, supply,
                          iter_cap=200, smoothing=alpha)
        assert r.certified, f"alpha={alpha} did not certify: {r.stopped}"
        answers[alpha] = (round(r.objective, 4), round(r.bound, 4))
    assert answers[0.0] == answers[0.5] == answers[0.7], answers


def test_alpha_seven_cuts_rounds_on_the_cold_board() -> None:
    """The sweep's pick: alpha 0.7 reaches the certificate in fewer rounds
    than 0.0 (measured 55 vs 75 on the day-0 board)."""
    env = new_environment()
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    supply = M.supply_from_obs(obs)
    r0 = M.equilibrate(object(), obs, contractor, supply,
                       iter_cap=200, smoothing=0.0)
    r7 = M.equilibrate(object(), obs, contractor, supply,
                       iter_cap=200, smoothing=0.7)
    assert r7.rounds <= r0.rounds, (
        f"alpha 0.7 took {r7.rounds} rounds vs {r0.rounds} at 0.0: the "
        "sweep's pick no longer holds — re-measure")


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
    print(f"{len(tests) - failures}/{len(tests)} smoothing checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
