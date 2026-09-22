"""The Wentges smoothing sweep on the exact pricer (#87 follow-up).

`colgen.generate`'s smoothing knob is DEFAULT OFF, and the comment block
in `colgen.py:453-485` says its old rejection (alpha 0.3/0.5 bought a
bound below the objective) predated the exact-pricer fix — "the knob
stays off until it is re-measured, which is the next thing to try".

This bench is that re-measurement. It wires `smoothing=alpha` through
`equilibrate` (the one-line hole this bench found: `equilibrate` had no
alpha parameter at all), runs the day-0 board at alpha in
{0.0, 0.3, 0.5, 0.7} and reports, per alpha: rounds to certificate,
objective, bound, gap, wall time. The claim under test: on the exact
pricer, smoothing either cuts ROUNDS (the win — fewer sweeps and LPs per
hour-0 solve) or it does not, and the table decides.

Run:  .venv/bin/python -m bench.bench_wentges
"""

from __future__ import annotations

import time

from offline_lab.kaggle_env import new_environment
from agent.planner import master as M
from agent.planner.inputs import load_contractor


def main() -> int:
    env = new_environment()
    obs = env.state[0].observation
    contractor = load_contractor(days=20)
    supply = M.supply_from_obs(obs)

    print("alpha | rounds | certified | objective | bound | gap% | wall ms")
    print("------+--------+-----------+-----------+-------+------ --------")
    for alpha in (0.0, 0.3, 0.5, 0.7):
        t0 = time.perf_counter()
        result = M.equilibrate(object(), obs, contractor, supply,
                               iter_cap=m_cfg_rounds(),
                               smoothing=alpha)
        wall = (time.perf_counter() - t0) * 1000
        gap = result.gap * 100.0
        print(f"{alpha:5.1f} | {result.rounds:6d} | {str(result.certified):>9} "
              f"| {result.objective:9,.0f} | {result.bound:5,.0f} | {gap:4.3f} "
              f"| {wall:6.0f}")
    return 0


def m_cfg_rounds() -> int:
    """The config's master_rounds (200), read without a Manager."""
    from agent.config import Config
    return Config.load().master_rounds


if __name__ == "__main__":
    raise SystemExit(main())
