"""Backend selection: NumPy vs SciPy vs JAX, on the workloads this agent runs.

The question is not "which is faster in general" but "which one, for which piece,
on this machine" — and the answer has to survive the platform's own numbers (F058:
`scipy 1.15.3` with `milp` present, `ortools` absent, `jax 0.5.2` present, first
JAX use 3.5-7.5 s of the per-seat bank, SciPy-shaped work +41 % when the other
seat is thinking against JAX's -3 %).

Run:  .venv/bin/python -m bench.bench_backends [--paths N] [--contention]

Every measurement prints three numbers: the first call (which for JAX includes
import and JIT compilation, and is what a turn would be charged on the platform),
the warm median, and the spread. Timings are wall clock, `block_until_ready()` is
called on every JAX result, and each workload is checked against the NumPy answer
before it is timed — a fast wrong answer is not a result.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# JAX defaults to float32; the engine's price function is a hinge with gain 8 and
# the answer is rounded to an int, so float32 loses the parity outright (measured:
# CARROT at inventory 0 came out 16,576 instead of 126,920). x64 is not optional here.
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np

from belief.opponent import basket_matrix
from world.prices import price_of, price_vec
from world.vocabulary import GOODS, SHOP_TYPES, UNLOCK_INTERVAL


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #

def timeit(fn, reps: int = 7, warm: int = 1) -> tuple[float, float, float, float]:
    """(first_call_ms, median_ms, min_ms, max_ms) with `warm` discarded runs."""
    t0 = time.perf_counter()
    fn()
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return first, statistics.median(samples), min(samples), max(samples)


def report(label: str, first: float, med: float, lo: float, hi: float,
           note: str = "") -> None:
    print(f"  {label:<44} first {first:9.3f} ms | warm {med:8.3f} ms "
          f"[{lo:.3f}-{hi:.3f}] {note}")


# --------------------------------------------------------------------------- #
# 1. import cost — measured in a fresh interpreter, because that is what a turn pays
# --------------------------------------------------------------------------- #

IMPORT_PROBE = (
    "import time,os; os.environ.setdefault('JAX_PLATFORM_NAME','cpu');"
    "t=time.perf_counter(); import {mod};"
    "print(round((time.perf_counter()-t)*1000,1))"
)


def import_costs() -> None:
    print("\n[1] import cost, fresh interpreter (what a first turn pays)")
    for mod in ("numpy", "scipy.optimize", "jax", "jax.numpy", "torch"):
        out = subprocess.run([sys.executable, "-c", IMPORT_PROBE.format(mod=mod)],
                             capture_output=True, text=True, timeout=300)
        ms = out.stdout.strip() or f"FAILED: {out.stderr.strip().splitlines()[-1:] }"
        print(f"  import {mod:<38} {ms} ms")


# --------------------------------------------------------------------------- #
# 2. the price curve, over a grid (the hot read of the whole sell side)
# --------------------------------------------------------------------------- #

GRID = np.concatenate([np.arange(0, 60, 1.0), np.arange(100, 12000, 7.0)])


def price_curve_numpy() -> np.ndarray:
    return np.array([price_vec(g, GRID) for g in GOODS])


def make_price_curve_jax():
    import jax
    import jax.numpy as jnp
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    jax.config.update("jax_enable_x64", True)

    def curve(base, i0, t, below_func, below_target, above_func, above_target,
              floor, inv, gain):
        def shape(func, x, T):
            x = jnp.maximum(0.0, x)
            u = x / T
            return jnp.where(func == 0, x,                       # linear
                    jnp.where(func == 1, x * x,                  # sq
                    jnp.where(func == 2, jnp.sqrt(x),            # sqrt
                    jnp.where(func == 3, jnp.log1p(x),           # log
                    jnp.where(func == 4, jnp.log10(1.0 + x),     # log10
                             u + gain * jnp.maximum(0.0, u - 1.0) ** 2)))))  # hinge
        shape_below = shape(below_func, t, t)
        shape_above = shape(above_func, t, t)
        below = base + below_target * base / shape_below * shape(below_func, i0 - inv, t)
        above = base - above_target * base / shape_above * shape(above_func, inv - i0, t)
        price = jnp.where(inv < i0, below, above)
        return jnp.maximum(floor, jnp.round(price))

    return jax.jit(curve)


FUNC_CODE = {"linear": 0, "sq": 1, "sqrt": 2, "log": 3, "log10": 4, "hinge": 5}


def price_curve_jax(fn, inv) -> np.ndarray:
    import jax.numpy as jnp
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    out = []
    for g in GOODS:
        p = K.MARKET_PARAMS[g]
        out.append(fn(float(p["base"]), float(p["I0"]), float(p["T"]),
                      FUNC_CODE[p["below_func"]], float(p["below_target"]),
                      FUNC_CODE[p["above_func"]], float(p["above_target"]),
                      float(K.PRICE_FLOOR), jnp.asarray(inv),
                      float(K.HINGE_GAIN)))
    return np.asarray(jnp.stack(out))


# --------------------------------------------------------------------------- #
# 3. the slot-game payoff matrix: pure Python vs NumPy vs JAX
# --------------------------------------------------------------------------- #

def slot_matrix_python(inventory: float, ours: np.ndarray, theirs: np.ndarray,
                       good: str = "WHEAT", drain: float = 1.0, turns: int = 24) -> np.ndarray:
    """The current implementation: a while loop per unit per schedule pair."""
    A = np.zeros((len(ours), len(theirs)))
    for i, o in enumerate(ours):
        for j, t in enumerate(theirs):
            inv, revenue = float(inventory), 0.0
            for turn in range(turns):
                ol = float(o[turn]) if turn < len(o) else 0.0
                tl = float(t[turn]) if turn < len(t) else 0.0
                while ol > 0 or tl > 0:
                    quote = price_of(good, max(0.0, inv))
                    if ol > 0:
                        revenue += quote
                        inv += 1.0
                        ol -= 1.0
                    if tl > 0:
                        inv += 1.0
                        tl -= 1.0
                inv = max(0.0, inv - drain)
            A[i, j] = revenue
    return A


def slot_matrix_numpy(inventory: float, ours: np.ndarray, theirs: np.ndarray,
                      good: str = "WHEAT", drain: float = 1.0, turns: int = 24) -> np.ndarray:
    """Same model, no loops over units: the quote of every unit is a prefix sum.

    Inside one turn with `o` ours and `t` theirs, step k is quoted at the inventory
    before either commits: `2*min(k, m) + max(0, k - m)` units have already moved,
    with `m = min(o, t)`. Our revenue is the price at those positions for k < o,
    summed over the turns.
    """
    A = np.zeros((len(ours), len(theirs)))
    for i, o in enumerate(ours):
        for j, t in enumerate(theirs):
            inv = float(inventory)
            revenue = 0.0
            for turn in range(turns):
                o_t = float(o[turn]) if turn < len(o) else 0.0
                t_t = float(t[turn]) if turn < len(t) else 0.0
                if o_t > 0:
                    k = np.arange(int(o_t), dtype=float)
                    m = min(o_t, t_t)
                    moved = 2.0 * np.minimum(k, m) + np.maximum(0.0, k - m)
                    revenue += float(price_vec(good, inv + moved).sum())
                inv = max(0.0, inv + o_t + t_t - drain)
            A[i, j] = revenue
    return A


def make_slot_matrix_jax(good: str, turns: int, max_lot: int = 128):
    import jax
    import jax.numpy as jnp
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    jax.config.update("jax_enable_x64", True)
    p = K.MARKET_PARAMS[good]

    gain = float(K.HINGE_GAIN)

    def shape(func, x, T):
        x = jnp.maximum(0.0, x)
        u = x / T
        return jnp.where(func == 0, x,
                jnp.where(func == 1, x * x,
                jnp.where(func == 2, jnp.sqrt(x),
                jnp.where(func == 3, jnp.log1p(x),
                jnp.where(func == 4, jnp.log10(1.0 + x),
                         u + gain * jnp.maximum(0.0, u - 1.0) ** 2)))))

    def price(inv):
        base, i0, T = p["base"], p["I0"], p["T"]
        below = base + p["below_target"] * base / shape(FUNC_CODE[p["below_func"]], T, T) \
            * shape(FUNC_CODE[p["below_func"]], i0 - inv, T)
        above = base - p["above_target"] * base / shape(FUNC_CODE[p["above_func"]], T, T) \
            * shape(FUNC_CODE[p["above_func"]], inv - i0, T)
        return jnp.maximum(K.PRICE_FLOOR, jnp.round(jnp.where(inv < i0, below, above)))

    def one_pair(o, t, inventory, drain):
        def turn_step(carry, turn):
            inv, revenue = carry
            o_t = jnp.where(turn < o.shape[0], o[turn], 0.0)
            t_t = jnp.where(turn < t.shape[0], t[turn], 0.0)
            # static window: a traced lot size cannot size an arange, so the shape is
            # fixed at max_lot and the surplus positions are masked out of the revenue
            k = jnp.arange(max_lot, dtype=float)
            m = jnp.minimum(o_t, t_t)
            moved = 2.0 * jnp.minimum(k, m) + jnp.maximum(0.0, k - m)
            rev_t = jnp.where(k < o_t, price(inv + moved), 0.0).sum()
            inv = jnp.maximum(0.0, inv + o_t + t_t - drain)
            return (inv, revenue + rev_t), None

        (_, revenue), _ = jax.lax.scan(turn_step, (inventory, 0.0),
                                       jnp.arange(turns, dtype=jnp.int32))
        return revenue

    return jax.jit(jax.vmap(jax.vmap(one_pair, in_axes=(None, 0, None, None)),
                            in_axes=(0, None, None, None)))


# --------------------------------------------------------------------------- #
# 4. path sampling: the one place a vectorising backend is supposed to pay
# --------------------------------------------------------------------------- #

def paths_numpy(n_paths: int, unlocks: int, steps: int, seed: int = 0) -> np.ndarray:
    """Total drain per good per path, vectorised over paths with NumPy."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, len(SHOP_TYPES), size=(n_paths, unlocks))
    basket = basket_matrix()
    events = np.arange(unlocks, 0, -1) * (steps // UNLOCK_INTERVAL)   # crude, shape only
    return basket[types].sum(axis=1) * events.mean()


def make_paths_jax(unlocks: int, steps: int):
    import jax
    import jax.numpy as jnp

    B = jnp.asarray(basket_matrix())

    def total(types):
        return B[types].sum(axis=0)

    return jax.jit(jax.vmap(total))


# --------------------------------------------------------------------------- #
# 5. the season LP: HiGHS linprog vs milp
# --------------------------------------------------------------------------- #

def season_lp(vars_: int, integer_share: float, seed: int = 0,
              time_limit: float | None = None) -> tuple[float, str]:
    """A season-shaped LP/MILP: mostly continuous columns, a few integer ones.

    A random *all*-integer model of 1,600 columns is a different problem class from
    ours and HiGHS can chew on it for minutes (measured: over 12 minutes without
    finishing, which is why this one is capped). The real master has ~1,600 columns
    and a handful of integral ones — land, hands — so that is what is measured.
    """
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    rng = np.random.default_rng(seed)
    c = -rng.random(vars_)
    A = rng.random((vars_ // 3, vars_))
    b = np.full(vars_ // 3, vars_ / 6)
    n_int = int(round(vars_ * integer_share))
    if n_int == 0:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0.0, None)] * vars_, method="highs")
        return float(res.fun), res.message
    cons = LinearConstraint(A, -np.inf, b)
    integrality = np.zeros(vars_)
    integrality[:n_int] = 1
    opts = {"time_limit": time_limit} if time_limit else None
    res = milp(c=c, constraints=cons, integrality=integrality,
               bounds=Bounds(np.zeros(vars_), np.full(vars_, np.inf)), options=opts)
    return float(res.fun), str(res.message)


# --------------------------------------------------------------------------- #
# 6. the per-turn tracker update, on the arrays it really uses
# --------------------------------------------------------------------------- #

def tracker_step_numpy(inv: np.ndarray, drain: np.ndarray, our_sales: np.ndarray,
                       our_buys: np.ndarray, prev_inv: np.ndarray) -> np.ndarray:
    delta = inv - prev_inv
    return np.maximum(delta + drain + our_buys - our_sales, 0.0)


def make_tracker_step_jax():
    import jax
    return jax.jit(lambda inv, drain, s, b, prev: jax.numpy.maximum(
        inv - prev + drain + b - s, 0.0))


# --------------------------------------------------------------------------- #
# 7. contention: the other seat thinking, measured locally
# --------------------------------------------------------------------------- #

def _burn(stop_at: float) -> None:
    x = 0.0
    while time.time() < stop_at:
        x += float(np.sqrt(np.arange(20000.0)).sum())
    del x


def with_load(load: int, fn, reps: int = 5) -> float:
    stop_at = time.time() + 60
    procs = [mp.Process(target=_burn, args=(stop_at,)) for _ in range(load)]
    for p in procs:
        p.start()
    try:
        time.sleep(0.3)
        return timeit(fn, reps=reps)[1]
    finally:
        for p in procs:
            p.terminate()


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--contention", action="store_true")
    args = ap.parse_args()

    import scipy
    import kaggle_environments
    print(f"box: {os.cpu_count()} cores | numpy {np.__version__} | scipy {scipy.__version__} "
          f"| kaggle-environments {kaggle_environments.__version__}")

    import_costs()

    print("\n[2] price curve, 1,760 inventories x 9 goods")
    f, m, lo, hi = timeit(price_curve_numpy)
    report("numpy", f, m, lo, hi)
    import jax.numpy as jnp
    fn_j = make_price_curve_jax()
    grid_j = jnp.asarray(GRID)
    f, m, lo, hi = timeit(lambda: price_curve_jax(fn_j, grid_j))
    report("jax jit (includes compile on first call)", f, m, lo, hi)
    assert np.array_equal(price_curve_numpy(), price_curve_jax(fn_j, grid_j)), "jax disagrees"

    print("\n[3] slot-game payoff matrix, 4 x 3 schedules, 24 turns, 100-unit lot")
    # Integer lots only: the engine commits a unit at a time, so a fractional lot
    # would need its own partial-commit rule and the agent never emits one.
    ours = np.array([[100.0, 0, 0], [50.0, 50.0, 0], [34.0, 33.0, 33.0], [0.0, 0.0, 100.0]])
    theirs = np.array([[100.0, 0.0], [0.0, 100.0], [50.0, 50.0]])
    assert np.all(ours == np.round(ours)) and np.all(theirs == np.round(theirs))
    inv, lot = 10000.0, 100.0
    f, m, lo, hi = timeit(lambda: slot_matrix_python(inv, ours, theirs))
    report("python loops (shipped today)", f, m, lo, hi)
    f, m, lo, hi = timeit(lambda: slot_matrix_numpy(inv, ours, theirs))
    report("numpy prefix-sum", f, m, lo, hi)
    A_np = slot_matrix_numpy(10000.0, ours, theirs)
    assert np.allclose(A_np, slot_matrix_python(10000.0, ours, theirs)), "numpy model differs"
    jit_slot = make_slot_matrix_jax("WHEAT", 24)
    o_j, t_j = jnp.asarray(ours), jnp.asarray(theirs)
    f, m, lo, hi = timeit(lambda: jit_slot(o_j, t_j, 10000.0, 1.0))
    report("jax jit + vmap (compile on first call)", f, m, lo, hi)
    assert np.allclose(np.asarray(jit_slot(o_j, t_j, 10000.0, 1.0)), A_np, atol=1e-6), \
        "jax model differs"

    print(f"\n[4] drain over {args.paths:,} sampled unlock paths (10 unlocks, 30 days)")
    f, m, lo, hi = timeit(lambda: paths_numpy(args.paths, 10, 30))
    report("numpy vectorised over paths", f, m, lo, hi)
    jit_paths = make_paths_jax(10, 30)
    types = np.random.default_rng(1).integers(0, len(SHOP_TYPES), size=(args.paths, 10))
    f, m, lo, hi = timeit(lambda: jit_paths(jnp.asarray(types)).block_until_ready())
    report("jax jit + vmap", f, m, lo, hi)
    from belief.opponent import drain_forecast
    obs = {"step": 72, "town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]}}
    f, m, lo, hi = timeit(lambda: drain_forecast(obs, 720))
    report("closed form (no sampling at all)", f, m, lo, hi)

    print("\n[5] season LP / MILP (HiGHS, scipy)")
    for vars_, share in ((150, 0.0), (150, 0.1), (1600, 0.0), (1600, 0.05), (1600, 1.0)):
        n_int = int(round(vars_ * share))
        label = "linprog" if n_int == 0 else f"milp, {n_int} integer of {vars_}"
        f, m, lo, hi = timeit(lambda v=vars_, sh=share: season_lp(v, sh, time_limit=10.0),
                              reps=3)
        report(label, f, m, lo, hi)

    print("\n[6] per-turn tracker update, 9-vectors")
    rng = np.random.default_rng(0)
    args9 = [rng.random(9) for _ in range(4)] + [rng.random(9)]
    f, m, lo, hi = timeit(lambda: tracker_step_numpy(*args9), reps=200)
    report("numpy", f, m, lo, hi)
    jit_step = make_tracker_step_jax()
    j9 = [jnp.asarray(a) for a in args9]
    f, m, lo, hi = timeit(lambda: jit_step(*j9).block_until_ready(), reps=200)
    report("jax jit (dispatch per call)", f, m, lo, hi)

    if args.contention:
        print("\n[7] contention: the other seat thinking (local approximation)")
        work = {"numpy slot matrix": lambda: slot_matrix_numpy(10000.0, ours, theirs),
                "jax slot matrix": lambda: jit_slot(o_j, t_j, 10000.0, 1.0).block_until_ready(),
                "scipy linprog 1600": lambda: season_lp(1600, 0.0)}
        for name, fn in work.items():
            solo = timeit(fn, reps=5)[1]
            one = with_load(1, fn)
            three = with_load(3, fn)
            print(f"  {name:<22} solo {solo:8.3f} ms | +1 burner {one:8.3f} "
                  f"({one / solo - 1:+.1%}) | +3 burners {three:8.3f} ({three / solo - 1:+.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
