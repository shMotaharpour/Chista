# Kaggle Probe #1 — what the real grading machine measures

Source: one submission (`kaggle_probe-1.py`) uploaded to Kaggle, run by the
platform as its Validation Episode. Both seats are the same file, so the two
logs are: `kaggle_probe-1_0.json` = **seat 0** (measure seat, plays all 719
turns) and `kaggle_probe-1_1.json` = **seat 1** (census + contention burner +
error ladder, dies at step 324).

Every number below is a `PROBE|<tag>|<json>` line from those two files; the tag
is named so the claim can be re-checked in one grep. Derived arithmetic is
marked *(derived)*.

The four raw logs are archived as `KaggleProbes/kaggle_probes_results.zip`; the
`kaggle_probe-1_*.json` names used here are the members of that archive.

---

## 0. TL;DR

1. **The graded config is visible in the observation** and it is the real one:
   720 steps, `actTimeout 1 s`, `runTimeout 1200 s`, `startingMoney 3000`,
   `weedSpawnChance 0.005`, `shedCapacity 100`, `boardSize 10`,
   `turnsPerDay 24`, `farmHandCostMult 1`.
2. **The compute model is a free second per turn, plus a 60 s per-seat bank.**
   An overrun costs exactly the overrun (plus ~30 ms of harness overhead). Any
   turn ≤ 1.0 s is free. *(derived)* season ceiling per seat ≈ 719 free seconds
   + 60 s bank = 779 CPU-seconds.
3. **Our probe used 12.7 s of that 779 s** — 1.6 %. The master plan at ~10 ms
   per turn would use ~7 s. Compute is not our binding constraint; latency and
   *variance* are.
4. **`ortools` is NOT in the grading image.** `scipy.optimize.milp`, `highspy`,
   `cvxpy` are. A shipped agent that imports `ortools` dies on Kaggle.
5. **A 1600-variable MILP solves in 24 ms** on this machine (`milp`).
6. **The opponent costs us 25.6 % of throughput** when it thinks in the same
   turn (contention ratio 0.744) — plan the budget on the contended number.
7. **The machine wanders ±25 %** over a season (reference bench 36.0→59.8
   GFLOPS), so any timing decision needs a noise floor.
8. **Never ask OpenBLAS for 3–4 threads at runtime**: measured 18×–158× slower
   per call. Set the thread count in the environment before numpy loads, or stay
   at 1–2 threads.
9. **The error ladder answers its question**: `None`, `"PASS"`, `{}` and a 3 s
   overrun all survive; a 70 s turn (over the 60 s bank) ends that seat for
   good. Seat 0 was untouched — the two-seat split works.
10. **A dead opponent is invisible**: no field in the observation changes.
    Issue #16 must infer it from mass balance, not from a status field.

---

## 1. Episode shape and the two-seat split

| fact | value | tag |
|---|---|---|
| turns seen, seat 0 | 719 (steps 0–718) | `summary_partial` |
| turns seen, seat 1 | 325 (steps 0–324, then never called again) | `drift`, log end |
| module load time | 0.0000–0.0001 s (numpy already imported by the image) | `machine` / `boot` |
| module → first call | 0.0001 s | `boot` |
| episode result for seat 1 | last line is the 70.000 s turn: no `SURVIVED` after it | `ATTEMPT burn_the_bank` |

Two independent pieces of evidence that **each seat is its own process with its
own module state and its own 60 s bank**:

* both seats printed `-1` from the file's share-variable scratch line on their
  first call, i.e. neither saw the other's module state;
* seat 0's queue drained from 21 tasks (`queue: 20` after the first pop) while
  seat 1's own queue drained its 2 tasks independently (`queue: 1`);
* seat 1 sat at `bank 60` while seat 0 had already spent 12.7 s of its own.

---

## 2. The graded configuration (this is the competition's real setup)

`PROBE|boot` echoes the config the platform passed in:

```
seed: null                episodeSteps: 720      actTimeout: 1
runTimeout: 1200          boardSize: 10          startingMoney: 3000
maxMarketOrdersPerTurn: 10                      turnsPerDay: 24
shedCapacity: 100         weedSpawnChance: 0.005
townShopUnlockInterval: 3  townShopSellInterval: 4
townCenterSellInterval: 24  farmHandCostMult: 1
remainingOverageTime: 60  __raw_path__: /kaggle_simulations/agent/main.py
```

Consequences worth acting on:

* `weedSpawnChance = 0.005` — our tile tests have used 0.0 and 0.2 as the two
  extremes; 0.005 is the middle the grader actually uses.
* `actTimeout = 1 s` is the per-turn wall limit; `runTimeout = 1200 s` is the
  whole-season limit. 719 turns × 1 s = 719 s fits under 1200 s, so the two
  limits do not contradict each other.
* `seed: null` — the validation episode runs unseeded; nothing may depend on a
  particular seed.
* the agent is loaded as `/kaggle_simulations/agent/main.py`, so the submission
  entry point is `main.py` with an `agent(obs, config)` callable.

---

## 3. Machine and image census

`PROBE|machine`, `PROBE|packages`, `PROBE|memory`, `PROBE|multiprocessing`:

* **CPU**: `os.cpu_count() = 2`, `sched_getaffinity = 2`, **no cgroup CPU quota**
  (`cpu.max` = null, `cfs_quota_us` = -1) — the container is not quota-throttled.
  `/proc/cpuinfo` gives no model name (Xen/VM), but numpy's build reports
  OpenBLAS 0.3.31.188.0 / Haswell / `MAX_THREADS=64`.
* **SIMD**: avx, avx2, fma, sse4_2 (no avx512, no amx).
* **Python**: 3.11.13; platform `Linux-4.4.0-x86_64-with-glibc2.35`.
* **Memory**: no cgroup limit (`memory.max` = null); host `MemTotal = 6.5 GiB`;
  peak RSS measured **3.86 GiB** after loading numpy + scipy + torch + jax.
  A cgroup OOM would kill the process outright — there is very little headroom
  if torch, jax and networkx are all resident.
* **Image**: `gcr.io/kaggle-images/python-simulations@sha256:754ed0bc…`.
* **Thread control**: `threadpoolctl 3.6.0` present (so in-process limiting
  works); no `OMP_NUM_THREADS`/`MKL_NUM_THREADS` preset in the environment.
* **Packages** (`PROBE|packages`): numpy 2.4.6, scipy 1.15.3, torch 2.6.0+cu124,
  jax 0.5.2 / jaxlib 0.5.1, numba 0.60.0, cvxpy 1.6.7, highspy 1.11.0,
  networkx 3.5, numexpr 2.11.0, psutil 7.1.3, kaggle_environments 1.32.7 —
  and **`ortools: null`**.
* **Processes**: `multiprocessing` start method `fork`, one child costs
  **0.153 s**; `cpu_count 2`, child exits cleanly.

---

## 4. How much compute do we actually get?

### 4.1 Thread sweep in-process (`PROBE|cpu_capacity`, 320×320 matmul)

| threads requested | applied | reps in ~0.10 s | GFLOPS | speedup |
|---|---|---|---|---|
| 1 | 1 | 62 | 39.95 | 1.00× |
| 2 | 2 | 77 | 44.36 | 1.11× |
| 3 | 3 | 4 | **2.22** | 0.056× |
| 4 | 4 | 1 | **0.47** | 0.012× |

Seat 1's independent run of the same sweep agrees: 38.29 / 47.53 / **1.83** /
**1.23** GFLOPS. The same collapse appears in the neural-net bench:
numpy MLP 1 thread 880 µs (62.0 GF) → 2 threads 637 µs (85.6 GF, 1.38×) →
**4 threads 138 863 µs (0.39 GF — 158× slower)**.

**Reading:** 2 threads is a modest but real win (1.11–1.38×, ~1.5–1.8 cores
busy); asking for 3–4 threads *at runtime* is catastrophic. The mechanism is not
proven by this run (OpenBLAS re-sizing its buffers above 2 threads is the
plausible cause) — the measurement is what matters.

### 4.2 The sweep that is always safe (`PROBE|thread_sweep_subprocess`)

A child process with `OMP_NUM_THREADS` set *before* numpy is imported:

| threads | GFLOPS | speedup | child total |
|---|---|---|---|
| 1 | 40.59 | 1.00× | 0.481 s |
| 2 | 49.20 | 1.21× | 0.543 s |
| 4 | 50.83 | 1.25× | 0.572 s |

**Rule:** thread count must be set before the BLAS library loads (environment /
child process). Setting it at runtime is safe only at 1–2.

### 4.3 Reference-bench drift (`PROBE|drift`, n=256)

Seat 0 over the season: 40.2, 36.0, 38.0, 48.2, 59.8, 47.6, 55.1, 47.7, 46.9,
53.8, 55.7 GFLOPS — **min 36.0, max 59.8, mean 48.1, 1.66× spread**.
Seat 1: 37.3, 31.8, 33.5, **13.8** (its own burner running), 28.1.

**Rule:** the same code can run 1.7× faster or slower depending on when in the
season it runs. A decision that depends on "is this faster?" needs a tolerance
band of at least ±25 %, not a single measurement.

### 4.4 Matmul size curve (`PROBE|matmul_sizes`)

n=64: 16.9 µs (30.9 GF) · n=128: 113 µs (37.1 GF) · n=256: 911 µs (36.8 GF) ·
n=512: 7 949 µs (33.8 GF). Sustained throughput is flat ~31–37 GFLOPS, i.e.
small matrices are fine, large ones are only cheap if they are genuinely
parallel.

---

## 5. Contention: what the opponent costs us (`PROBE|contention_summary`)

Six 0.2 s reference windows while nothing else ran (steps 150–155), six while
seat 1 was burning CPU in the same turn (steps 200–205):

* quiet: 44.19, 40.92, 42.00, 42.37, 43.41, 44.51 → mean **42.90** GFLOPS
* contended: 33.42, 30.11, 33.43, 34.12, 31.34, 28.98 → mean **31.90** GFLOPS
* **ratio 0.744** → the opponent's thinking takes **25.6 %** of our throughput.

Seat 1 measuring itself while its own burner thread ran dropped to 13.8 GFLOPS
(0.40× of its own baseline): **our own background threads cost us throughput
too**, and more of it than the other seat does.

**Rule:** size every timing budget on the contended number (≈0.74× solo), and
do not run background burners in the shipped agent.

---

## 6. The billing model, measured (`PROBE|billing`)

Three deliberate overruns, each read back from `remainingOverageTime`:

| self-time | over 1 s | bank before | bank after | billed | harness overhead |
|---|---|---|---|---|---|
| 1.100 s | 0.100 | 48.3974 | 48.2695 | **0.1279** | 27.9 ms |
| 1.300 s | 0.300 | 48.2695 | 47.9304 | **0.3391** | 39.1 ms |
| 1.600 s | 0.600 | 47.9304 | 47.3003 | **0.6301** | 30.1 ms |

**Charge = max(0, turn_seconds − 1.0) + ~30 ms.** Confirmed by the ladder's 3 s
burn on seat 1: `bank 60 → 57.960592` = 2.0 + 0.0395.

The whole-season accounting of seat 0 (60 → 47.3003 = **12.70 s spent**)
decomposes as: `scipy.optimize` import 0.445 + `torch` import 7.547 + `jax`
import 1.689 + jax JIT 1.747 + the three billing turns 1.000 + ~0.27 of
surcharges *(derived)*. Every ordinary turn was free.

**Rule:** the budget is 1 s per turn, free, forever; the 60 s bank is only for
rare expensive turns and is never refilled. A 10 ms/turn planner uses ~0.9 % of
the season's compute ceiling.

---

## 7. Solvers and heavy imports

* `PROBE|import`: scipy 0.037 s · **scipy.optimize 1.445 s** · torch 8.547 s ·
  jax 2.689 s (+ JIT compile 0.164 s).
* `PROBE|milp` (assignment+capacity MILP, scipy `milp`): 100 vars 46.0 ms
  (first call, includes warm-up), 400 vars 5.4 ms, 800 vars 10.0 ms,
  **1600 vars 24.1 ms**, all `status 0` / `success true`.
* MLP forward, same shape everywhere (64×256→512→512→64): numpy 1 thread
  880 µs, 2 threads 637 µs, 4 threads 138 863 µs; **torch 1 thread 746 µs**
  (best single result), torch 2 threads 876 µs, torch 4 threads 1 790 µs;
  jax (JIT) 1 101 µs.
* jax additionally spews a stderr storm at import (rocm/TPU backend init
  failures, then a logging error `I/O operation on closed file`) and warns that
  `os.fork()` after JAX is a deadlock risk.
* **`ortools` is absent** — any CP-SAT/OR-Tools dependency in a shipped agent is
  an ImportError on Kaggle. `scipy.optimize.milp`, `highspy`, `cvxpy` are the
  available solvers.

**Rule:** the only real import tax is torch (7.5 s of bank, ~12 % of the whole
bank) and, second, jax. An agent that stays on numpy + `scipy.optimize` pays
1.45 s once and nothing else.

---

## 8. The error ladder: what the harness survives (`PROBE|ladder_armed`, `ATTEMPT`, `SURVIVED`)

Schedule actually run (seat 1 only): 300 `return_none`, 306 `return_string`,
312 `return_empty_dict`, 318 `burn_3s`, 324 `burn_the_bank`.

| step | rung | outcome |
|---|---|---|
| 300 | return `None` | **survived** — called again at 301 (`SURVIVED`) |
| 306 | return `"PASS"` (a string as the whole action) | **survived** at 307 |
| 312 | return `{}` | **survived** at 313 |
| 318 | burn 3 s (overrun ~2 s) | **survived** at 319; bank 60 → 57.96 |
| 324 | burn 70 s (bank is 60 s) | **fatal for this seat**: the 70.000 s turn is the last line, no `SURVIVED`, seat 1 is never called again |

Seat 0 was unaffected: it kept PASSing to step 718 with `bank 47.3003`, and the
episode as a whole completed. That is the design paying off — the expendable
seat can die and the measurements still come home.

Not tested this run (rungs commented out in the file): malformed action shapes
(`missing_keys`, `wrong_inner_types`, `unknown_op`, `known_op_wrong_arity`,
`bad_crop_name`), non-serialisable action (`{1,2,3}` inside `market`), deep
nesting, and an uncaught `raise` (it had already proved fatal in the local dry
run).

---

## 9. Can we see the opponent die? (`PROBE|opponent_death_visibility`)

Four digests of the opponent's farm were taken — intended as before/after around
the scheduled death — and **all four are byte-identical**:

```
{money: 3000.0, planted_tiles: 0, hands: 0, hires_today: 0,
 quadrants: ["NW"], farmer: "[4, 4]"}
fields_that_changed_across_the_death: []
```

**Caveat, and it is ours:** the snapshot schedule was computed for the full
12-rung ladder (`LADDER_START + 11 × gap = 366`), but this run enabled only five
rungs, so the fatal rung was step 324. The "before" snapshot at step 356 is
therefore *already after the death*. What the log proves is that a dead seat's
observable farm is frozen (identical digests at 356/386/516/685) and that seat 0
keeps receiving a perfectly normal-looking observation for it. What it does not
yet prove is the intended A/B straddling the death — that needs a re-run with
the snapshot steps derived from the *enabled* rung list, not the full one.

The conclusion for issue #16 is unchanged in direction and now has a second,
independent reason to be believed: the earlier local dry run produced a false
"opponent died" at step 26 on a healthy opponent precisely because a passive
opponent and a dead one look the same from inside the observation. **Death must
be inferred from mass balance (their market activity), not from any status
field.**

---

## 10. What this means for Chista

1. **Hardcode the real config into the tests**: `weedSpawnChance 0.005`,
   `actTimeout 1`, `episodeSteps 720`, `shedCapacity 100`, `startingMoney 3000`.
   Our sweeps used 0.0/0.2 for weeds; the grader uses 0.005.
2. **Drop `ortools` from any shipped path** (it is not in the image); keep
   `scipy.optimize.milp` with the existing absent-scipy guard, and treat
   `highspy`/`cvxpy` as the fallbacks that are actually present.
3. **Thread policy**: set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to 1 or 2 in
   the environment before numpy loads; never call `set_num_threads(3|4)` at
   runtime; import torch/jax in a child if at all.
4. **Timing budget**: 1 s free per turn, 60 s bank, ~0.74× throughput when the
   opponent is thinking, ±25 % machine drift. Any benchmark assertion needs a
   tolerance band wider than both.
5. **The 8.5 s torch import is 12 % of the season bank** — if any future work
   wants a learned model, its import belongs in a child process or behind a
   lazy gate, not in `main.py`'s module scope.
6. **Memory**: peak RSS 3.86 GiB of a 6.5 GiB host with no cgroup limit. The
   shipped agent must not hold torch + jax + networkx at once, and any `fork`
   copies that RSS.
7. **Nothing may depend on a seed**: the validation episode ran `seed: null`.
8. **For #16**: opponent death must be inferred from market/mass signals; there
   is no status field, and a dead seat still receives up-to-date observations
   of a frozen opponent.

---

## 11. What this run does *not* establish

* Why 3–4 runtime threads collapse (measured, mechanism unproven).
* Whether the 1 s free turn is per turn *per seat* or per turn *per episode
  step* — both seats' logs are consistent with per-seat, but only seat 0 spent
  anything (seat 1's bank never moved except for its own ladder rung).
* The per-rung *scores*: the ladder only tells us whether the agent was called
  again, not what the harness did to the reward or the final ranking.
* Anything about the leaderboard: this was the platform's validation episode
  (self-play), config `seed: null`.
* The malformed-action rungs (commented out) — their behaviour is still unknown.

---

## 12. Evidence index

| file | lines | content |
|---|---|---|
| `kaggle_probe-1.py` | 1 300+ | the probe agent (both seats) |
| `kaggle_probe-1_0.json` | 719 turns, 85 PROBE lines | seat 0: census, benchmarks, MILP, billing, drift, contention, opponent digests, summaries |
| `kaggle_probe-1_1.json` | 325 turns, 26 PROBE lines | seat 1: census, burner, ladder (5 rungs), death at 324 |

Re-check any number with:
`grep -o 'PROBE|<tag>|.*' kaggle_probe-1_0.json` (or `_1.json`).
