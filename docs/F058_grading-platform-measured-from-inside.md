# F058 — The grading platform, measured from inside a submission

**Summary (<=50 words):** Two submissions measured the grading machine: the
per-seat 60 s bank stopped one seat at step 503 of 719 while the other finished
the season; scipy.optimize.milp is present, ortools absent, PyMC cannot import;
a shared turn costs +41 % on SciPy-shaped work; environment variables and the
environment itself do not cross seats.

## Measured

Everything below is a `PROBE|<tag>|<json>` line from the four logs in
`KaggleProbes/kaggle_probes_results.zip` (probe 1 = machine census, billing,
ladder; probe 2 = inference, bank drain, contention arms, isolation).
`kaggle_probe-N_0.json` is seat 0, `_1.json` is seat 1.

### The graded configuration, echoed by the platform (`boot.config`)

```
episodeSteps 720   actTimeout 1   runTimeout 1200   boardSize 10
startingMoney 3000   maxMarketOrdersPerTurn 10   turnsPerDay 24
shedCapacity 100   weedSpawnChance 0.005   seed: null
townShopUnlockInterval 3   townShopSellInterval 4   townCenterSellInterval 24
farmHandCostMult 1   __raw_path__ /kaggle_simulations/agent/main.py
```

The validation episode runs **unseeded**, so nothing in the closure may depend on
a seed.

### The runtime budget, confirmed and refined (F046)

A deliberate 3 s burn in the last hour of each day, with the bank read back:

```
day 0  burn 3.0 s  bank 60.000 → 57.961   charged 2.0388  (surcharge 38.8 ms)
day 1  burn 3.0 s  bank 57.961 → 55.920   charged 2.0413  (surcharge 41.3 ms)
day 2  burn 3.0 s  bank 55.920 → 53.888   charged 2.0319  (surcharge 31.9 ms)
…      20 drains settled, 40.77 s charged in total
```

The fatal turn (`drain_attempt`, no `drain_settled` after it):

```
day 20  step 503  burn 23.0 s   bank_before 13.902   over_act_timeout 22.0
→ seat 1 is never called again; seat 0 plays all 719 turns and ends with bank 45.561
```

Three facts F046 does not state, all measured here:

* **the bank is per seat** — seat 1 spent 46.10 s while seat 0 spent 14.44 s in
  the same episode, each against its own 60 s;
* **only the offending seat is stopped** — the episode continued for 216 more
  turns on the other seat, which is what a two-seat split is for;
* the surcharge is **~31-41 ms per charged turn** on the platform (probe 1 read
  27.9 / 39.1 / 30.1 ms for 0.1 / 0.3 / 0.6 s overruns), consistent with F046's
  ~35 ms.

### What the image contains (`packages`, `milp`, `import`, `arm_measure`)

* `scipy 1.15.3` with `scipy.optimize.milp`: 400/800/1600 variables solved in
  5.4 / 10.0 / 24.1 ms (probe 1 `milp`).
* **`ortools` is absent** (probe 1 `packages`: `ortools: null`) — a submission
  whose closure reaches it dies on import.
* **PyMC is installed but cannot import**: `find_spec("pymc")` is true, and
  `import pymc` raises `ImportError('numpy.core.multiarray failed to import')`
  (a NumPy-ABI failure; 36 identical failures in probe 2's `arm_measure`).
  Failing that import inside a turn cost 5.56 s of wall time, ≈4.6 s of bank.
* Loading a jitted JAX kernel is not free either: the first JAX chunk cost seat 0
  a 7.50 s turn (import + compile, 3.07 s of it measured as `setup_s`) and seat 1
  a 3.46 s turn.

### Sharing a turn costs what it costs (`arm_window`)

Same fixed work per turn, ms per unit, pooled over two repetitions each:

| chunk | solo | both seats work | opponent burns in a thread |
|---|---|---|---|
| JAX (jitted matmul) | 1.445 ms | 1.400 ms (−3 %) | 1.270 ms (−12 %) |
| SciPy (30-dim log-posterior + gradient) | 4.265 ms | **6.010 ms (+41 %)** | 4.600 ms (+8 %) |

The SciPy penalty appears in both repetitions (5.690 and 6.030 ms) and is
**symmetric** — seat 1 measures 5.436 ms median from its own side — so the seats
really share a CPU. The JAX chunk shows nothing, and its "faster" arms are
within the drift band, which per probe 1's own rule means the reading, not the
load, is what moved. **The mechanism is not established**: the two chunks differ
in shape (short jitted kernels on reused buffers vs allocating NumPy vector ops),
and allocator contention is a hypothesis, not a measurement.

### The seats are isolated (`envvar_read`, `env_intercept`, `patch_effect`)

* `opponent_var_present: false` in **all 15** reads of the other seat's private
  environment variable, while our own value still matched its boot value at step
  697 ⇒ two separate OS environments; `shared_module_state: false` in both seats
  ⇒ two processes. Env vars are not a channel between the seats.
* An `Environment` object **is** reachable in the agent's process: probe 2
  wrapped `Environment.step` and planted 23 well-formed MELON tiles through the
  engine's own `_new_plant`. But the wrapper was **never called**
  (`intercept_calls: 0`) and our own next observation showed **no change at all**
  (money 3000 → 3000, zero melon tiles). The reachable object is not the scored
  one. The identical code does change the game in a local same-process run
  (money 3000 → 4000, 24 melon tiles visible), so the platform, not the code, is
  what stops it.

### Machine behaviour (supporting)

* The reference bench wanders **36.0 → 59.8 GFLOPS** across one season (probe 1
  `drift`, 11 samples): a single timing measurement is not a budget.
* For the same single-threaded NumPy work the platform is **~1.2-1.3× slower than
  this development VM** (SciPy MAP 0.62 → 0.73 s, Laplace 0.19 → 0.22 s, MH
  0.38 → 0.49 s). F046's "~1.95×" compares against a different development
  machine, so the two numbers are not in conflict — but neither should be quoted
  as a machine constant.

## What it changes for us

* SciPy-shaped per-turn work: budget **1.4×** the solo figure for turns where the
  opponent is also thinking, on top of the drift band.
* The shipped closure may use `scipy.optimize.milp`; it must not reach `ortools`
  (absent) or PyMC (present, unimportable).
* A seat that overspends is stopped for good, and the other seat keeps playing —
  the measured reason the two-seat split is worth its complexity.
* There is no side channel between the seats: not environment variables, not
  module state, not the environment object. Any cooperation goes through the
  observation and the market.
* The bank is per seat, so "we have 60 s" means 60 s **each**, and a seat that
  plans to borrow the other's headroom is planning on nothing.

## Sources

* `KaggleProbes/kaggle_probe-1.py`, `KaggleProbes/kaggle_probe-2.py` — the probes
  (tags: `machine`, `packages`, `milp`, `billing`, `drift`, `contention_summary`,
  `boot`, `predicted_cutoff_seat`, `drain_attempt`, `drain_settled`, `arm_window`,
  `env_scan`, `env_intercept`, `patch_effect`, `envvar_read`).
* `KaggleProbes/kaggle_probes_results.zip` — the four raw logs
  (sha256 `154689395372c821fd0743adcfa4610e2610a1353c6611f52942d774000d8188`).
* `KaggleProbes/kaggle_probe-1-analysis.md`,
  `KaggleProbes/kaggle_probe-2-analysis.md` — the readings and their limits.
* F046 (runtime budget) and F047 (silent operations) for the rules this refines.
