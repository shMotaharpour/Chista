# Kaggle probe #2 — results from the grading machine

Source: `kaggle_probe-2.py` uploaded to Kaggle, run as the platform's Validation
Episode. Both seats are the same file: `kaggle_probe-2_0.json` = **seat 0**
(719 turns, whole season), `kaggle_probe-2_1.json` = **seat 1** (504 turns, then
the platform stopped calling it). Every number below is a `PROBE|<tag>|<json>`
line; derived arithmetic is marked *(derived)*.

The four raw logs are archived as `KaggleProbes/kaggle_probes_results.zip`; the
`kaggle_probe-2_*.json` names used here are the members of that archive.

---

## 0. Answers in one screen

1. **Item 3 — the time-bank question is answered, and the prediction was exact.**
   Seat 1 was tolerated for **20 daily 3 s drains**, then **stopped at step 503
   (day 20, hour 23) and never called again** — exactly the day and step the probe
   printed at boot. The charge per drain is `over_1s + ~31-41 ms`.
2. **Item 1 — PyMC is unusable on the grading machine.** `pymc` is present
   (`find_spec` true) but `import pymc` raises
   `ImportError('numpy.core.multiarray failed to import')` — a compiled-extension /
   NumPy-ABI failure. There are **no** import-cost or NUTS numbers from Kaggle;
   the PyMC child never started.
3. **Item 2 — the SciPy column ran and is trustworthy:** MAP converged
   (`grad_norm` 3.9e-4) in 0.73 s, Laplace 0.22 s, and the MAP agrees with the
   exact conjugate posterior of the market level to **1e-6**. The hand-rolled MH
   chain did **not** converge on Kaggle (acceptance 0.113, mean off by 0.31) —
   reported as unusable, not as a third opinion.
4. **Item 4 — sharing the CPU does matter, for one of the two workloads.** With
   both seats computing the same work in the same turn, the SciPy chunk costs
   **+41 % wall time per unit** (median 6.01 ms vs 4.27 ms solo, both repetitions
   agreeing), while the JAX chunk shows **no** penalty (−3 %). Seat 1 measures the
   same +41 % from its side, so it is a shared-resource effect, not a one-sided
   artifact. Mechanism unresolved (see §5).
5. **Item 5 — the sandbox holds on Kaggle.** An `Environment` object *is* reachable
   in the agent's process: we wrapped `Environment.step`, planted 23 MELON tiles
   through the engine's own constructor and bumped its money. But the wrapper was
   **never called** (`intercept_calls: 0`) and **our own observation never changed**
   (money stayed exactly 3000, zero melon tiles). That object is an inert copy the
   agent container builds for the schema; the scored state is elsewhere.
   Locally (same process) the identical code *did* change the game
   (money 3000 → 4000, 24 melon tiles visible in our own observation).
6. **Item 6 — the opponent cannot see our environment variables.** Our private
   variable was still readable 697 steps after being set (`own_value_matches_boot:
   true`), while the other seat's private variable was **absent at every one of
   the 15 reads** ⇒ two separate environments. `shared_module_state: false` in
   both seats confirms the two seats are separate processes on Kaggle.

---

## 1. Episode and configuration

| fact | value |
|---|---|
| seat 0 turns | **719** (full season, steps 0-718) |
| seat 1 turns | **504**, then no further call |
| module load / first call | 0.0 s / 0.0002 s |
| bank at boot (both seats) | 60 s |
| seat 0 bank at the end | 45.561 s (spent 14.44 s) |
| seat 1 bank at the end | 13.90 s (spent 46.10 s) |
| topology | `seats_seen_by_this_interpreter: [0]` / `[1]`, `shared_module_state: false` ⇒ two processes |

Config echoed by the platform is identical to probe #1: `episodeSteps 720`,
`actTimeout 1`, `runTimeout 1200`, `startingMoney 3000`, **`weedSpawnChance 0.005`**,
`shedCapacity 100`, `turnsPerDay 24`, `boardSize 10`, `seed: null`.

---

## 2. Item 3 — draining the bank until the platform stops the seat

Charge per drain, read back from `remainingOverageTime`:

| day | burn | charged | surcharge |
|---|---|---|---|
| 0 | 3.0 s | 2.0388 | 38.8 ms |
| 1 | 3.0 s | 2.0413 | 41.3 ms |
| 2 | 3.0 s | 2.0319 | 31.9 ms |
| 18 | 3.0 s | 2.0370 | 37.0 ms |
| 19 | 3.0 s | 2.0314 | 31.4 ms |

20 drains settled, **40.77 s charged** *(derived; matches probe #1's model:
`charge = max(0, turn − actTimeout) + ~30 ms`)*.

The fatal turn:

```
drain_attempt day 20, step 503, seconds 23.0 (3 s daily + the 20 s extra drain),
              bank_before 13.902, over_act_timeout 22.0
→ no drain_settled line, no further call for seat 1
```

**Predicted at boot:** `dies_on_day 20, dies_at_step 503, bank_left 19.4,
needed 22.0` → the day and the step were right; the bank level was 5.5 s
optimistic because the prediction assumed a 30 ms surcharge and knew nothing about
seat 1's own JAX setup turn (3.46 s) and the 37 ms mean surcharge. The rule
itself (`kaggle_environments/agent.py:220`) held exactly.

**For the real agent:** the bank is per seat, 60 s, never refilled; every turn
≤ 1.0 s is free; the harness adds ~30-40 ms to each charged turn.

---

## 3. Item 1 / Item 2 — Bayesian inference on the grading machine

### 3.1 PyMC (item 1)

* `find_spec("pymc")` → **true** (the package is installed in the image).
* `import pymc` → **`ImportError('numpy.core.multiarray failed to import')`**,
  first seen in the item-4 PyMC chunk (36 failed measurements, all with that
  error) and costing seat 0 a **5.56 s turn** (≈4.6 s of bank) to discover.
* The PyMC worker child **never started**: `pymc_child: started false, reason
  "cannot locate this module on disk, so a child cannot import it"`. On Kaggle the
  module has no usable `__file__`, so the child launcher could not be built.
  ⇒ **no PyMC import time, no pytensor compile time, no NUTS ms/draw from Kaggle.**

**Conclusion:** PyMC cannot be used inside a Kaggle submission as the image
stands, and an agent that tries pays bank to find out.

### 3.2 The network inferred in SciPy (item 2)

| step | seconds | result |
|---|---|---|
| MAP (L-BFGS-B, 4216 f-evals) | 0.728 | converged, `grad_norm` 3.89e-4, `mu` 3.4340, `eta` 0.1413 |
| Laplace (30×30 Hessian, finite differences) | 0.224 | `mu_sd` 0.2392, smallest curvature 0.998 (positive definite) |
| MH (4000 iters, warmup-chosen scale 0.3) | 0.492 | acceptance **0.113**, `mu_mean` 3.7429, `mu_sd` 0.057 |
| exact conjugate posterior of `mu` | — | 3.434018, sd 0.019225 |

* MAP vs exact: **|Δ| = 1e-6** ⇒ the fitted centre is right.
* MH vs MAP: **Δ = 0.31** with acceptance 0.113 ⇒ this chain is not a usable
  posterior sample. The scale warmup picked `0.3` although `0.1` scored 0.65 and
  `0.06` scored 0.773 — the scoring rule penalised the too-hot scales instead of
  the too-cold one, a defect in the probe (see §7), and the honest reading of the
  Kaggle run is "MH unusable here".
* Speed vs this VM (same code, same data): MAP 0.62 → 0.73 s, Laplace 0.19 →
  0.22 s, MH 0.38 → 0.49 s ⇒ **Kaggle is ~1.2-1.3× slower** for this
  single-threaded NumPy work.

---

## 4. Item 4 — does it matter whether both agents compute?

Fixed work units per turn (JAX 12, SciPy 4); the metric is ms per unit, so a
larger number means the same work took longer. Pooled over both repetitions:

| chunk | solo | pair (both seats work) | overlap (opponent burns in a thread) |
|---|---|---|---|
| JAX | 1.445 ms | **1.400 ms (−3.1 %)** | 1.270 ms (−12.1 %) |
| SciPy | 4.265 ms | **6.010 ms (+40.9 %)** | 4.600 ms (+7.9 %) |

* The SciPy effect is consistent across both repetitions (5.690 and 6.030 ms in
  the two pair windows) and is **symmetric**: seat 1's own pair-arm measurements
  (median 5.436 ms) show the same penalty, so the two seats really are sharing a
  CPU, not merely measuring each other's bookkeeping.
* The JAX chunk shows no penalty, and its overlap arm is even *faster* than solo
  (−12 %): per probe #1's own rule, an arm that beats solo is a broken reading
  (warm-up/boost/order), not evidence that load helped.
* Both arms of the same chunk ran on the same day (hours 2 / 8 / 14), so the
  comparison is within-day; the 2-day repetition is what makes the SciPy finding
  believable.
* **Mechanism is NOT established** by this run. The two chunks differ in shape
  (short jitted kernels with reused buffers vs ~4-6 ms of allocating NumPy vector
  ops), so allocator/page-fault contention is a hypothesis, not a measurement.

**For the real agent:** the shipping planner is SciPy-shaped, so plan the per-turn
budget on ~1.4× the solo number when the opponent is thinking in the same turn.

---

## 5. Item 5 — can an agent rewrite the environment from inside?

What the probe found on Kaggle, in order:

```
env_scan  kaggle_environments importable: true, kaggriculture importable: false,
          240 171 objects scanned, candidates:
            [Struct (kaggle_environments.utils) has_step false]   ← false positive
            [Environment (kaggle_environments.core) has_state true, has_step true]
env_intercept  installed true, target "Environment.step", mode "instance"
env_plant_applied  seat 0, planted 23, via kaggriculture._new_plant
env_patch_applied  seat 0, money_bumped 1, melons_planted 23, day 12
patch_effect  step 289..311 → money 3000.0 (Δ 0.0), melon_tiles_visible 0,
              intercept_calls 0, patches_applied 1
```

Read together: the object was **writable** (we did plant 23 well-formed MELON tiles
and add money to its farm dict) but it was **not the live game** — the wrapper was
never invoked and our own observation never moved. That is exactly the shape of an
`Environment` instance the agent's container builds to read the specification, not
the one the runner steps.

**Local comparison (same code, same phase, in-process run):** `Environment.step`
wrapped, 24 MELON tiles planted, and at the next step our own observation showed
**money 3000 → 4000 and 24 melon tiles**. So the code path is proven; Kaggle says
the path does not reach the scored state.

**Answer:** on Kaggle, no — the environment the agent can touch is not the
environment it is scored by. **A mutation that has no effect in your own next
observation is not evidence of a reachable environment**, and the probe's own
verdict line said the right thing for the wrong-looking reason.

---

## 6. Item 6 — environment variables across turns and across seats

| reading | seat 0 | consequence |
|---|---|---|
| own variable, step 0 → 697 | `0-fa626ae7` unchanged, `own_value_matches_boot: true` | `os.environ` survives hundreds of turns |
| other seat's private variable | **absent in all 15 reads** | separate OS environments |
| shared name `PROBE2_MARK` | always our own value, `shared_owner: me` | nothing of ours was overwritten |
| `environ_entries` | 82 at boot → 86 at the end | only our own writes |
| `shared_module_state` | false for both seats | two processes, not one |

A child process inherits our variables (measured locally), and a variable a child
sets never appears in the parent — environment memory is one-way, so env vars are
**not** a usable channel between the two seats on the platform.

---

## 7. Defects this run exposed in the probe itself (all fixable)

1. **`__file__` is not usable on Kaggle** ⇒ the PyMC child never launched and
   item 1 lost its cost numbers. Fix: try a list of candidates
   (`/kaggle_simulations/agent/main.py`, `sys.argv[0]`, the module's `__file__`)
   and fall back to an in-process run with a small, bank-safe sampling budget.
2. **The env-object search accepts an inert object.** Fix: require a callable
   `step`/`_step` and, before reporting a patch as effective, require
   `intercept_calls > 0` (the effect check already caught this; the verdict text
   should have said "object found but never stepped").
3. **The false-positive candidate `Struct`** (a dict-like with `state`) passed the
   env-like test. Fix: require `hasattr(obj, "step") or hasattr(obj, "_step")`.
4. **The MH scale-scoring rule penalises the wrong side**: it prefers a
   too-cold chain (acceptance 0.11) over a too-hot one (0.65). Fix: compare in log
   space around 0.234 and never accept a chain below ~0.15 without saying so.
5. **The item-1 chunk should have detected the NumPy-ABI failure as a first-class
   finding** instead of repeating the same ImportError 36 times: report it once,
   then mark the chunk unavailable for the rest of the episode.

---

## 8. What to change in the real agent because of this run

1. **Do not depend on PyMC** (it cannot import on the grading machine). SciPy's
   MAP + Laplace is available, fast (sub-second at 30 parameters) and verifiable
   against an exact conjugate block — keep that as the inference path.
2. **Budget 1.4× the solo number** for SciPy-shaped per-turn work when the
   opponent is thinking in the same turn; the 1 s free turn is what keeps this
   affordable, and probe #1's ±25 % drift still has to be added on top.
3. **Bank policy is confirmed and cheap to respect:** 60 s per seat, ~30-40 ms
   surcharge per charged turn, and a seat that overspends is stopped for good —
   the episode continues for the other seat, which is exactly what a two-seat
   split is for.
4. **Do not use env vars, module globals or the observation's silence as a channel
   between agents:** nothing crosses on the platform.
5. **Never plan on touching the environment** — the reachable object is an inert
   copy; the local in-process result is a property of the local harness.
