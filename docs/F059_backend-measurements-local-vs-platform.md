# F059 — NumPy vs SciPy vs JAX, per workload, measured locally

**Summary (<=50 words):** The backend choice is per-workload, not per-project. On
this box NumPy wins the per-turn work (price curve 0.35 ms vs JAX 1.42 ms), SciPy
owns the LP/MILP, and JAX wins only batched kernels: the slot-game matrix 0.020 ms
against NumPy's 0.641 ms. JAX's default float32 silently breaks engine parity, and
`x64` fixes it at a cost.

Run: `.venv/bin/python -m bench.bench_backends --contention`
Readings: `bench/bench_backends.py`, printed first-call / warm-median, every JAX
result forced through `block_until_ready()`.

> Scope: this is the **development box** (8 cores, numpy 2.4.6, scipy 1.17.1, jax
> 0.10.2 with x86-64 CPU, kaggle-environments 1.32.7). The grading platform is a
> different machine with different versions (F058: scipy 1.15.3, jax 0.5.2, ~1.6
> vCPU) — the two sets are combined in §4, and nothing here supersedes F058.

## 1. Measured

### Import cost, fresh interpreter (what a first turn pays)

| module | this box | platform (F058) |
|---|---:|---:|
| numpy | 100 ms | — |
| scipy.optimize | 504 ms | — |
| jax | 554 ms | ~1,689 ms |
| jax + jax.numpy | 640 ms | ~1,689 ms |
| torch | 1,395 ms | 7,547 ms |

### Per-workload timings

| workload | pure Python | NumPy | JAX (first call) | JAX warm |
|---|---:|---:|---:|---:|
| price curve, 1,760 inventories × 9 goods | — | **0.35 ms** | 207 ms | 1.42 ms |
| slot-game matrix, 4×3 schedules, 24 turns, 100 wheat | 32.2 ms | 0.64 ms | 322 ms | **0.020 ms** |
| drain over 20,000 sampled paths | — | 9.1 ms | 79 ms | **2.9 ms** |
| the same drain, closed form | — | **0.094 ms** | — | — |
| per-turn tracker update, 9-vectors | — | **0.003 ms** | 48 ms | 0.009 ms |
| LP, 150 vars (HiGHS) | — | — | 7.0 ms (only SciPy can) | 7.0 ms |
| MILP, 15 integer of 150 | — | — | 23.1 ms | 23.1 ms |
| LP, 1,600 vars, dense random | — | — | 999 ms | 999 ms |
| MILP, 80 integer of 1,600, dense random | — | — | 3,954 ms | 3,954 ms |
| MILP, 1,600 integer of 1,600, dense random | — | — | 11,546 ms | 11,546 ms |

Contention, with 1 and 3 local CPU burners (`--contention`): NumPy slot matrix
−2.9 % / +17.9 %, JAX slot matrix −1.3 % / +20.1 %, `linprog` 1,600 +4.1 % / +6.1 %.
**This arm is weak evidence**: 8 cores absorbing two burners is not the platform's
1.6 vCPU with the other seat thinking. F058's platform numbers (+41 % on
SciPy-shaped work, −3 % on a jitted JAX chunk) are the ones that decide.

## 2. What the JAX runs cost that the numbers do not show

1. **JAX defaults to float32 and that breaks engine parity.** With the default,
   `CARROT` at inventory 0 priced at **16,576** instead of 126,920 — the price
   function is a hinge with `HINGE_GAIN = 8`, so a small relative error at
   `u ≫ 1` becomes a large absolute one, and the answer is then rounded to an int.
   `JAX_ENABLE_X64=true` is **mandatory** for anything that must agree with the
   engine; every JAX number above is with x64 on (x64 is the slower path).
2. **Static shapes.** A traced lot size cannot size an `arange`, so the JAX kernel
   pays for a fixed 128-unit window whether the lot is 3 or 100. Vectorised JAX is
   only available where an upper bound on the shape can be named in advance.
3. **JIT is per kernel and per shape**, 0.2–0.4 s each locally (F058: 0.16–1.7 s on
   the platform, and the first JAX use costs a seat 3.5–7.5 s of its 60 s bank).
4. **`os.fork()` + JAX threads = a deadlock warning.** The contention arm had to be
   allowed to fork anyway; any harness that adopts JAX must use `spawn`.
5. **A dense random MILP is a different problem class.** The first version of this
   bench handed HiGHS 1,600 all-integer columns and it did not finish in 12 minutes;
   with a 10 s limit it returns in 11.5 s. Our real master must stay structured and
   sparse, and must own a time limit plus a fallback plan.

## 3. Verdict

- **NumPy — the per-turn default.** It wins small arrays outright (tracker 3 µs vs
  JAX's 9 µs; price curve 4× faster warm) and the pumped-up slot matrix it already
  has is 50× the pure-Python loop it replaced. Per-turn work stays NumPy.
- **SciPy (HiGHS) — the only LP/MILP.** JAX has no solver of any kind. The season
  plan stays `scipy.optimize`, with a structure that stays sparse and a time limit.
- **JAX — a specialist, and only behind a switch.** It is the right tool for one
  shape of work: batched, hot, fixed shape. The slot-game matrix is 32× NumPy's and
  1,600× the Python loop; path sampling is 3.2×. It is the wrong tool for the
  per-turn path (dispatch overhead exceeds the work) and it cannot replace SciPy.
  Adopt it when the slot search outgrows ~1 ms in NumPy, load it lazily, and pay the
  import and JIT once at hour 0.

**The ordering the numbers actually support:** change the algorithm before you
change the backend (Python loop → NumPy is 50×; NumPy → JAX is another 32× on that
one kernel), and prefer a closed form to sampling at all (0.094 ms against 9.1 ms
for 20,000 sampled paths). The claim in the merged probe report that "JAX might even
be better than NumPy" holds for exactly one of the six workloads measured here.

## 4. Readings this revises or leans on

- **F058** (platform, authoritative for the graded machine): versions, the per-seat
  bank, the surcharge, the contention arms. This file adds the per-workload split
  F058 never took.
- The `world/prices.py` parity guard made the float32 defect visible; without it the
  JAX curve would have been reported as "4× slower **and** wrong".
