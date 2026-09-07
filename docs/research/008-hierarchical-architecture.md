# Hierarchical Architecture — Rethinking the ChistaAgent Problem

Branch: `hierarchical-architecture`
Status: decisions Q1–Q3 closed (2026-09); see §7.

## 1. Goal

**Maximize final coins at the end of a 720-turn season by making the best
possible action every turn, given perfect knowledge of our farm, public
market, and observed opponent state — everything computed on CPU within
Kaggle's per-turn time budget.**

Objective: `max E[money_final]`. Not "mimic the winner", not "solve a static
LP" — a sequential decision problem.

## 2. Constraints (authoritative)

1. **CPU-only runtime.** RL/DL are allowed as techniques, but every component
   must run on CPU within Kaggle's per-turn/episode time budget. Hardware
   budget numbers to be pinned down later.
2. **Crew work-division**: the primary tool is OR-Tools routing (VRP/CP-SAT)
   used OFFLINE to generate optimal schedules; from those we distill a
   learned heuristic that decides well WITHOUT or_tools at runtime. Fast,
   good-enough day-scheduling is a prerequisite for L1 — long-horizon
   planning is meaningless if day-level scheduling is slow or bad.
3. **Monotone layer inclusion**: resources grow incrementally over the season
   (land, crew, animals). A layer's solution must therefore always be
   **inclusive of the layer below** — the stack must guarantee
   "equal or better than the lower layer alone". Each layer may only add
   value over its sub-stack, never regress it.
4. **Action pruning harness**: the world engine already exists
   (kaggle-environments). Between its raw legal actions and our intent there
   is an intermediate harness layer (part of L0) that crops useless,
   wasteful, and pointless actions. Planners receive the pruned action set,
   never the raw one.
5. Every mechanic claim is proven in `lab/` first, then codified once in L0.

## 3. Why the current structure under-delivers

| Artifact | Role | Failure mode |
|---|---|---|
| `agent/main.py` (rule layer) | turn-level ops | hand-tuned, no lookahead, water coverage collapses |
| `agent/playbook.json` (Thomas ramp) | day-level targets | static; ignores cash, market drift, deaths |
| `lab/milp/cpsat_model.py` | season plan | optimistic: no travel time, no watering workload, fixed prices |
| gap between them | — | plan says $11k, execution returns ≈ $0 |

Root cause: three artifacts with three different objective functions and no
contract between layers. Each re-derives "what matters" from scratch.

## 4. The reframe: a Hierarchical MDP

The season is a sequential decision process: state = full observation,
action = 1 farmer op + N hand ops + ≤10 market ops per turn, reward = money
at turn 720. Exact solution is impossible (astronomic state space), so we
factor it into stacked controllers. Each layer is a smaller, well-posed
decision problem; **solutions pass down as constraints**, **feedback passes
up as calibration**.

```
L0  STATE MODEL      mechanics truth + simulator + pruned action harness
L1  ECONOMIC PLANNER per-day portfolio (what to own, buy, sell, hire)
L2  SCHEDULE PLANNER per-hour work division (who does what, when)
L3  DISPATCHER       per-turn executor with reflexes
```

### L0 — State Model (contract holder)

Single source of truth for mechanics; everything else queries it, nothing
re-implements it.
- `mechanics.py`: production tables, care-bank rules, decay, weeds, shed,
  town demand — all lab-verified, as code.
- `market.py`: the price function per product (shape + I0, both sides).
- `simulator.py`: forward simulator `(state, action) -> state` for
  verification, workload probing, and plan stress-testing.
- `tilegraph.py`: travel time between tiles (BFS over unlocked quadrants) —
  what the current MILP is missing.
- **`pruning.py` (the harness)**: raw legal action set → pruned candidate
  set. Removes: actions on locked tiles, pointless orders (selling into a
  $1 floor, buying seeds with no plantable tile), no-op sequences
  (PICKUP away from shed), wasteful repeats (watering an already-watered
  plant), actions that violate monotone-resource sanity. The pruned set is
  the ONLY action vocabulary any planner ever sees.

### L1 — Economic Planner (per-day, portfolio level)

Question: *what portfolio of crops/animals/land/crew maximizes final money?*
- Stage-MDP over `day = 0..29`; state: money, market inventory, live assets,
  shed. Actions: buy/sell amounts, plant/raise mix, buy land, hire crew size.
- **Solved ROLLING** (decision Q1): every evening, re-solve the remaining
  ~15-day horizon from the true current state. Reason: town shops unlock
  every 3 days (breaking initial price/demand assumptions) and the
  opponent's sell behavior moves shared market prices materially. A
  once-at-start plan is falsified by day ~10.
- Opponent modeling: phase 1 = none (prices observed today held across the
  horizon). Phase 2 = expected-supply profile from replay statistics
  (when the crowd's melons hit the market) feeding expected prices.
- Transition workload coefficients come from measured L2/L3 calibration,
  not guesses.
- Price handling in the MILP: piecewise-linear price curves over inventory
  intervals (SOS2 / interval binaries) — see §7 Q3 rationale.
- Output: `PortfolioPlan` = per-day target stocks, buys, sells, land, crew.
  Replaces `playbook.json`'s hand-tuned ramp (which becomes an L1 OUTPUT,
  not an input).

### L2 — Schedule Planner (per-hour, work division)

Question: *given today's portfolio targets and farmer + H hands × 24 turns,
what does every unit do every hour?*
- Assignment/scheduling problem: tasks (water plant i, feed animal j,
  harvest, build, collect fertilizer, travel) with travel times from L0's
  tile-graph, over the pruned action set.
- **Primary tool: OR-Tools routing (VRP/CP-SAT)** — exact schedules offline.
- **Distilled heuristic**: from optimal solutions, learn/derive a fast
  assignment rule (regret-based insertion, nearest-profitable-task) that
  runs without or_tools in Kaggle's per-turn budget. The heuristic is
  validated against the OR-Tools baseline on historical days: it must
  recover ≥95% of the optimal makespan/profit before it is trusted alone.
- Output: `DaySchedule` = unit → ordered task list, plus a **feasibility
  flag**. Infeasibility (workload > crew capacity) propagates up: L1 must
  shrink the portfolio. This closes the loop that made CP-SAT optimistic.
- Monotone guarantee (constraint 3): L2's schedule must never perform worse
  than the trivial "everyone waters in raster order" baseline; tests assert
  this.

### L3 — Dispatcher (per-turn, reactive)

Executes the `DaySchedule` but may interrupt on: plant death/escape
emergencies, rescue of 1-day-unwatered plants, end-of-day sell, weeds.
Pure deterministic rule engine over the pruned actions — simple, testable.
Output: the action dict for kaggle-environments.

### The two feedback loops

1. **Downward**: L1 → workload-constrained portfolio → L2 → per-unit task
   lists → L3.
2. **Upward**: L3 logs ops actually consumed per asset → L1's workload
   coefficients recalibrated (running stats, no ML); L2 infeasibility days
   force L1 to re-solve with tighter capacity. L0's simulator replays each
   day to verify the recorded trace matches (drift alarm).

## 5. Layer engineering rules

1. One package per layer with an explicit interface:
   `world/` (L0), `planner/` (L1), `schedule/` (L2), `dispatch/` (L3).
2. Lower layers never import upper layers. Ever.
3. Every layer has fixture tests against L0's simulator.
4. `lab/` stays the evidence home: mechanic changes proven there first,
   then codified in L0, then docs updated.
5. `playbook.json` and `agent/main.py` are superseded (become L1 output and
   L3 respectively) once the new stack passes them on the same seeds.

## 6. Directory structure

```
agent/
  entry.py            # kaggle entry point: obs -> L3 -> action
world/                # L0
  mechanics.py        # verified constants + rules
  market.py           # price function
  simulator.py        # forward simulator
  tilegraph.py        # distances, travel time
  pruning.py          # action-pruning harness
planner/              # L1
  economic.py         # rolling 15-day MILP (SOS2 prices)
  portfolio.py        # PortfolioPlan dataclass
schedule/             # L2
  day_planner.py      # or-tools routing schedule + distilled heuristic
  tasks.py            # Task dataclass, feasibility
dispatch/             # L3
  executor.py         # schedule executor
  reflexes.py         # emergency overrides
lab/                  # experiments & evidence (unchanged)
tests/                # per-layer fixture tests
```

## 7. Decisions (closed)

### Q1 — Rolling horizon: YES
L1 re-solves every evening over the remaining ~15-day horizon from the true
current state. Rationale: town shops unlock every 3 days up to day 24
(8 shop instances cap), breaking any initial price/demand assumption; and
the opponent's selling moves shared market prices materially. Computation
happens out-of-band (one solve per evening), not inside the per-turn budget.
Fallback if a total-episode CPU cap exists: re-solve only on days
0/5/10/15/20/25 + forced re-solve after any portfolio-breaking event
(animal escape, land purchase, shop unlock).

### Q2 — Opponent modeling: start with A, upgrade to B
Phase 1 (now): none — L1 holds today's observed prices across the horizon.
Phase 2 (after the stack runs end-to-end): expected-supply profiles from
replay statistics — when the aggregate crowd's melons/milk/wool hit the
market per day — feeding expected prices per horizon day. This matters most
for sell timing of glutable goods (melon/strawberry/milk/wool floor at $1
under oversupply). Full behavioral opponent modeling is deferred
indefinitely; shared market inventory already partially reflects it.

### Q3 — Prices in the MILP: why piecewise-linearization, not raw CP-SAT functions

The price function is `price(product, inventory)` with a shape function
(linear / sq / sqrt / log, different on each side of I0). The MILP needs
`revenue = sell_qty × price(inventory_after_sale)`. Why we cannot just call
the engine's price function inside the model:

1. **The model needs price as a DECISION VARIABLE, not a lookup.** L1 must
   *choose* `sell_qty` for each day; the price tomorrow depends on the
   inventory *caused by those chosen sales* — i.e. price is an endogenous
   function of decision variables. A lookup can only evaluate fixed numbers;
   it cannot appear inside an objective or constraint that the solver is
   optimizing over.
2. **CP-SAT is integer-only.** Its arithmetic is over integer variables and
   constants. The price shapes (sq, sqrt, log over real-valued inventory)
   are not expressible as CP-SAT constraints. What CP-SAT *does* support and
   what we use:
   - `AddElement` (table lookup on a variable index) — but the price table
     is huge and lookup indices must themselves be integer decision
     variables; encoding inventory → table row works, but scales badly.
   - `AddAllowedAssignments` (table constraints) — same scaling problem.
   - `AddMaxEquality`/absolute-value tricks — only for max/min shapes, not
     sqrt/log.
   - **SOS2 / interval formulation** (the standard OR-Tools idiom:
     `AddExactlyOne` over interval binaries + break-point interpolation) —
     this is the canonical way to embed any 1-D nonlinear curve in an
     integer program: approximate the curve with break points, linearly
     interpolate inside each interval, charge the solver only to pick the
     interval. Error is controlled by break-point density (e.g. 8–12
     intervals per product side), and it composes with the rest of the
     integer model.
3. **Piecewise linearity is exact at the break points and bounded-error
   between them** — with ~10 break points per product the pricing error is
   small vs. the uncertainty in opponent behavior, and the model stays one
   solvable integer program. This is why "linearize" is not a hack but the
   standard, controlled way to put an endogenous nonlinear curve inside an
   integer model.

(If a future MDP/DP formulation is needed, this same curve enters as the
transition reward table — the linearization is then replaced by exact table
values per discretized inventory state.)

## 8. Naming

"Hierarchical MDP" = سلسله‌مراتبی. Layers = لایه. The stack is 4 layers:
L0 state, L1 season economy, L2 day schedule, L3 turn dispatch.

## 9. Build order (each step verified before the next)

1. **L0 `world/mechanics.py` + `market.py`** — port lab-verified mechanics
   and price function into code. Verify: reproduce one known goose day and
   one melon price series exactly.
2. **L0 `world/simulator.py` + `tilegraph.py` + `pruning.py`** — forward
   simulator, travel times, action harness. Verify: simulator replay of a
   full episode matches engine state per turn (drift = 0); pruning removes
   only provably-useless actions.
3. **L2 `schedule/`** — or-tools routing schedule + distilled heuristic,
   given today's playbook as a fixed portfolio. Verify: 100% watering
   coverage on 8 melons + 2 pastures; heuristic ≥95% of OR-Tools baseline.
4. **L1 `planner/`** — rolling MILP with L2-measured workload coefficients
   and SOS2 prices. Verify: plan survives L0 simulation to end-of-season
   with money > playbook baseline.
5. **L3 `dispatch/`** — port `playbook_agent.py` fieldwork as executor +
   reflexes; wire `agent/entry.py`. Verify: end-to-end season money beats
   current rule agent on 3 seeds.
6. **Calibration loop** — run seasons, feed op-logs to L1 coefficients,
   re-solve. Verify: measured ops/asset converge and plan quality improves
   monotonically across iterations.
