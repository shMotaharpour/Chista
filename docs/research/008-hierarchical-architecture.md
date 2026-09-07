# Hierarchical Architecture — Rethinking the ChistaAgent Problem

Branch: `hierarchical-architecture`

> **Revision 1** (user corrections):
> 1. RL/DL are NOT banned. The constraint is: everything must run on CPU
>    (hardware budget to be defined later).
> 2. Crew work-division: primary idea is OR-Tools routing (VRP/CP-SAT);
>    additionally train/derive a learned heuristic that decides well WITHOUT
>    or_tools at runtime — fast scheduling is a prerequisite for L1, because
>    long-term planning is meaningless if day-level scheduling is slow/bad.
> 3. Resource monotonicity: the problem's resources grow incrementally
>    (land, crew, assets). Upper layers must therefore be *inclusive* of
>    lower-layer results: a layer's solution is always equal to or better
>    than the layer below it (monotone improvement guarantee).
> 4. The world engine already exists (kaggle-environments). But between the
>    raw legal actions and our intent there must be an intermediate
>    "harness/pruning layer" that crops useless, wasteful, and pointless
>    actions — the action space handed to planners is pruned, not raw.

## 1. Goal restated (single sentence)

**Maximize final coins at end of a 720-turn season by making the best possible
action every turn, given perfect knowledge of our farm, public market, and
(observed) opponent state — everything computed on CPU within Kaggle's
per-turn time budget.**

Not "mimic the winner". Not "solve a static LP". One objective:
`max E[money_final]`.

## 2. Why the current structure under-delivers

| Artifact | Role | Failure mode |
|---|---|---|
| `agent/main.py` (rule layer) | turn-level ops | hand-tuned, no lookahead, water coverage collapses |
| `agent/playbook.json` (Thomas ramp) | day-level targets | static; ignores cash, market drift, deaths |
| `lab/milp/cpsat_model.py` | season plan | optimistic: no travel time, no watering workload, fixed prices |
| gap between them | — | plan says $11k, execution returns ≈ $0 |

Root cause: three artifacts with three different objective functions, no
contract between layers. Each layer re-derives "what matters" from scratch.

## 3. The reframe: a Hierarchical MDP

The season is a sequential decision process (MDP): state = full observation,
action = 1 farmer op + N hand ops + ≤10 market ops per turn, reward = money at
turn 720. Solving it exactly is impossible (state space astronomic), so we
factor it into stacked controllers, each a smaller, well-posed MDP whose
**solution is passed down as constraints** and whose **performance feedback is
passed up**. This is a Hierarchical MDP / option-framework style decomposition —
deterministic, no RL/DL, per project rules.

```
L0  STATE MODEL      (ground truth + world simulator)      no decisions
L1  ECONOMIC PLANNER (season MDP, coarse time: per day)    outputs portfolio
L2  SCHEDULE PLANNER (day MDP, time: per hour)             outputs day schedule
L3  DISPATCHER       (turn MDP, time: per turn)            outputs actions
```

### L0 — State Model (the contract holder)

The single source of truth for mechanics. Everything else queries it.
- Verified mechanics as code (not docs): production tables, care-bank rules,
  price function shape per product, decay, weeds, shed caps, town demand.
- A fast **forward simulator**: given (state, action) -> next state.
- A **tile-graph**: distance/time between any two tiles (BFS over unlocked
  quadrants) — this is what the MILP was missing.

Contract: every upper layer may only *read* L0. No layer may re-implement a
mechanic. When a mechanic is re-verified in the lab, L0 is patched once.

### L1 — Economic Planner (per-day, portfolio level)

Question: *what portfolio of crops/animals/land/hiring maximizes final money?*
- MDP over `day = 0..29`; state: money, market inventory, live assets, shed.
- Actions: buy/sell X, plant/raise portfolio mix, buy land, hire crew size.
- Transition uses **measured** per-asset workloads from L2/L3 calibration
  (actions/day actually consumed per melon/cow/...), and market price model
  from L0.
- Solved as MILP/DP over 30 stages (not RL) — deterministic, all-integer.
- Output: `PortfolioPlan` = for each day: target stocks, buys, sells, land,
  crew size. (This replaces `playbook.json`'s hand-tuned ramp.)

### L2 — Schedule Planner (per-hour, workload level)

Question: *given today's portfolio targets and one farmer + H hands with 24
turns each, what should every unit do each hour?*
- This is an assignment/scheduling MDP: tasks (water plant i, feed animal j,
  harvest, build, collect fert, travel) with travel times from L0's tile-graph.
- Solved by greedy-with-regret or small MILP per day (720 tasks max — cheap).
- Output: `DaySchedule` = unit → ordered task list with feasibility flag.
- **Feedback up to L1**: if infeasible (more work than hands), L1 must shrink
  the portfolio. This closes the loop that made CP-SAT optimistic.

### L3 — Dispatcher (per-turn, reactive)

Question: *the schedule says "hand 2 waters (3,1) at hour 5"; the world just
changed (weed spawned, plant died, price spiked). What now?*
- Executes the DaySchedule but may interrupt: death/emergency overrides,
  end-of-day sell, rescue of 1-day-unwatered plants.
- Pure rule engine, deterministic, no planner. Simple, testable.
- Output: the actual action dict for kaggle-environments.

### The two feedback loops that make it correct

1. **Downward**: L1 passes workload-constrained portfolio to L2; L2 passes
   per-unit task lists to L3.
2. **Upward**: L3 logs actual ops consumed per asset → L1's workload
   coefficients are calibrated (running average, no ML — just stats);
   L2 infeasibility days force L1 to re-solve with tighter capacity.

## 4. Layer engineering rules

1. Each layer lives in its own package with an explicit interface
   (`L0` = `world/`, `L1` = `planner/`, `L2` = `schedule/`, `L3` = `dispatch/`).
2. Lower layers never import upper layers. Ever.
3. Every layer has a fixture test against L0's simulator (replay a known day,
   assert outputs).
4. The lab (`lab/`) stays the *evidence* home: every mechanic change is proven
   there first, then codified in L0, then docs updated.
5. Replaces: `playbook.json` (becomes L1 output, not input), `main.py` rule
   soup (becomes L3).

## 5. Directory structure (proposed)

```
agent/
  entry.py            # kaggle entry point: obs -> L3(obs) -> action
world/                # L0
  mechanics.py        # verified constants + rules (from lab evidence)
  simulator.py        # forward simulator (state, action) -> state
  tilegraph.py        # distances, travel-time
  market.py           # price function
planner/              # L1
  economic.py         # 30-stage MILP/DP
  portfolio.py        # PortfolioPlan dataclass
schedule/             # L2
  day_planner.py      # per-day task assignment
  tasks.py            # Task dataclass, feasibility
dispatch/             # L3
  executor.py         # schedule executor
  reflexes.py         # emergency overrides
  entry.py            # kaggle glue
lab/                  # experiments & evidence (unchanged)
tests/                # per-layer fixture tests
docs/research/        # 008-hierarchical-architecture.md (this file's home)
```

## 6. Build order (each step must be verified before next)

1. L0 `world/`: port verified mechanics from docs + simulator skeleton.
   Verify: simulate one known goose day, matches lab result exactly.
2. L0 `tilegraph.py` + `simulator.py` workload probes: measure real
   ops/day for watering N plants, feeding M animals (lab evidence).
3. L2 `schedule/`: given a fixed portfolio (today's playbook), produce
   feasible day schedules; verify watering coverage hits 100% on 8 melons.
4. L1 `planner/`: port CP-SAT model, but with L2-derived capacity
   constraints (workload coefficients from step 2). Verify plan survives
   execution in sim.
5. L3 `dispatch/`: port current `playbook_agent.py` fieldwork as executor
   + reflexes; wire `entry.py`. Verify: end-to-end season, money > current.
6. Calibration loop: run seasons, feed op-logs back, re-solve L1.

## 7. Open questions (need owner decisions)

- Q1: Should L1 re-solve daily (rolling horizon, uses observed market) or
  once at season start? (Recommend: rolling — market drift is large.)
- Q2: Opponent modeling inside L1? (Recommend: defer; market inventory
  already reflects opponent sales.)
- Q3: Is MILP enough at L1 or do we need DP over price states?
  (Recommend: MILP first with piecewise-linear prices; upgrade only if
  solver gap > 5%.)

## 8. Naming

"Hierarchical MDP" = سلسله‌مراتبی. Layers = لایه. The stack above is 4 layers
(L0 state, L1 season economy, L2 day schedule, L3 turn dispatch).
