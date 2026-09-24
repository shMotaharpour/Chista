# Project: Worker Distribution & Scheduling Optimization on a 10×10 Grid

> This document evolved over a design conversation that moved through 4 stages:
> 1) Aligning on the problem statement → 2) Identifying gaps/contradictions →
> 3) Formulating the optimal computational model → 4) Defining tests and
> sections correct/refine earlier ones, and superseded material is marked
> explicitly rather than deleted, so the reasoning trail stays auditable.
>
> This is the English translation of the Persian original (which is NOT in this
> tree any more - see the note at the end), written as
> original, in Persian). Mathematical notation is language-independent and
> is reproduced verbatim; only prose is translated.

## Context
The user defined a worker scheduling and routing problem (a mix of VRPTW +
RCPSP + Pickup-Delivery with STRIPS-like preconditions) on a 10×10 grid,
which must be solved repeatedly (batch by batch, often with similar or
repeated conditions). The end goal: a precise formulation of the problem +
speed up repeated solves. Before formulating, both sides needed to be sure
they shared the same understanding of the problem, and any ambiguities or
contradictions in the problem statement needed to be resolved.

## Current understanding of the problem (summary)

### Environment
- A 10×10 grid (likely `(row, col)` coordinates); some cells carry tasks with
  sequencing dependencies between them.
- A central "shed"/warehouse reachable from 4 cells: (4,4), (4,5), (5,4), (5,5).
- Each cell has a `(typ, fill)` state: `typ ∈ {p, g, n, w, cs}`, `fill ∈
  {True, False}`. Worker actions change this state (like a farming state
  machine: plant → water → harvest, or building a structure).

### Workers
- Worker *i*'s cost equals Fibonacci(i), with Fib(0)=0, Fib(1)=1, Fib(2)=1,
  Fib(3)=2, Fib(4)=3, Fib(5)=5, ...
- Each worker has an earliest-availability floor (worker 0: from time 0;
  workers 1-8: from time 1 or later; workers 9-14: from time 3 or later) —
  these floors are **input to the problem on every call**, and the "final
  choice" (apparently: each worker's actual start time, ≥ its floor) is an
  optimization decision variable.
- Workers' initial placement into one of the 4 shed cells is based on
  "fewest workers currently present in that cell, at the entry time step,"
  in list order (4,4)→(4,5)→(5,4)→(5,5).
- Worker carrying capacity is unlimited.

### Tasks
- Some tasks = pick up an "item" from an origin/the shed and deliver it to
  a target cell.
- Item stock at the shed/origin cell is limited; the amount picked up from
  the shed is adjustable (any amount up to the stock cap), while picking
  up from another cell = that cell's entire stock.
- Some delivery pairs have a timing-sequence constraint (e.g., delivery A
  must be completed at least one step before B), without requiring the
  same worker.
- Overall time horizon: all tasks must be complete by the end of step 24.
- Objective: minimize total cost (sum of the Fibonacci costs of the workers
  used) while completing all tasks; output = number of workers used + each
  one's start time.
- Repeated calls with similar conditions ⇒ a caching/warm-start mechanism
  must be designed so repeated/similar solves reach an answer faster (or
  without a full recomputation).

### ⚠️ The table below is obsolete — the valid replacement is in the "Final
### Redefinition" section further down.
| # | Action | Precondition | Effect |
|---|------|----------|-----|
| 1 | `pass` | - | one turn elapses |
| 2 | `fd` | worker is carrying `w`; cell `typ∈{g,cs}` and `fill=True` | one `w` is removed from the worker |
| 3 | `mk[mod]` | cell `typ=n` | cell → `typ=mod`, `fill=True` |
| 4 | `put[item]` | if item is type g: cell `typ=g`,`fill=False`; if `cs_c`/`cs_s`: cell `typ=cs`,`fill=False`; worker must be carrying the item | cell `fill=True`; one item removed from worker's inventory |
| 5 | `pik[item,n=1]` | worker is in one of the 4 shed cells | `item∈{w,f,cs_c,cs_s,g,...}` is added to the worker, quantity n |
| 6 | `frtz` | worker is carrying `f`; cell `typ=p` | (exact effect not stated) |
| 7 | `wtr[f=False,p=False]` | if f=True: `f` must have already been delivered to this cell; if p=True: `plant` must have already been delivered | marks the cell "watered" (precondition for hvst) |
| 8 | `wet_hvst[x,n=1,me=False]` | precondition: `wtr` already done; `x∈{w,c,t,m,s}` | n units of x are harvested; if me=True, cell → `typ=n` |
| 9 | `dry_hvst[x,n=1,me=False]` | no `wtr` precondition; `x∈{ge,cm,sw}` + the wet options | same as above |
| 10 | `plnt` | cell `typ=n` | cell → `typ=p` |
| 11 | `dig` | cell `typ∈{p,w}`, or if `typ∈{g,cs}` then `fill=False` is required | cell → `typ=n` |
| 12 | `clct[s=23]` | — (this is a **global constraint**, not a worker action) | every worker who did a `wet_hvst`/`dry_hvst` must be back at one of the 4 shed-adjacent cells by step s |

## Identified ambiguities and contradictions

### ✅ Resolved (confirmed by the user)
1. **Task granularity:** confirmed — full planning with all 12 actions and
   their preconditions (STRIPS-like); tasks are not merely high-level
   pickup-delivery.
2. **Initial-placement order:** the contradiction was resolved. Correct
   naming of the shed cells:
   `NW=(4,4)`, `NE=(5,4)`, `SW=(4,5)`, `SE=(5,5)` — priority order:
   `NW → NE → SW → SE`.
   (Explanation of the earlier contradiction: the order the user first
   wrote, "(4,4)/(4,5)/(5,4)/(5,5)," did not match this mapping; the real
   order follows the new NW/NE/SW/SE naming.)
   Assignment rule: at each entry time step, the cell with the fewest
   workers present is chosen (using NW→NE→SW→SE to break ties); for
   workers entering simultaneously, they are processed in ascending
   worker-index order (the lower-indexed worker is assigned first).
3. **Worker pool cap:** unbounded/parametric — on every call, the number of
   workers and their availability thresholds are given as input (we do
   not assume a fixed cap in the model design).

4. **Movement:** ✅ resolved. Movement is exactly one cell per step,
   4-directional (no diagonal), with no obstacles (the grid is fully open;
   multiple workers can share a cell simultaneously). On each time step a
   worker either moves or performs one of the 12 actions — never both.

7. **`me` in `wet_hvst`/`dry_hvst`:** ✅ resolved. It is part of each task's
   definition (fixed/input to the problem), not an optimization decision
   variable.
8. **Picking up from the shed vs. from other cells:** ✅ resolved + an
   important correction.
   - `pik[item,n]` is **only valid at shed cells**; the quantity n is
     freely chosen up to the shed's stock cap.
   - `pik` doesn't work at all outside the shed. Picking up from other
     cells happens **only via `wet_hvst`/`dry_hvst`**, and the amount
     harvested is always **exactly equal to that cell's entire stock**
     (not adjustable) — this stock is a variable/value on the cell itself.
   - ⚠️ **Newly discovered point:** so each cell, besides `(typ, fill)`,
     must also carry a numeric "qty" (harvestable stock). Details are
     still open (see item 11 below).
9. **Time indexing:** ✅ resolved. Steps 0 through 24 (25 time
   points/states), but a maximum of "24 actions" can be executed by each
   worker across the horizon (i.e. 24 transitions between 25 states); step
   24 (the last state) is the result of the final actions. `clct`'s
   default s=23 means one step before the end.
10. **`clct[s=23]` as a global constraint, not a worker action:** ✅
    implicitly confirmed (the user described it as a scheduling
    constraint, not a worker action).

9-a. **`wtr`'s precondition:** ✅ resolved. It's only executable on cells
with `typ=p` (after `plnt`).
11. **The `qty` variable on a cell:** ✅ resolved (with an important
simplification). A cell's harvestable amount is not separately
stored/simulated in cell state; it is exactly the same `n` given as a
fixed instance input to the `wet_hvst`/`dry_hvst` action in that task's
definition. That is, "how much can be harvested from this cell" is part
of the problem's input data, not a dynamic variable that must be
simulated/computed. ⇒ we can simplify the model: each "harvest task" = a
predetermined (cell, item x, fixed amount n, wet/dry, me).
12. **Cross-feeding loop between the g/cs and p paths (an important
discovery, finalized in item 19 below):**
    - `typ=p` (farming) cells produce item `w` via `wet_hvst[x=w]`; `w` is
      consumed as feed by `fd` on `typ∈{g,cs}` cells.
    - Item `f` (fertilizer) is obtained from a separate action called
      **`f_clc`** (not `dry_hvst`) on `typ∈{g,cs}` cells; `f` is consumed
      by `frtz` on `typ=p` cells. (Full details in item 19.)
    - Other than this pair of exchange paths, the two paths are entirely
      independent.
6. **`frtz`'s effect:** ✅ resolved. It just sets a flag on the cell (no
effect on n or typ).
5-a. **Meaning of `typ=w`:** ✅ resolved. It means the cell is
"weed-covered"; weeds must be removed with `dig` to bring the cell back to
`typ=n` (empty/usable). So `typ=w` is an initial/natural cell state, not
the product of another action.

14. **`care`** — ✅ resolved. Executable on `typ∈{g,cs}` cells with
`fill=True`; no item precondition (like `wtr`'s default case); just sets a
flag on the cell. **Important:** this flag is not a precondition for any
other action — performing `care` is itself part of the "target"/goal of a
task (similar to how put/fd/... can themselves be the final goal of a
task, not necessarily a stepping stone to a later one).

> ⚠️ **The user's explicit instruction:** "Never assume anything yourself;
> if you're not sure about the explanations, you must ask so I can
> confirm." So we do not start the formulation phase until the following
> items (found on a closer review of the 12+1 action table) are also
> clarified.

15. **The `mk` vs. `put` contradiction:** ✅ resolved + **an important
correction**. My original text was wrong: after `mk[mod]` builds the new
typ, it sets `fill` to **False** (not True). So the correct sequence is:
`n --[mk[mod]]--> (typ=mod, fill=False) --[put[item]]--> (fill=True)`.
16. **The `p` parameter in `wtr[f,p]`:** ✅ resolved. It's essentially
synonymous with "this is a watering right after a fresh planting" —
consistent with the new precondition (`wtr` only on `typ=p`) and adds no
independent new constraint to the model; it simply flags that, in each
planting cycle, `wtr` (with p=True) must be redone after `plnt`.
18. **Effect of `fd`/`put`/`pik`:** ✅ resolved. `pik`: a specified amount
is removed from the shed and added to the worker. `put`: flips `fill`
from False to True. `fd`: **has no persistent effect within the problem
domain** (it just needs to happen, consuming item `w`); the user allowed
that, if useful for modeling, we could define an internal flag ("fd was
done") for g/cs cells — this is **our own design decision**, not a domain
fact.

19. **`f_clc`** — a separate action (not `dry_hvst`!) that produces item
`f` (fertilizer). ⚠️ This corrects something that was previously (wrongly)
confirmed ("g/cs → dry_hvst(n=1) → f").
    - ✅ It works for **any item** that has filled the cell (`typ∈{g,cs}`),
      not just g/cs_c/cs_s.
    - ✅ The cell's `fill` is unrelated to this action and doesn't change;
      but **in practice it can only be done once per cell**, and which
      cells are even eligible for `f_clc` **is determined by the
      problem's input/instance data**, not a generally inferable rule.
    - ✅ `cs_w` is a completely separate item/code (not a typo for `cs_s`).
    - ✅ `care` and `f_clc` are entirely independent of each other.
    - ⚠️ **This point (eligibility comes from the input side) is a key
      insight for `f_clc`:** similar to `wet_hvst`/`dry_hvst` (where `n`
      comes from the instance), it seems executing these "production"
      actions is also conditioned on that (cell, action) being given as a
      specific task in the input; a more general question is asked in
      item 20.

20. **✅ Key architecture question — resolved (a very important insight):**
Every call's input includes "task functions" that fully specify each
task's conditions/parameters. The solver **never invents new work** — only
a translation layer is needed that unrolls each given "task" (e.g., a
`wet_hvst` on cell X) into its chain of prerequisite actions (per the
fixed rules documented in this doc, e.g.: `wtr` must precede `wet_hvst`,
and if the cell wasn't already `p`, `plnt` must precede that). This
unrolling is **mechanical and deterministic** (based on the fixed action
rules), not a creative search. ⇒ **Problem class = RCPSP + Routing with a
deterministic precondition-expansion layer**, not open-ended AI Planning.
This is a large simplification for the computational formulation.
21. **`cs_w`:** ⚠️ **The user stated no such item exists** and asked me to
point out exactly where they had said this, so they could correct it. The
user's exact quote (from an earlier answer): *"f comes from a function
called f_clc, and cells that have g, cs_c, or cs_w as their filling
item..."*. This needed final confirmation: most likely a typo for `cs_s`
(consistent with the known `put`/`pik` items `g`, `cs_c`, `cs_s`) — **to
be confirmed in the next question**.
23. **The `care` action:** ✅ resolved. Exactly like `f_clc` — doable once
per cell, and cell eligibility comes from the input/instance (not a
general rule over every `typ∈{g,cs}` cell with `fill=True`).

21. **`cs_w`:** ✅ finally confirmed — it was a typo for `cs_s`. Final
known item set: `w, f, g, cs_c, cs_s` (+ farming/animal products
`c,t,m,s,ge,cm,sw`).
24. **`dry_hvst`'s use on `g`/`cs`:** ✅ resolved. Contrary to the initial
assumption, `dry_hvst` applies both on `typ=p` and on `typ∈{g,cs}`; on
`g/cs` cells it collects products `ge, cm, sw` (separate from `f_clc`,
which is specific to item `f`).
25. **Valid values of `mod` in `mk[mod]`:** ✅ resolved. Only `g` or `cs`.
26. **`frtz`'s limitation:** ✅ resolved. Exactly like `care`/`f_clc` — once
per cell, eligibility comes from the input/instance.
27. **The "at most once" constraint for `care`/`f_clc`/`frtz`:** ✅
resolved. This is a **natural property of the input data/task list**, not
a hard constraint that must be enforced in the model (in real input it is
never repeated twice for the same cell).

28. **The `fill` precondition for `dry_hvst` on `g/cs`:** ✅ resolved. Yes,
it's required (`fill=True`), but this state is guaranteed by the input and
doesn't need to be checked in the algorithm (a more general point in item
29).

### ✅ Final architecture correction — the minor_task / major_task
### framework (final replacement for item 20)
29. **The user's precise framework (replacing all my earlier guesses):**
    - **`minor_task`:** an atomic action that consumes exactly **one
      worker's turn** (the same 12+2 actions documented in this doc:
      `pass, fd, mk, put, pik, frtz, wtr, wet_hvst, dry_hvst, plnt, dig,
      clct, care, f_clc`, plus `move`). Its actual execution is
      conditioned on satisfying constraints/preconditions.
    - **`major_task`:** a chain of `minor_task`s that must be executed **in
      a specific order** (per the fixed precondition/effect rules
      gathered in this doc) for the task to be complete.
    - **The problem input can include both kinds** (standalone
      `minor_task`s as well as composite `major_task`s).
    - **Our program's job:** take each given `major_task` and, by applying
      the rules/conditions (the action precondition/effect table),
      "break" it (decompose it) into the ordered chain of `minor_task`s
      that make it up — then all the resulting `minor_task`s (from
      `major_task`s) and the independently given `minor_task`s must be
      scheduled/assigned/routed together.
    - This "breaking down" is rule-governed by the very definition of the
      `major_task` (which is an order-driven chain), not an open search —
      consistent with the earlier conclusion that this is "not hard and
      well-defined."
    - ⚠️ A remaining point for the formulation phase (not a domain
      ambiguity, but an implementation detail): we need to pin down
      exactly which `major_task`s exist in the problem domain (names /
      common patterns, like "a full plant-harvest-replant cycle") and
      what each one exactly unrolls into — we'll complete this in the
      formulation/implementation phase, with real examples from the user.

### ✅ Final minor/major correction (the user corrected again — this
### version is the valid one)
30. My previous item's categorization ("conditional minor," etc.) was also
wrong. **The final, correct definition:**
    - **There is no "conditional minor."** All 14 actions (`pass, move,
      pik, mk, put, wtr, wet_hvst, dry_hvst, plnt, dig, frtz, fd, care,
      f_clc`) are uniformly `minor_task`s — no exceptions.
    - **The golden rule:** if a `minor_task` appears directly in the input
      (not wrapped in a `major_task`), **its precondition is already
      guaranteed/accounted for by the input** — the model must not go
      looking for how to satisfy its precondition, or invent an extra
      action for it. (If executed without its precondition being met, it
      simply "burns a turn" — no effect; but this should never happen in
      valid data.)
    - **`major_task`** means: (a) an ordering of several `minor_task`s
      where one depends on a `minor_task` **on a different cell**
      (cross-cell dependency, like `pik` at the shed before
      `put`/`frtz`/`fd` elsewhere), or (b) a fixed ordering of several
      minors that has been given a **specific name** (a named recipe).
    - ⇒ **The important implication for the model:** the solver's job is
      purely to "unroll" exactly what the input gives it (whether a
      standalone minor, or a major with its components specified) — it
      should never "guess" a new prerequisite action; any needed chain is
      explicitly specified as a `major_task` in the input.
    - **Pending:** the exact, named list of the domain's real
      `major_task`s (with real names instead of codes, e.g. `cs_c`=cow/
      barn, `g`=goose/coop, `m`=melon, `sw`=sheep's wool) — the user was
      going to define these for me one by one.

## ✅ Final, valid redefinition (fully replaces earlier sections) — the
## user's version, with real names

### Cell State
```
cell_state = {
  type: None | Plant | Weed | Pasture | Coop
  is_wheat: False | True                    # meaningful only for type=Plant
  watered: False | True | N/A                # Plant only
  fertilized: False | True | N/A             # Plant only
  fed: False | True | N/A                    # Pasture/Coop only
  produced_fer: False | True | N/A           # Pasture/Coop only
  occupied: False | True | N/A               # Pasture/Coop only
  cared: False | True | N/A                  # Pasture/Coop only
  n_yield: INT                                # harvestable amount (crop or animal product)
  harvest_to_None: False | True | N/A         # whether HARVEST resets typ to None
}
```
This state is given only for the cells relevant to the instance at hand;
every other grid cell is merely passable terrain (a worker does nothing on
it besides moving through).

### Minor Actions (each exactly 1 worker turn)
- **Movement:** `NORTH, SOUTH, EAST, WEST, PASS`
- **Shed/inventory:**
  - `PICKUP <item> [n]` — from the shed, if n is available.
  - `PLACE <item> [n]` — either places an animal on the ground (when the
    worker is standing on the matching structure; n is ignored, always 1
    animal), or (when adjacent to the shed) drops the item into the shed.
  - `DROP` — when adjacent to the shed (one of the 4 center cells), dumps
    the worker's entire inventory into the shed.
  - Items: animals `[cow, sheep, goose]`, crops `[melon, tomato,
    strawberry, carrot, wheat]`, animal products `[milk, wool, egg]`.
- **Plants:**
  - `PLANT <crop>` — cell `type=None` → `type=Plant, watered=False`.
  - `WATER` — cell `type=Plant` → `watered=True`.
  - `HARVEST` — the `n_yield` amount of the crop is added to the worker's
    inventory and `n_yield=0`; if `is_wheat=True` the product is wheat
    (usable in `FEED`); if `harvest_to_None=True` the typ reverts to
    `None`.
  - `FERTILIZE` — if the worker has fertilizer: cell `Plant` →
    `fertilized=True`; one fertilizer is removed from the worker's
    inventory.
- **Animals:**
  - `BUILD_COOP` — `None` → `Coop`, `occupied=False, fed=False,
    cared=False`.
  - `BUILD_PASTURE` — `None` → `Pasture`, `occupied=False, fed=False,
    cared=False`.
  - `FEED` — if `occupied=True` and the worker has wheat: it's consumed,
    `fed=True`.
  - `COLLECT_FERTILIZER` — if `occupied=True` and `produced_fer=True`: one
    fertilizer is added to the worker's inventory, `produced_fer=False`.
  - `CARE` — if `occupied=True`: `cared=True`.
  - `HARVEST` (on animals) — if `occupied=True`: the `n_yield` amount of
    the animal product is added to the worker's inventory, `n_yield=0`.
- **Terrain:** `DIG` — removes a plant, a weed, or an *empty* structure
  (coop or pasture with no animal) and sets `type=None`; a structure with
  an animal on it cannot be dug.

### Major Actions (named chains of Minors)
- `feed`: (get wheat from wherever possible) → `FEED`
- `frtz`: (get fertilizer from wherever possible) → `FERTILIZE`
- `wet_harvst`: `WATER` → `HARVEST`
- `plnt(crop)`: `PLANT crop` → `WATER`
- `wet_harvst_plnt(crop)`: `wet_harvst` → `plnt(crop)` (i.e.
  `WATER→HARVEST→PLANT→WATER`)
- `frtz_water`: `frtz` → `WATER`
- `clct(s=23)`: a global constraint — every worker who did a final
  `HARVEST` must, by step s, **both reach one of the 4 shed-adjacent cells
  and have done `DROP`**.

**General rule:** the program's input is a mix of minors and majors; the
program must break the majors down into their constituent minors (per the
definitions above), and ultimately output, for each worker, a sequence of
minor_tasks, and, for the whole problem, the number of workers used plus
each one's start time.

### ✅ This round's ambiguities — all resolved
31. **`PLACE`'s dual behavior:** confirmed — it really does have two
distinct behaviors (an animal onto a matching structure; any other item
next to the shed); `DROP` = a full inventory dump (as opposed to `PLACE`,
which is selective/partial).
32. **Source of wheat/fertilizer:** confirmed — both (`wheat`,
`fertilizer`) are also `PICKUP`-able from the shed (in addition to
`HARVEST`/`COLLECT_FERTILIZER`).
33. **The `produced_fer` mechanism:** confirmed — it's a property given
in advance in the initial state/instance; the model doesn't need to
simulate how it becomes True (consistent with the same general rule:
"eligibility comes from the input").
36. **Scope of "delivery":** implicitly confirmed — wheat's destination
is "feeding an animal" (via `FEED`), not an arbitrary target cell;
delivery destinations are the same two cases (animal→structure,
item→shed) plus "direct consumption within an action" (like wheat in
`FEED`, fertilizer in `FERTILIZE`) — there is no separate "arbitrary
target cell" destination beyond these.

34. **The `fed`/`produced_fer` relationship:** ✅ resolved — entirely
independent.
35. **Scope of `harvest_to_None`:** ✅ resolved — meaningful only for
`HARVEST` on `Plant`.

## ✅ Understanding of the problem is considered complete
With items 1-35 complete, the following full model has emerged:
- A 10×10 grid environment, a shed with 4 entry cells (order
  NW→NE→SW→SE).
- Workers with Fibonacci cost, an entry time floor (input on every call),
  initial placement based on least congestion.
- Movement: 4-directional, 1 cell per step; each step = either a move or
  one `minor_task`.
- Horizon: 24 actions/steps per worker (25 time-point states).
- `cell_state` and 14 `minor_task`s and 7 `major_task`s (+ the global
  `clct` constraint) as finally defined above.
- The instance input is a mix of required minors/majors (with
  preconditions already guaranteed) + timing-sequence constraints between
  some deliveries + the grid's initial state.
- Objective: minimize total worker cost while completing all tasks by step
  24; output = number of workers + each one's start time + each one's
  minor_task sequence.
- Caching/warm-start is needed for repeated/similar calls.

## ✅ Phase 3 started — the user confirmed ("Yes, let's go to phase 3")

---

# Phase 3: Optimal Computational Formulation

## 1. Classifying the problem class
This problem is a combination of:
- **A multi-vehicle VRPTW with variable activation cost** (each worker = a
  "vehicle" with a fixed Fibonacci cost + a start time window),
- **RCPSP** (sequencing dependencies between tasks, both from unrolling
  `major_task`s and from explicit delivery constraints),
- **Pickup-Delivery with stock limits** (a shed with globally limited
  stock + each worker's inventory as a changing resource).

**Key point for computational efficiency:** because the grid is fully open
(no obstacles) and movement is always exactly one cell per step,
4-directional, **the travel time between any two cells = their Manhattan
distance** — a fixed, precomputable value. So **there is no need for
per-turn position variables** — the whole problem can be formulated like a
standard VRPTW with "travel time = Manhattan distance"; this is a major
computational simplification.

## 2. Data pipeline (before reaching the solver)
1. **Unrolling `major_task`s:** per the fixed recipes in this doc (`feed`,
   `frtz`, `wet_harvst`, `plnt(crop)`, `wet_harvst_plnt(crop)`,
   `frtz_water`), each `major_task` is unrolled into its constituent
   `minor_task`s, along with an ordering constraint between them (which
   can even be carried out by different workers — only their relative
   timing must be preserved).
2. **Building the final `minor_task` list:** each item = `(id, cell,
   action_type, params)`.
3. **Building the precedence/ordering graph:** an edge i→j if i must be
   finished before j (from unrolling majors + the user's explicit
   delivery constraints).
4. **Extracting resource constraints:** for every item
   (wheat/fertilizer/cow/sheep/goose/crops/...): which minors "produce" it
   (PICKUP, HARVEST) and which "consume" it (PLACE, FEED, FERTILIZE, DROP
   as a deposit).

## 3. Mathematical formulation — independent of any library
This formulation is the formal definition of the optimization problem
(sets/parameters/decision variables/constraints/objective), with no
(section 4) simply solve this same mathematical model with their own
tools.

### Sets and indices
- Candidate workers: `w ∈ {0,...,K-1}` (for now K comes directly from the
  input; an automatic bound-computation formula will be finalized later).
- Final `minor_task`s (after unrolling all majors): `i ∈ T = {1,...,N}`.
- Items: `k ∈ I = {wheat, fertilizer, cow, sheep, goose, melon, tomato,
  strawberry, carrot, milk, wool, egg}`.
- Time steps: `t ∈ {0,...,24}`.
- Shed entry cells (fixed priority): `E = (NW, NE, SW, SE)`.

### Parameters (derived from the instance input)
- `fib(w)`: worker w's activation cost.
- `earliest(w)`: worker w's earliest-availability floor.
- `d(c1,c2)`: the Manhattan distance between two cells.
- `cell(i)`, `action(i)`: the cell and action type of `minor_task` i.
- `stock(k)`: item k's initial stock at the shed.
- `produce(i,k)`, `consume(i,k)`: how much of item k is produced/consumed
  by executing `minor_task` i.
- `P ⊆ T×T`: the set of precedence/ordering edges (from unrolling majors +
  the user's explicit delivery constraints) — `(i,j)∈P` means i must
  strictly finish before j.
- `S`: a collection of subsets of T that must be carried out by **a single
  worker** (the `feed`, `frtz`, and animal-transfer chains).
- `H = 24` (horizon), `s = 23` (`clct`'s default).

### Decision variables
- `u_w ∈ {0,1}`: whether worker w is active.
- `σ_w ∈ ℤ`: worker w's start time (meaningful only if `u_w=1`;
  `earliest(w) ≤ σ_w ≤ H`).
- `τ_i ∈ {0,...,H}`: `minor_task` i's execution time.
- `a_i ∈ {0,...,K-1}`: the worker responsible for executing `minor_task` i.
- `cell0(w)`: worker w's entry cell — a deterministic function of `(u,σ)`
  per the placement rule (least congestion, order NW→NE→SW→SE).
- The relative order of a worker's own tasks (for building its route) — an
  auxiliary sequencing/permutation variable per worker.

### Objective function
```
minimize  Σ_w  fib(w) · u_w
```

### Constraints
1. **Unique assignment:** every `i∈T` is assigned to exactly one active
   worker: `u_{a_i} = 1`.
2. **Route sequencing (travel time = Manhattan distance):** if i
   immediately precedes j on the same worker's route: `τ_j ≥ τ_i + 1 +
   d(cell(i), cell(j))`.
3. **Route start:** for each worker's first task: `τ_first ≥ σ_w + 1 +
   d(cell0(w), cell(first))`. **⚠️ Fixed off-by-one (found by the user,
   who noticed a worker's usable capacity was being computed as one turn
   more than reality — a worker starting at `σ_w` can fit at most
   `H − σ_w` actions, not `H − σ_w + 1`):** entering the grid at `σ_w`
   only means the worker is *standing* at `cell0(w)`, not that it has
   already spent a turn there. Reaching `cell(first)` still costs
   `d(cell0(w), cell(first))` movement turns *plus* the first task's own
   turn — exactly the same `+1` that constraint 2 already charges between
   any two consecutive tasks on a route. The formula used to omit that
   `+1` for the very first task only (silently treating a worker's arrival
   as if it were itself a free action), which is why it went unnoticed:
   every solver, `verify.py`, and the independent brute-force checker all
   implemented the same shortfall identically, so no cross-check could
   catch it.
4. **Strict precedence:** `∀(i,j)∈P: τ_i < τ_j` (two dependent tasks, even
   done by two different workers, must not be simultaneous; independent
   tasks can be simultaneous and done by different workers).
5. **Single-worker requirement:** `∀ Grp∈S, ∀ i,j∈Grp: a_i = a_j`.
6. **Global shed stock:** `∀k∈I: Σ_{i: action(i)=PICKUP} produce(i,k) ≤
   stock(k)`.
7. **Each worker's inventory must never go negative:** `∀w,∀t:
   Σ_{i: a_i=w, τ_i≤t} (produce(i,k) − consume(i,k)) ≥ 0, ∀k`.
8. **Time horizon:** `∀i: 0 ≤ τ_i ≤ H`.
9. **The `clct(s)` constraint:** if worker w did at least one `HARVEST`, a
   `DROP` must exist at one of the cells in `E` with `τ ≤ s`, occurring
   after that worker's last `HARVEST`.
10. **The initial-placement rule:** `cell0(w)` is computed by the
    deterministic function "least congestion, priority NW→NE→SW→SE,
    processed in ascending worker-index order for simultaneous entries."
11. **✅ Prefix worker activation (a new constraint — missing from the
    formulation until a user's manual walkthrough of a real instance
    surfaced it):** the set of active workers must be a **contiguous
    prefix** by worker index -- worker 12 can't be active while worker 9
    isn't. Formally: `∀ w, w'` with `index(w) < index(w')`:
    `u_{w'} = 1 ⟹ u_w = 1` (equivalently `u_{w'} ≤ u_w`). Before this was
    caught, the models (CP-SAT and SciPy in particular, which leave `u_w`
    a free decision variable) had no guarantee of this -- only an
    optional symmetry-break for *equal-cost* tiers (see "Symmetry
    breaking" below), not a hard domain constraint. The shared greedy
    decoder (section 4) always satisfies this structurally, since the
    next worker it opens is always the next one in a pre-sorted list,
    never a skip.

---

## 4. Independent implementation approaches (per the user's explicit
## request)
The user was explicit: the libraries must not be combined; each tool must
independently solve this same mathematical model above (not one as the
main engine with the rest merely helping). Originally three tools —
(MILP), added later on the same basis. All four use a shared
infrastructure (the section-2 pipeline: unrolling major→minor, computing
Manhattan distances, the precedence graph, resource constraints), but
their search/optimization engines are entirely separate.

| Tool | Nature | Role in this project |
|---|---|---|
| **CP-SAT (OR-Tools)** | Exact solving with logical/numeric constraints (Constraint Programming) | Builds the full problem model directly as variables and constraints and solves it with an optimality guarantee (or a proof of infeasibility). |
| **SciPy/HiGHS** | Exact solving as a Mixed-Integer Linear Program | Builds the same model as a MILP (every constraint hand-linearized — big-M for conjunctive gating, exact AND-linearization for logical ANDs) and solves it to proven optimality with `scipy.optimize.milp`. A second, independent exact cross-check on CP-SAT. |

> CP-SAT and SciPy/HiGHS are both "exact" methods and naturally serve as
> the correctness/optimality benchmark for the other two (and cross-check
> optimality guarantee but better scalability on very large instances.
> Between the two exact methods, CP-SAT is expected to scale better: its
> native reification and global constraints (`OnlyEnforceIf`,
> `AddReservoirConstraintWithActive`) express things directly that plain
> MILP must hand-linearize into extra rows and auxiliary variables.

Input: a priority/rank for each `minor_task` + (optionally) a suggested
worker for it.
Algorithm: process tasks in priority order; for each task, pick the first
available/existing worker (per the suggestion, or in its absence the
cheapest active worker or the next one) that, respecting all the section-3
constraints, can perform that task at the earliest possible time; if no
worker (active or new) can do it within the horizon → a large penalty
(infeasible). Output: total cost + the full schedule.

### 4.1 Mapping onto CP-SAT
The same sets/parameters/variables/constraints of section 3 map directly
onto CP-SAT:
- `u_w, σ_w, τ_i, a_i` → standard CP-SAT boolean/integer variables.
- **Each worker's route (constraints 2 and 3):** via a "circuit" structure
  (the usual VRP pattern in CP-SAT, using `AddCircuit`/
  `AddMultipleCircuit`) that fixes the visiting order of the tasks
  assigned to each worker; a virtual "shed" node is each worker's
  origin/destination.
- **Each worker's inventory (constraint 7):** a cumulative dimension along
  the route, similar to a `Dimension` in RoutingModel but built manually
  with CP-SAT.
- **Symmetry breaking (meaningful only for CP-SAT):** workers with equal
  cost are interchangeable — a constraint `u_i ≥ u_{i+1}` within each
  equal-cost tier; sorting candidate workers by (cost, earliest-start)
  before building the model.

Each "trial" produces a parameter vector: each `minor_task`'s execution
priority + (optionally) a suggested worker. A **shared greedy decoder**
(below) turns these parameters into a real `(u,σ,τ,a)` that satisfies all
sampler) searches this parameter space to minimize the objective +
penalty.

Each chromosome = a permutation of tasks + a worker assignment for each.
The same shared greedy decoder turns the chromosome into a real
`(u,σ,τ,a)` and a fitness value (fitness = cost + penalty);
selection/crossover (order crossover, suited to permutations)/mutation
operators improve the population across generations.

### 4.4 Mapping onto SciPy/HiGHS (MILP)
The same sets/parameters/variables/constraints of section 3 map onto a
MILP solved by `scipy.optimize.milp` — but MILP has no native reification
or global constraints, so each piece CP-SAT gets built-in has to be
hand-linearized:
- **Conjunctive gating** ("this inequality holds whenever these booleans
  are all 1, and is vacuous otherwise") → a big-M inequality, shifting the
  right-hand side by `M` per un-satisfied gate.
- **Logical AND of binaries** (`z = a ∧ b ∧ …`) → an exact, big-M-free
  linearization (`z ≤` each literal, plus `z ≥ Σliterals − (m−1)`); used
  to build the "this event contributed to this worker's running balance"
  indicators the per-worker inventory check (constraint 7) needs, in
  place of CP-SAT's `AddReservoirConstraintWithActive`.
- **The initial-placement rule (constraint 10):** the same sequential
  argmin-with-tiebreak construction CP-SAT uses (worker start times are
  fixed, so the processing order is known when the model is built — only
  which workers end up active is a real decision), expressed as two
  big-M inequalities per (worker, candidate cell) instead of CP-SAT's
  reified `AddBoolAnd`/`AddBoolOr`.
- **Route sequencing/start (constraints 2 and 3):** applied to every
  pair, not just consecutive tasks in the final route — exact rather than
  an over-approximation, because Manhattan distance's triangle inequality
  makes this a property of the math model itself, not of CP-SAT
  specifically.

This yields a correct, independent cross-check of the same formulation,
but with a constraint count that grows faster than CP-SAT's (the extra
big-M rows and AND-linearization variables for gates CP-SAT gets
natively) — expect it to scale worse than CP-SAT on larger instances,
even though both are exact.

> CP-SAT and SciPy/HiGHS are both "exact" methods and naturally serve as
> with a weaker optimality guarantee but better scalability on very large
> instances.

### ✅ Computing the worker cap/floor (K) — per the user's confirmation
- **For now (per the user's decision):** K is taken directly from the
  input (no automatic computation).
- **Future direction (open, needs a real example to finalize):** an
  optional bound-computation function will also be designed, deriving a
  lower/upper bound from the final number of `minor_task`s (`N_actions`)
  and the number of distinct target cells (`N_cells`) (initial draft:
  `LB=max(3, ceil(N_actions/24))`, since the first three workers are
  nearly free; `UB` is still open) — this function will be implemented in
  code as an **optional** fallback for when the input doesn't give K
  explicitly; an explicit input always takes priority.
  range.

### ✅ Solve time limit — "should be configurable" (confirmed by the user)
All three solvers take a config parameter: `time_limit_seconds` (or
`None` = unlimited, until optimality is proven / evolution finishes).
Suggested defaults: a few seconds for CP-SAT (since it usually reaches the
optimum quickly), and a configurable number of iterations/generations for

## 5. Caching / Warm-Start mechanism (for repeated calls, independent of
## which solver is chosen)
1. **Exact-match cache:** a canonical hash of the entire instance input →
   if seen before, the cached answer is returned without re-solving
   (SQLite/Redis/file, persistent across calls).
2. **Warm-start:**
   - CP-SAT: `CpModel.AddHint(...)` with the best solution from a similar
     prior instance.
     instance (`enqueue_trial`).
     a similar instance, instead of a fully random population.
3. **(More advanced, a later phase):** splitting the solve into a
   "structural" layer (cacheable) and a "numeric scheduling" layer (fast,
   always re-solved).

## ✅ Phase 4 started — the user confirmed ("Great. Let's move forward")

---

# Phase 4: Defining Tests and Implementation

## Project structure (Python)
```
chistawrs/
  models.py              # Cell, CellState, Item, MinorTask, MajorTaskSpec (recipes), Worker, Instance, Solution
  major_task_expander.py # unrolls each major_task into its ordered minor_task chain (the 7 recipes in this doc)
  instance_compiler.py   # raw input -> the final minor_task list + precedence graph P + single-worker groups S + resource constraints
  distances.py           # Manhattan distance, the initial placement rule (NW->NE->SW->SE, least congestion)
  fibonacci.py           # worker cost
  solvers/
    cpsat_solver.py       # CP-SAT formulation and solve (doc section 4.1)
  cache.py                # instance hashing, solution storage/retrieval, warm-start hints
  output.py               # output formatting: worker count + each one's start time + each one's minor_task sequence
tests/
  test_major_task_expander.py   # each of the 7 recipes: sample input -> the expected minor sequence
  test_instance_compiler.py     # correctly building the precedence graph/single-worker groups/resource constraints from a sample instance
  test_placement_rule.py        # the NW->NE->SW->SE rule with several worker-simultaneity scenarios (the user's exact example)
  test_fibonacci_cost.py
  test_cpsat_solver.py          # several small hand-crafted instances with a pre-computed optimal answer
  test_invariants.py            # property-based (Hypothesis): inventory never negative, global shed stock never violated, clct honored, each minor executed exactly once
  test_cache.py                 # exact-match cache: a second call with an identical instance returns the same answer without re-solving
```

## Test strategy
1. **Deterministic unit tests:** for `major_task_expander` (the 7
   recipes), `distances` (the placement rule, with the user's exact
   example: worker 0&4→NW, worker 1→NE, worker 2→SW, worker 3→SE),
   `fibonacci.py`.
2. **Small instances with a hand-derived answer:** a few small scenarios
   (2-3 workers, 3-5 tasks) whose optimal answer is verifiable by
   hand/simple computation; run on CP-SAT and compared against the
   expected answer.
3. **Invariant/property-based tests (with Hypothesis):** for randomized
   (but valid) instances, automatically checking that no constraint
   (inventory, horizon, precedence, clct, single-worker) was violated —
   independent of which solver was used.
4. **Cross-solver consistency test:** on identical instances (small to
   medium), CP-SAT's cost (a proven optimum) is used as a lower bound for
5. **Cache test:** a repeated call with the same instance should return
   the same answer quickly, without re-invoking the solver.

## Suggested implementation order (Milestones)
1. `models.py` + `major_task_expander.py` + `instance_compiler.py` +
   `distances.py` + `fibonacci.py` + their unit tests (no solver yet) —
   the foundation all three solvers build on.
2. `solvers/cpsat_solver.py` + small-instance tests (the
   correctness/optimality reference).
4. `cache.py` + `output.py` + cache and end-to-end tests.

## ✅ Milestone 1 done and pushed
`models.py`, `fibonacci.py`, `distances.py`, `major_task_expander.py`,
`instance_compiler.py` + 23 unit tests — on branch
`claude/worker-distribution-optimization-7wo23y`, commit `d0ae981`. This
doc was also moved into `docs/problem-formulation.md` (now split into
`.fa.md`/`.en.md`).

---

# Two new user requests (before Milestone 2)

## a) A bilingual version of the formulation doc
`docs/problem-formulation.md` (Persian, at the time) had to become two
files:
- `docs/problem-formulation.fa.md` — the same content (just renamed/moved).
- `docs/problem-formulation.en.md` — this file: a complete, accurate
  English translation, keeping section 3's structure/tables/math formulas
  verbatim (mathematical notation is language-independent; only the
  Persian prose is translated).
- `README.md` links to both.

## b) ✅ Confirmed need: tests validating solver-solution optimality
The user asked whether we need tests that validate the actual optimality
of solvers' real solutions — **yes, this was a genuine gap in the test
strategy above** (the "small instances" and "cross-solver consistency"
items above only gestured at this superficially). The refined plan:

### 1. An independent "verifier" (`chistawrs/verify.py`)
A function `verify_solution(instance, solution) -> VerificationResult`
that, **entirely independent of any solver's internal logic**, walks each
worker's route by hand and re-checks all 10 constraints of the
mathematical formulation (doc section 3) from scratch: unique assignment,
travel-distance/route sequencing, route start, strict precedence, the
single-worker requirement, global shed stock, each worker's inventory
(never negative), the time horizon, the `clct` constraint, and that the
reported cost equals the real `Σ fib(w)·u_w`. Output: an exact list of any
violated constraints + the computed cost.
- `tests/test_verify.py`: we deliberately construct broken solutions
  (negative inventory, precedence violations, clct violations, duplicate/
  missing tasks, single-worker violations) and make sure the verifier
  catches each one.
- **All three solvers, in every one of their tests, must first pass
  through this verifier** — before any cost comparison.

### 2. A library of small "known-optimal" instances
### (`tests/fixtures/hand_solved_instances.py`)
A handful of very small instances (2-6 tasks, 1-3 workers) whose optimal
cost is known by hand/logical computation (e.g.: a single task → only
worker 0 (cost 0) should be used; two tasks with a precedence constraint
on far-apart cells that's only achievable with 2 workers; and so on).

### 3. Brute-force cross-validation independent of the CP-SAT model
### (`tests/test_brute_force_cross_check.py`)
A completely separate enumerator (plain Python, with no dependency
whatsoever on the CP-SAT formulation) that, for small instances (e.g. ≤4
candidate workers × ≤6 tasks), tries every possible assignment/ordering
and finds the true optimal cost. **This is the most important test** —
because the other tests only measure a solver's "internal consistency,"
while this one can also catch a bug in the CP-SAT formulation itself (not
just its implementation). CP-SAT's answer on these instances must exactly
match the brute-force result.

- Their solution must always pass the verifier.
- Their cost must never be lower than CP-SAT's proven optimum (if it is,
  there's a bug somewhere — either the verifier accepted an invalid
  solution, or CP-SAT is missing a constraint).
- On the very small instances (the same brute-force ones), they're
  expected to reach the optimum within a few retries.
- On larger instances, an acceptable tolerance (e.g. at most a 10-20% gap
  from CP-SAT's optimum, configurable) is used as the test's "pass"
  criterion.

## Next step
Carry out the two tasks above (translating the doc + designing/
implementing `verify.py`, `tests/test_verify.py`, and
`tests/fixtures/hand_solved_instances.py`) before starting Milestone 2
(the CP-SAT solver); `test_brute_force_cross_check.py` and the rest of the
cross-consistency tests are completed alongside the solvers themselves
(Milestones 2/3), since they require a solver to exist.
