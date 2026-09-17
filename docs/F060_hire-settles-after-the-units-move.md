# F060 — HIRE settles after the units move, and the night resets the crew

**Summary (<=50 words):** A HIRE ordered at hour 0 spawns its hand on a
shed-access tile that the turn's own unit actions have already vacated — the
farmer walks off (4,4) and the hand appears there, not on (5,4). The night
clears hands, resets `hires_today`, empties every bag into the shed and returns
the farmer to (4,4).

## Measured

Two engine behaviours that decide where a plan may put a unit:

1. **`_process_market` runs after the turn's unit actions** (`kaggriculture.py`,
   the turn loop: units act, then the market). A `HIRE` in turn 0's market
   therefore sees the positions the units have *after* their turn-0 ops, and
   `_spawn_hand` (`:533`) picks the least-occupied shed-access tile of that
   state. Measured on `fast_sim` (seed 3): the farmer at (4,4) moves `WEST` at
   turn 0 and the hand is hired that turn — it spawns on **(4,4)**, the tile the
   farmer just left, not on (5,4).
2. **`_end_of_day` resets the crew** (`:877-882`): `farm["hands"] = []`,
   `farm["hires_today"] = 0`, every unit's bag is poured into the shed, and
   `farm["farmer"] = list(_default_spawn(board_size))` — the farmer returns to
   (4,4). So every unit starts every day within reach of the shed, hands are a
   daily purchase (F039), and a bag left un-dropped is not lost to the night but
   to the shed's capacity.

## Why it is a finding, not a detail

Predicting a new hand's spawn from the day-start board puts every hand one tile
out, and then **every op in its route lands on the wrong tile** — a `PLANT` on a
tile that is not empty, a `WATER` on nothing, a `HARVEST` on a tile that is not
under it. All three are refused **in silence** (F047), which is exactly the
failure class the day compiler exists to close: measured before the fix, the
per-op instrument in `tests/test_day_plan.py` reported six silent no-ops on a
hand's route (`PLANT` refused on a WEED tile at (0,1) — one tile west of where
the plan thought the hand stood). `secretary/routing.py::plan_day` now simulates
turn 0 and spawns from the post-move occupancy.

## Source

`kaggriculture.py`: the turn loop (`_process_market` after the unit actions),
`_spawn_hand:533`, `_end_of_day:877-882`; `_default_spawn`, `_shed_access_tiles`.
Reproduced by `tests/test_day_plan.py::test_the_hand_spawns_after_the_turns_unit_actions`
and `tests/test_day_plan.py::test_every_op_moves_what_it_promises`.
