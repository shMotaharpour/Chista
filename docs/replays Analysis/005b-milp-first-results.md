# MILP CP-SAT — first real results & remaining gaps

## Status: model solves, plan looks plausible, but predicted $511k is OPTIMISTIC

The model now enforces (after 3 bug fixes):
- exact harvest schedule per crop (equality, zero on non-production days)
- exact animal production (zero on non-production days, zero before first_yield)
- cash path ≥ 0 every day
- tiles, shed cap, work capacity, terminal zero-residue

## Current plan shape (baseline, $3k start):

- day 0: 23 strawberry + 2 geese + 2 cows + 2 sheep (everything immediately!)
- day 1: 30 tomato + 30 strawberry + 10 melon
- crew stays ~3 all game (hiring is fiber-cost-gated by money*0.1)
- land NE@6, SW@11, SE@20
- final: $511,875 predicted

## Known gaps (predicted >> executed):

1. **Animal buy timing**: model buys animals day 0 (money allows), but executing
   agent needs PICKUP+BUILD+PLACE (3 actions) — the plan says "buy" not the
   full action sequence. Execution harness must convert.
2. **Watering capacity is a constant** (CREW_RATE × crew): real VRP may differ.
   100 tiles need 100 water actions/day; crew=3 × 24 turns = 72 → model is
   under-constrained vs reality until crew grows. This is why crew stays low:
   the model doesn't know watering needs EXACTLY.
3. **Movement not modeled**: distance between shed and NE quadrant ~5 steps
   each way — real tours waste many turns walking.
4. **Plant→water→harvest action accounting incomplete**: model counts plantings
   as actions but not daily watering of ALL live plants.

## Next steps (in order):

1. Build `lab/milp/execute.py` — convert plan.json to actual agent ops, run
   real env, measure gap (predicted vs executed).
2. Feed the measured gap back: tighten watering constraint with real numbers
   (tiles watered per crew-day measured from execution).
3. If gap > 30%: switch capacity modeling to VRP-derived capacity (space c).

## Decision: is this the right next step?

The model found $511k with correct constraints — far above the $87k ladder
median. Even at 50% execution efficiency that's $250k+ >> ladder. The plan
structure (tomato+strawberry+animals early, melon later, wheat never) matches
what top players do (001 §4: wheat backbone was a RE-interpretation — actually
winners plant strawberry heavily mid-season, matching this plan).

Proceed with executor implementation.
