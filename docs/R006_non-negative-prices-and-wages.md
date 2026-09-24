# R006 — Non-negative prices and wages

**Summary (<=50 words):** The tile graph is dominance-pruned, and that pruning
is optimality-preserving only while every price and every wage is
non-negative. A negative component makes the pruned graph silently wrong, so
the contractor asserts `p >= 0` and `w >= 0` on entry, every call.

## Decision

`agent/tile_dp/graph.py::_prune` drops `e2` when some `e1` costs no more of **every**
resource and produces no less of **every** resource, componentwise (#7). The
contractor then prices the surviving edges:

```
value(e) = produce(e)·p − cost(e)·w
```

and the pruning is only sound if `value(e1) − value(e2) >= 0` follows from
`_dominates(e1, e2)`. That step needs exactly one thing:

> **`p >= 0` and `w >= 0` componentwise.**

If any component of `p` were negative, an edge that produces *less* of that
resource would be the better one — and it is not in the graph any more. If any
component of `w` were negative, an edge that *consumes more* would be the
better one, and it is not in the graph any more either. In both cases the
sweep returns a confident, well-formed, wrong plan: the graph looks healthy,
the arithmetic looks healthy, and the failure has no symptom.

So the contract has two halves:

1. **The contractor asserts it.** `price_board()` raises on entry when any
   component of `p` or `w` is negative, in dev mode and in fast mode (the check
   is one `.min() < 0` per vector, which is far below the sweep it guards). The
   assertion names this rule.
2. **The master produces duals that satisfy it.** The Dantzig–Wolfe master
   (#12) hands the contractor non-negative prices and wages, and projects the
   duals back onto the non-negative orthant after every tâtonnement step. Its
   rows are inequalities only, and any dual HiGHS reports below zero is
   clamped.

## Why it is a rule and not a finding

Findings describe the engine. This constrains every future consumer of the
graph: if the graph is pruned more aggressively, if a new resource is priced,
if a second master is written, the same precondition comes with it. It is
recorded here so it cannot be rediscovered as a symptom.

## Relation to R005

R005 says **a number must have a source**. R006 says **a sign must have one
too**. They are close relatives and they fail the same way: a dual that comes
back negative from HiGHS at a degenerate optimum is not a fact about the world,
it is an artifact of the LP — and unlike an invented constant it does not
merely mislead a reader, it silently invalidates the pruning the whole artifact
rests on. Clamping is not a convenience here; it is part of the contract that
makes the graph the right object to price.

## How it is enforced

- `agent/tile_dp/contractor.py::_check_non_negative` — the assertion, on every call.
- `tests/test_tile_dp_contractor.py::test_negative_price_raises_r006` — the
  guard tested directly, not only the happy path.
- The master (#12) owns the other half: every duel export is projected onto
  the non-negative orthant before it reaches the contractor.
