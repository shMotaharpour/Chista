# F053 — A refused op still bills the cost vector

**Summary (<=50 words):** A chain's cost comes from the ops it was asked
to run, not from what the engine accepted. On a bare tile the engine refuses
WATER, HARVEST, DIG and FERTILIZE in silence (F047) and the model still bills
them. Shipped chains are safe; hand-built ones are not.

## Measured

`bench/oracle_transitions.py` on CARROT (13 states, 86 edges, every state
exercised, sequences up to 3 ops):

```
COVERED: 456   ALIAS: 18   NOOP: 57   UNCOVERED: 0
REFUSED: 0     PERM_BETTER: 0   CANON_WORSE_ORDER: 2   REFUSED_BUT_BILLED: 56
```

Fifty-six of the enumerated sequences carry a model cost while the engine
accepted nothing and spent nothing. Four of them, from the `NONE` state:

```
NONE -(FERTILIZE)-> the model bills LABOR_HOURS=1, FERTILIZER=1 | the engine did nothing
NONE -(WATER)->     the model bills LABOR_HOURS=1               | the engine did nothing
NONE -(HARVEST)->   the model bills LABOR_HOURS=1               | the engine did nothing
NONE -(DIG)->       the model bills LABOR_HOURS=1               | the engine did nothing
```

## Why the shipped graph is fine and this is still a finding

`chain_requirements` derives cost from the chain's op list. The builder only
ever registers chains whose ops the engine accepts from that state, so no
**shipped** edge carries a phantom cost — `UNCOVERED: 0` and `REFUSED: 0` say
the graph and the engine agree on every transition that exists.

The exposure is in chains built by hand, outside the registry. Anything that
assembles ops itself and prices them through the same path gets billed for
refusals it never notices, because the engine's refusal is silent (F047) and
the cost vector's arithmetic is not.

## Where it bites next

Issue #14's adapter. Three of Chista's ops — `CARE`, `DIG` and
`BUILD_COOP`/`BUILD_PASTURE` — have no major-task recipe in
`secretary/models.py`, so the adapter emits them as standalone minor tasks with
explicit precedence. That is a hand-built chain by construction, and it reaches
the cost path the same way.

The check is cheap and belongs on the chain, not on the plan: before pricing a
hand-built chain, confirm the engine accepts each op from the state it will run
in, and drop the ones it refuses **before** they reach `chain_requirements` —
not after, where they have already been paid for.

## The reverse case is a separate number

`NOOP: 57` counts sequences the engine refused *and* the model billed nothing
for. Those are correct. `REFUSED_BUT_BILLED` is only the overlap where the
model charged for a refusal, which is why the two are reported apart.
