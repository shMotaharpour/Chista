"""The world, as the engine defines it.

Four files, in the order a reader wants them:

- `model` — the names: products, crops, animals, structures, tile states, the 18
  columns a plan is priced over, the unit actions and the market orders.
- `rules` — the numbers: the season, the crop and animal tables, land, labour, the
  shed, the town, and the engine's turn order.
- `action_rules` — what each action needs and what it does, from the engine's handlers.
  (`actions.py` is something else: the action-shape validator the pool guard uses.)
- `prices` — the price function and the town's drain.

Nothing in this package imports our other code. The authority is
`kaggle_environments.envs.kaggriculture` (its `kaggriculture.py`, its
`kaggriculture.json` and the `README.md` it ships); each fact cites the lines it
came from, and the environment's own docs are quoted where they are clearer than the
code.
"""
