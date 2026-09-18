# world/

The reference of definitions: what the game is, as the engine has it.

| file | what it holds |
|---|---|
| `model.py` | the names: products, crops, animals, structures, tile states, the 18 columns, the 18 unit actions, the 6 market orders |
| `rules.py` | the numbers: the season, the crop and animal tables, land, labour, the shed, the town, the turn order |
| `action_rules.py` | what each action needs and what it does |
| `tile.py` | a tile: the engine's fields decoded, for any layer to read |
| `prices.py` | the price function and the town's drain |

Authority: `kaggle_environments.envs.kaggriculture` — its `kaggriculture.py`, its
`kaggriculture.json`, and the `README.md` it ships. Every fact cites the line it came
from. Nothing here imports our code, and nothing here reimplements the game: the
engine stays the executor (R002/R003).

The tools that used to live here — the two simulators, the harness wrapper, the
replay agent and the action-shape check — are in `offline/`. They are not
definitions.
