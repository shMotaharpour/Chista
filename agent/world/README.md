# world/

The definitions: what the game is, as the engine has it. Nothing here imports our code,
and nothing here reimplements the game — the engine stays the executor (R002/R003).

| file | what it holds |
|---|---|
| `model.py` | the names: products, crops, animals, structures, tile kinds, the 18 unit actions, the 6 market orders, the 18 columns |
| `rules.py` | the numbers: season, crop and animal tables, land, labour, shed, town, turn order |
| `action_rules.py` | what each action needs and does, and its arguments (`SIGNATURE`, `ITEM_OF`) |
| `action.py` | `WorkerAction` and `MarketAction`: op, item, count |
| `tile.py` | `TileHourZero` at a day start, `TileInDay` inside a day, `delta()` |
| `board.py` | cells, quadrants, `tiles[y][x]`, the shed's four doors |
| `worker.py` | `WorkerTrace`: a worker's start cell and its 24 hours |
| `prices.py` | the price function and the town's drain |

Authority: `kaggle_environments.envs.kaggriculture` — every fact cites its lines. Tools
(simulators, harness wrapper, replay agent, action-shape check) are in `offline_lab/`.
