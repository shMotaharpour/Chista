# Replay Episode File — Schema `chistaagent.replay.v1`

A replay file records the turn-by-turn actions of **ONE agent** (not a full
two-player match). The `ReplayAgent` in `world/replay_agent.py` plays such a
file against any opponent, on either engine path (`world.fast_sim` or
`world.kaggle_env`), in either seat.

## File format

```json
{
  "schema": "chistaagent.replay.v1",
  "seed": 1064607942,
  "configuration": {"episodeSteps": 720},
  "agent_name": "my-agent-v3",
  "turns": [
    {"step": 0, "action": {"farmer": ["PASS"], "hands": [], "market": []}},
    {"step": 1, "action": {"farmer": ["BUILD_PASTURE"], "hands": [],
                            "market": [["BUY_ANIMAL", "COW", 2]]}},
    {"step": 2, "action": {"farmer": ["PICKUP", "COW", 1],
                            "hands": [["WEST"], ["NORTH"]],
                            "market": [["BUY_SEED", "MELON", 11]]}}
  ]
}
```

## Fields

| field | type | required | meaning |
|---|---|---|---|
| `schema` | string | yes | must be exactly `"chistaagent.replay.v1"` |
| `seed` | int \| null | no | the episode seed the actions were produced with. Informational only — the replay does not enforce it; whoever runs the episode decides the environment seed (use `record.seed` for a bit-exact replay of the recorded world, any other seed for a fresh world). |
| `configuration` | object | no | episode configuration the record was made under (e.g. `episodeSteps`). Free-form; hard validation reads `episodeSteps` from it when present. |
| `agent_name` | string \| null | no | label for the recorded agent (reports, file organization). |
| `turns` | array | yes | one entry per recorded turn, **sorted by strictly increasing `step`** (binary-searched at play time). |

Each `turns[i]`:

| field | type | meaning |
|---|---|---|
| `step` | int ≥ 0 | the observation step this action answers (`obs["step"]`) |
| `action` | object \| null | the action dict exactly as the engine accepts it; `null` = this turn is explicitly unrecorded |

## Action shape (engine-native — see `docs/player_agent.md`)

```json
{"farmer": ["PLANT", "WHEAT"],
 "hands":  [["WATER"], ["NORTH"]],
 "market": [["BUY_SEED", "WHEAT", 3], ["SELL", "CARROT", 5]]}
```

- `farmer` — one op for the main farmer (list)
- `hands` — one op per hired hand, in hands order (list of lists)
- `market` — ordered market orders, capped at `maxMarketOrdersPerTurn` (list of lists)
- Missing keys are treated by the engine as empty; `null` action = unrecorded turn.

## Seat independence (design contract)

The file contains **no seat identity**. `ReplayAgent` picks the action purely
by `obs["step"]` — it works unchanged at seat 0, seat 1, or both seats at once
(one instance per seat, sharing or not sharing the record). Instances hold no
cross-call state beyond hit/miss counters, so any number of replays can run
concurrently.

## Fallback rule

A step absent from the record (or an explicitly `null` action) plays **PASS**.
An episode can therefore never crash because of a short or gappy record —
beyond the last recorded turn the agent simply passes, exactly like the
engine's own no-op handling (F047).

## Validation

`EpisodeRecord` exposes validation as class methods, never run automatically
during play (R004 — on-demand analysis only):

- `EpisodeRecord.load(path)` — validates before constructing (pass
  `validate=False` for trusted files; the constructor still refuses a wrong
  schema or missing `turns`).
- `EpisodeRecord.validate(path, mode="soft" | "hard")` — returns a summary
  dict, raises `ReplayValidationError` on the first defect with a precise
  location (`turns[37].action.hands[2]`).
  - **soft** — structure: schema, required fields, action shape, step
    ordering/duplicates. Shape-valid but suspicious constructs (e.g. the
    legacy `["BUILD", "COOP"]` form) are reported in `summary["problems"]`.
  - **hard** — everything from soft, plus engine-aware depth: every op is a
    real op and every argument matches the shipped engine's vocabularies
    (imported from `kaggle_environments`, R002 — never transcribed), and the
    last step fits `configuration.episodeSteps` when present.

## Playing

```python
from world.replay_agent import EpisodeRecord, ReplayAgent
from world import kaggle_env
from world.fast_sim import FastSim

rec = EpisodeRecord.load("my_episode.json")
agent = ReplayAgent(rec)

# official path, seat 1, vs the starter agent
env = kaggle_env.run_episode(["starter", agent],
                             configuration={"seed": rec.seed,
                                            **rec.configuration})

# fast path, both seats replayed from two records of the same match
rec1 = EpisodeRecord.load("my_episode_p1.json")
sim = FastSim({"seed": rec.seed, **rec.configuration})
rewards = sim.run([agent, ReplayAgent(rec1)])
```

## Copy semantics

`ReplayAgent(record, copy=False)` (default) hands out the recorded action
object directly — zero per-turn overhead. This is safe: neither engine path
(`fast_sim` nor the harness) writes into a submitted action (pinned by
`test_engine_never_mutates_shared_actions`), but the *caller* must not
mutate what it receives either. With `copy=True` every handout is a fresh
deepcopy — defensive mode; measured ~10 ms per 720-turn season per seat
(~25% of a replay season), so keep it off in bulk runs.

Note: two `ReplayAgent` instances sharing ONE record receive the SAME
object per turn in share mode — fine for the engine, wrong only if a
caller edits what it receives.

