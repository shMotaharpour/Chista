# offline_lab/

Tools that never run inside a turn: evaluation, simulation, the pool.

| need | use |
|---|---|
| the official harness (schema-validated, slow) | `kaggle_env.run_episode` |
| bulk simulation (DP/MDP/RL) | `fast_sim.FastSim` |
| a recorded tape as an agent | `replay_agent` (`REPLAY_SCHEMA.md` has the format) |
| an agent's action shape | `actions.validate_action` |
| a paired evaluation of two agents | `evaluate.py` |
| one episode, one process, per seat | `runner.py` |
| what a day's search costs, day by day | `search_cost.py` |
| the opponent pool | `pool/` |

`offline_lab/` may import `agent/world/` (the definitions). `agent/world/` never imports
`offline_lab/` — and nothing under `agent/` does: the submission is self-contained, and the
builders that write its artifacts live here.
