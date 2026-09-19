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
| the opponent pool | `pool/` |

`offline_lab/` may import `world/` (the definitions). `world/` never imports `offline_lab/`.
