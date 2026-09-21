"""Rebuild the mixed day's SECOND day: the board the four-hand day left, and the work that board needs.

Not a test - a tool, run by hand:

    .venv/bin/python tests/day_layer/corpus/build_mixed_second_day.py

The day after a day-0 plan is a shape the day-0 tests cannot cover: the wheat is in the ground and the
animals are in their structures, so the day's work is watering and feeding rather than building and
planting. The builder runs the four-hand mixed day (`test_mixed_day.py`'s own day) through the engine,
reads the board at day 1 hour 0, and writes the work the engine's own rules say that board needs:

  WATER   a plant is watered once per day (kaggriculture.py:434-436), and two consecutive missed days
          turn it to a weed - the planting day counts as the first miss (:222, :783-784).
  FEED    an animal is fed once per day (:505-513). A newly placed animal survives its first day
          unfed (:236), so by day 1 every animal on the board is one missed day from escaping (:813).
  CARE    once per day (:524-530), banked against the next scheduled production.

The chains are the day's work and nothing else - the same split the day-0 scenario makes: the cow,
whose chain stops at PLACE, gets FEED and not CARE. What the fixture carries is the board and its
work, never a plan: the test searches it.
"""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.dispatch import dispatch_plan
from agent.tile_dp.chains import chain_id_of, chain_ops
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import compile_route, to_plan
from offline_lab.kaggle_env import new_environment

OUT = pathlib.Path(__file__).parent / "mixed_second_day.json"
HANDS = 4

spec = importlib.util.spec_from_file_location(
    "mixed", ROOT / "tests" / "day_layer" / "test_mixed_day.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def day_one_board() -> dict:
    """The mixed day played on day 0, and the board the engine holds at day 1 hour 0."""
    chains = [(cell, chain_ops(chain_id_of(ops)), entity) for cell, ops, entity in M.TILES]
    tasks = T.build(chains, available=M.AVAILABLE)
    day = B.Day(chains=tuple(chains), available=M.AVAILABLE, hire_times=(1,) * HANDS)
    result = B.search(day, tasks, beam=64, hands=HANDS, max_hands=HANDS)
    assert result.complete, f"the four-hand day did not carry day 0: {len(result.route)} tasks"
    plan = to_plan(compile_route(day, tasks, result))
    orders = [order for order in M.ORDERS if order[0] != "HIRE"] + [["HIRE"]] * HANDS

    def agent(obs):
        if int(obs["day"]) != 0:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        action = dispatch_plan(plan, obs)
        action["market"] = orders if int(obs["hour"]) == 0 else []
        return action

    env = new_environment({"episodeSteps": 25, "seed": 0})
    env.run([agent, "random"])
    last = env.steps[-1][0]["observation"]
    assert int(last["day"]) == 1 and int(last["hour"]) == 0, (
        f"the episode ended at day {last['day']} hour {last['hour']}, not day 1 hour 0")
    return last


def work_of(board: list) -> list:
    """The day's work, from the board: WATER every planted tile, FEED and CARE the animals."""
    chains = []
    for y in range(len(board)):
        for x in range(len(board[y])):
            tile = board[y][x]
            if not isinstance(tile, dict) or not tile:
                continue
            if tile.get("crop") == "WHEAT":
                chains.append([[x, y], ["WATER"], "WHEAT"])
            elif tile.get("animal"):
                ops = ["FEED"] + (["CARE"] if tile["animal"] != "COW" else [])
                chains.append([[x, y], ops, tile["animal"]])
    return sorted(chains, key=lambda chain: (chain[0][1], chain[0][0]))


def main() -> None:
    observation = day_one_board()
    farm, private = observation["farms"][0], observation["private"]
    board = farm["tiles"]
    chains = work_of(board)

    fixture = {
        "scenario": "the mixed day's second day, four hands",
        "source": "day 0 of the mixed day (test_mixed_day.py) played on the engine, day 1 hour 0",
        "hands": HANDS,
        "hire_times": [1] * HANDS,
        # Every good the day's work consumes, and the hour it is in the shed: the wheat for the
        # feeding is there from the previous day, so hour 0.
        "available": {"WHEAT": 0},
        "state": {
            "money": int(farm["money"]),
            "shed": {good: int(n) for good, n in private["shed"].items()},
            "seeds": {crop: int(n) for crop, n in private["seeds"].items()},
            "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        },
        "board": [[[x, y], board[y][x]] for y in range(len(board)) for x in range(len(board[y]))
                  if isinstance(board[y][x], dict) and board[y][x]],
        "chains": chains,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    planted = sum(1 for _cell, ops, _e in chains if ops == ["WATER"])
    animals = len(chains) - planted
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  {len(chains)} chains: {planted} waterings, {animals} animals; "
          f"money {fixture['state']['money']}, shed wheat {fixture['state']['shed']['WHEAT']}")


if __name__ == "__main__":
    main()
