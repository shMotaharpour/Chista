"""Where a worker stands, outside the beam's own step: one reader, and it agrees with the day written.

The search decides who does what and when; everything else about a worker's POSITION is derived:
where the hands land (`_hand_doors`), where a unit is at a given turn (`_stand_after`), how much of
its day is left (`remaining_turns`), and what a warm start hands the beam (`_warm_row`). Each of
those used to read the route its own way, and they disagreed with the compiled ops on one shape:
a drop with an empty bag (turn -1). It has no turn, no op and no walk, but read as a leg it put the
worker on its door from turn 0 - so the next hand's door was wrong, and the engine placed that hand
elsewhere and banked less than the day promised.

What is asserted:

    `_stand_after` equals the position the compiled ops walk to, every worker, every turn
    `remaining_turns` equals the PASS turns the compiler writes
    the doors the day is written from are the doors the engine spawns the hands on
    a warmed row stands where the route ends and is charged the walk the ops take
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from agent.dispatch import dispatch_plan
from agent.wsr import beam as B
from agent.wsr import tasks as T
from agent.wsr.emit import compile_route, to_plan

MOVE = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
#: Four geese across the west half, one fertilizer each to collect, banked by hour 20: far enough
#: from the shed that one drop banks two collections and the second drop is idle. The shape that
#: exposed the disagreement (`/tmp` probe: 13 of 13 idle-drop days wrong before the fix).
CELLS = [(2, 6), (1, 6), (2, 1), (1, 1)]
GOOSE = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0, "yield_units": 0,
         "consecutive_unfed": 0, "fed_today": False, "cared_today": False,
         "fertilizer_available": True, "pending_care_bonus": 0}
DEADLINE = 20
POOLS = (0, 1, 2)


def _search(hands: int):
    chains = [(cell, ("COLLECT_FERTILIZER",), None) for cell in CELLS]
    tasks = T.build(chains, available={}, drop_by=[DEADLINE] * len(chains))
    day = B.Day(chains=tuple(chains), available={}, hire_times=(1,) * hands)
    return day, tasks, B.search(day, tasks, hands=hands, max_hands=hands, budget_s=10.0)


@pytest.fixture(scope="module", params=POOLS, ids=[f"hands={h}" for h in POOLS])
def searched(request):
    hands = request.param
    day, tasks, result = _search(hands)
    return hands, day, tasks, result, compile_route(day, tasks, result)


def _walked(ops, starts):
    """Where each worker is after each turn, read off the compiled ops the way the engine moves it."""
    out = []
    for worker, unit in enumerate(ops.units):
        pos, seq = (int(starts[worker][0]), int(starts[worker][1])), []
        for op in unit:
            if op[0] in MOVE:
                pos = (pos[0] + MOVE[op[0]][0], pos[1] + MOVE[op[0]][1])
            seq.append(pos)
        out.append(seq)
    return out


def test_the_day_has_an_idle_drop() -> None:
    """The premise: at least one pool's route carries a drop with an empty bag."""
    idle = [tid for hands in POOLS for turn, tid, _w in _search(hands)[2].route if turn < 0]
    assert idle, "no route has an idle drop, so this file tests nothing"


def test_stand_after_reads_the_day_the_compiler_writes(searched) -> None:
    hands, day, tasks, result, ops = searched
    starts = B._start_positions(day, result.pool, result.doors)
    walked = _walked(ops, starts)
    per: dict[int, list] = {}
    for turn, task_id, worker in result.route:
        per.setdefault(int(worker), []).append((int(turn), task_id))
    for worker, entries in per.items():
        entries.sort()
        wrong = [(turn, B._stand_after(tasks, entries, starts[worker], turn), walked[worker][turn])
                 for turn in range(day.horizon)
                 if B._stand_after(tasks, entries, starts[worker], turn) != walked[worker][turn]]
        assert not wrong, (f"hands={hands} worker {worker}: the model puts the worker elsewhere "
                           f"than the ops walk it (turn, model, ops): {wrong[:3]}")


def test_the_room_left_is_the_room_written(searched) -> None:
    hands, day, tasks, result, ops = searched
    hours = B._start_hours(day, result.pool)
    written = [sum(1 for op in unit[int(hours[w]):] if op[0] == "PASS")
               for w, unit in enumerate(ops.units)]
    assert B.remaining_turns(day, tasks, result) == written, (
        f"hands={hands}: remaining_turns {B.remaining_turns(day, tasks, result)} against the PASS "
        f"turns the compiler writes {written}")


def test_the_hands_start_where_the_engine_spawns_them(searched) -> None:
    from offline_lab.fast_sim import FastSim

    hands, _day, _tasks, result, ops = searched
    if not hands:
        pytest.skip("the farmer alone has no door to choose")
    sim = FastSim({"episodeSteps": 25})
    live = sim.observations(copy_state=False)[0]
    for x, y in CELLS:
        live["farms"][0]["tiles"][y][x] = dict(GOOSE)
    plan = to_plan(ops)
    obs = sim.observations()[0]
    action = dict(dispatch_plan(plan, obs))
    action["market"] = [["HIRE"]] * hands
    sim.step([action, PASS])
    spawned = tuple(tuple(p) for p in sim.observations()[0]["farms"][0]["hands"])
    assert spawned == tuple(result.doors), (
        f"hands={hands}: the day is written from {result.doors} and the engine spawned {spawned}")


def test_a_warmed_row_stands_where_its_route_ends(searched) -> None:
    hands, day, tasks, result, ops = searched
    n, m = tasks.n, len(day.units) + result.pool
    done = np.zeros((1, n), bool)
    when = np.zeros((1, n), np.int16)
    who = np.full((1, n), -1, np.int16)
    free = B._start_hours(day, result.pool)[None, :].astype(np.int16).copy()
    starts = B._start_positions(day, result.pool, result.doors)
    where = starts[None].astype(np.int16).copy()
    travel = np.zeros(1, np.int16)
    count = np.zeros((1, n), np.int16)
    B._warm_row(tasks, result, done, when, who, free, where, travel, count)

    walked = _walked(ops, starts)
    moves = sum(1 for unit in ops.units for op in unit if op[0] in MOVE)
    ends = [walked[w][-1] for w in range(m)]
    assert [tuple(int(v) for v in where[0, w]) for w in range(m)] == ends, (
        f"hands={hands}: the warmed row stands at {where[0].tolist()} and the route ends at {ends}")
    assert int(travel[0]) == moves, (
        f"hands={hands}: the warmed row is charged {int(travel[0])} steps and the ops walk {moves}")


def test_a_farmer_who_walks_in_turn_0_is_not_a_reason_to_refuse_a_route() -> None:
    """The consistency check asks whether the HANDS start where the day says; the farmer's own start
    is not a choice. It used to compare the farmer's start cell with where it stands after turn 0,
    so every candidate on a day whose farmer walks in turn 0 was refused - measured on twelve
    waterings, farmer alone: 0 of 2 tightened candidates accepted, against 2 of 2 now."""
    import json

    board = json.loads((pathlib.Path(__file__).parent / "corpus" / "mixed_second_day.json")
                       .read_text())
    waters = [(tuple(c), tuple(o), e) for c, o, e in board["chains"] if o == ["WATER"]][:12]
    tasks = T.build(waters, available={})
    day = B.Day(chains=tuple(waters), available={}, hire_times=())
    result = B.search(day, tasks, hands=0, max_hands=0, budget_s=20.0)
    ops = compile_route(day, tasks, result)
    assert ops.units[0][0][0] in MOVE, "the premise: the farmer walks off its door in turn 0"
    assert B._consistent(day, tasks, result), (
        "a route the compiler writes and the rules accept is refused because the farmer walks")
