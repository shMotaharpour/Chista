"""The land image is a view of the arrays, and the depth layer says how long a chain runs.

Two structural facts, and neither is a timing: the image is a scatter of columns `build` already
filled, so it cannot disagree with them; and a tile's depth is how many of its ops wait on one another
in a row. Two `WATER`s on the same tile are NOT a chain of two - the second waits on nothing - so the
depth is the longest PRECEDENCE path and not the op count.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/chista/pm/world")

from agent.wsr import tasks as T

CORPUS = pathlib.Path(__file__).parent / "corpus" / "real_days.json"
REAL_DAYS = json.loads(CORPUS.read_text())


def _tasks(entry):
    grid = [(tuple(cell), tuple(ops), entity) for cell, ops, entity in entry["chains"]]
    return T.build(grid, available={good: int(hour) for good, hour in entry["available"].items()})


def test_the_image_says_what_the_arrays_say():
    """Every layer is a scatter of a column, so the totals have to agree with the columns."""
    entry = max(REAL_DAYS, key=lambda e: sum(len(ops) for _c, ops, _e in e["chains"]))
    tasks = _tasks(entry)
    image = T.land_image(tasks)

    assert image.shape == (T.BOARD_SIZE, T.BOARD_SIZE, len(T.IMAGE_LAYERS))
    assert image[:, :, T.IMAGE_LAYERS.index("tasks")].sum() == tasks.n

    # One tile's own vector is the same numbers read the other way round.
    for cell in tasks.cells:
        x, y = int(cell[0]), int(cell[1])
        assert image[x, y, 0] >= 1, f"tile {x},{y} carries a task and the image says none"

    # And a tile no task touches has nothing in any layer.
    worked = {tuple(int(v) for v in cell) for cell in tasks.cells}
    for x in range(T.BOARD_SIZE):
        for y in range(T.BOARD_SIZE):
            if (x, y) not in worked:
                assert image[x, y].sum() == 0, f"tile {x},{y} has no tasks and a non-zero image"


def test_a_chain_runs_as_deep_as_its_precedence():
    """A PLANT then a WATER is a chain of two, and a second WATER beside them is not a third."""
    tasks = T.build([((0, 0), ("PLANT", "WATER", "WATER"), "WHEAT")], available={"WHEAT": 0})
    depth = T.chain_depth(tasks)

    assert depth.size == tasks.n
    assert depth.min() >= 1, "a task sits in a chain of at least itself"
    assert depth.max() == 2, (
        f"the chain came back {depth.max()} deep: a PLANT before a WATER is two, and the second "
        f"WATER waits on nothing")
    assert T.land_image(tasks)[0, 0, T.IMAGE_LAYERS.index("depth")] == 2


def test_the_depth_never_exceeds_the_tasks_on_its_tile():
    """A chain runs on one tile, so its depth cannot be more than that tile's own task count."""
    entry = REAL_DAYS[0]
    tasks = _tasks(entry)
    depth = T.chain_depth(tasks)
    image = T.land_image(tasks)

    for row, cell in enumerate(tasks.cells):
        assert depth[row] <= image[int(cell[0]), int(cell[1]), 0], (
            f"tile {int(cell[0])},{int(cell[1])} runs {depth[row]} deep with fewer tasks on it")
