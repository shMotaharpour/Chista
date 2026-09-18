from agent.wsr.distances import assign_entry_cells


def test_worked_example_from_spec():
    """The exact scenario from docs/problem-formulation.md: worker 0 stays
    at NW from time 0; workers 1-4 all enter at time 1 and must land
    NE, SW, SE, NW respectively (worker 4 ties back onto NW with worker 0)."""
    worker_starts = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)]
    assignment = assign_entry_cells(worker_starts)
    assert assignment == {0: "NW", 1: "NE", 2: "SW", 3: "SE", 4: "NW"}


def test_single_worker_gets_nw():
    assert assign_entry_cells([(0, 0)]) == {0: "NW"}


def test_earlier_start_time_processed_first_regardless_of_index():
    # worker 5 starts earlier than worker 2, so it claims NW first even
    # though its index is larger.
    worker_starts = [(2, 3), (5, 0)]
    assignment = assign_entry_cells(worker_starts)
    assert assignment[5] == "NW"
    assert assignment[2] == "NE"


def test_ties_within_same_start_time_broken_by_ascending_index():
    worker_starts = [(3, 0), (1, 0), (2, 0)]
    assignment = assign_entry_cells(worker_starts)
    assert assignment[1] == "NW"
    assert assignment[2] == "NE"
    assert assignment[3] == "SW"
