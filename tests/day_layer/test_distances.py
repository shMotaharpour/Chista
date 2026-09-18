from agent.day.distances import distance_to_nearest_entry, manhattan


def test_manhattan_basic():
    assert manhattan((0, 0), (0, 0)) == 0
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (4, 5)) == 5


def test_distance_to_nearest_entry_from_inside_the_block():
    # (4,4) is itself a shed cell.
    assert distance_to_nearest_entry((4, 4)) == 0
    # (0,0) -> nearest of {(4,4),(5,4),(4,5),(5,5)} is (4,4), distance 8.
    assert distance_to_nearest_entry((0, 0)) == 8
    # (9,9) -> nearest is (5,5), distance 8.
    assert distance_to_nearest_entry((9, 9)) == 8
