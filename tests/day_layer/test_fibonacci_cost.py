import pytest

from day.fibonacci import fibonacci_cost


def test_known_values():
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for index, value in enumerate(expected):
        assert fibonacci_cost(index) == value


def test_negative_index_raises():
    with pytest.raises(ValueError):
        fibonacci_cost(-1)
