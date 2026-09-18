"""Self-consistency check for tests/fixtures/hand_solved_instances.py: each
case's example_optimal_solution must actually be a valid solution (per the
independent verifier) whose cost matches the claimed optimum. This has to
hold *before* any solver exists — it's what makes these fixtures a
"""
import pytest

from agent.day.verify import verify_solution
from tests.day_layer.fixtures.hand_solved_instances import HAND_SOLVED_CASES


@pytest.mark.parametrize("case", HAND_SOLVED_CASES, ids=lambda case: case.name)
def test_example_solution_is_valid_and_matches_optimal_cost(case):
    assert case.example_optimal_solution is not None, f"{case.name} has no example_optimal_solution to check"
    result = verify_solution(case.instance, case.example_optimal_solution)
    assert result.is_valid, (case.name, result.violations)
    assert result.total_cost == case.optimal_cost, (case.name, result.total_cost, case.optimal_cost)
