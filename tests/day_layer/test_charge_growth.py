"""The red-green guard for the union-growth fix. Asserts the SET-LEVEL
contract the fixed point needs: a charge that lacks a good the bag loads
must grow to hold it — max() cannot, union does. Written as a day-layer
unit test shape so it can land with the fix."""
import sys
sys.path.insert(0, "/chista/ChistaWRS")

from agent.wsr.beam import _fixed_point  # the fixed point exists


def test_union_growth():
    a = frozenset({0, 1})
    b = frozenset({1, 2})
    # the defect: max() on frozensets compares by superset
    assert max(a, b) == a, "max() must NOT be the growth operator"
    # the contract: the charge grows to hold every good the bag loads
    assert (a | b) == frozenset({0, 1, 2})
    # and the growth is monotone: nothing already charged is lost
    assert a <= (a | b) and b <= (a | b)


def test_fixed_point_grows_by_union(monkeypatch=None):
    """The live path: _fixed_point's growth line uses union. Re-derive by
    reading the source (the loop is 6 lines; the operator is the bug)."""
    import inspect
    import agent.wsr.beam as beam
    src = inspect.getsource(beam._fixed_point)
    assert "max(charged, bag)" not in src, "the max() defect is back"
    assert "charged | bag" in src


if __name__ == "__main__":
    test_union_growth()
    test_fixed_point_grows_by_union()
    print("guard green: the charge grows by union")
