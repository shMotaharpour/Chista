"""The day: routing, carries, drops, and the worker schedulers.

`docs/ARCHITECTURE.md` §4. The day layer turns the planner's per-tile chains into a unit-day
the engine accepts (`routing.py`) and schedules the workers (`models.py`, `oxa_solver.py`). It
does not price anything and never touches the market itself.
"""
