"""The day: routing, carries, drops, the market queue, and the WSR schedulers.

`docs/ARCHITECTURE.md` §4. The day layer turns the planner's per-tile chains into a
unit-day the engine accepts (`routing.py`) and schedules the workers (`models.py`,
`solvers/`). It does not price anything and never touches the market itself.
"""
