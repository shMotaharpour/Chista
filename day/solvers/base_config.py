"""Base configuration shared by every solver.

`validate` (default True): after a solver produces a schedule, it runs the
independent verifier (`day.verify.verify_solution`) on its own answer
and reports INVALID_SOLUTION instead of returning an unverified schedule.
Set it to False only for experiments that re-implement validation
elsewhere — a solver answer should always be verified somewhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseSolverConfig:
    validate: bool = True
