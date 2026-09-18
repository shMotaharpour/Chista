"""Worker activation cost: Fib(0)=0, Fib(1)=1, Fib(2)=1, Fib(3)=2, Fib(4)=3, ..."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def fibonacci_cost(worker_index: int) -> int:
    if worker_index < 0:
        raise ValueError(f"worker_index must be >= 0, got {worker_index}")
    a, b = 0, 1
    for _ in range(worker_index):
        a, b = b, a + b
    return a
