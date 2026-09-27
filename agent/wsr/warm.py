"""The warm-route memory: lookup and storage for beam search seeds (#207 family).

Replaces `_SEARCH_CACHE` (#206), whose key omitted `drop_by` and the time
columns and returned one day's route as another day's answer. The design here
was agreed with the owner (2026-09-27):

- the key is a SIMILARITY signature, not an exact identity: what makes two
  days the same search problem is the LOGISTIC composition of the work —
  how many bag-consumers, bag-fillers, deliver-to-shed harvests, no-bag
  fillers and drops each quadrant carries — plus the pool and the hands-per-
  hour vector (`hire_times`, which fixes every hand's spawn door, F040).
  Task IDs are deliberately NOT in the key: a route stored on one layout is
  re-projected onto another by matching each placement's TASK CATEGORY.
- lookups return up to WARM_K candidate seeds, nearest first; every candidate
  is re-timed onto the current day by a greedy legal-hours pass and verified
  with `check_route` + `compile_route` BEFORE it reaches the beam, so a warm
  row can never poison the search (the `latest`-window hole the probes hit).
- exact-match semantics for "the same question" are NOT wanted here: two
  runs of one seed already share their pool through `planner.day`, and a
  cache that returns answers short-circuited the search (the #206 defect).
  This memory stores SEEDS only — the search always runs and always owns
  the answer.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from agent.wsr.emit import check_route, compile_route

#: Candidates returned per lookup. Measured trade: 1-2 underuses a good seed;
#: beyond ~8 the beam's own width absorbs the extra rows and time grows.
WARM_K = 8

#: Where the memory persists between seasons. A missing file is an empty memory:
#: the search runs cold and nothing else changes.
WARM_PATH = Path(__file__).resolve().parents[1] / "artifact" / "warm_routes.json"

#: The engine's own bag-cycle goods (`action_rules.CARRIES`): the two yields that
#: feed a consumer's bag inside the day. Every other yield is a shed delivery.
_CYCLE_GOODS = ("WHEAT", "FERTILIZER")

#: Categories by op name, from the engine's tables. `items`/`yields` on the
#: TaskArray are the authority; the names here only make the stored signature
#: readable.
CONSUMERS = ("FEED", "FERTILIZE", "PLACE")   # needs the good IN THE BAG
CATEGORIES = ("consumer", "producer", "deliver", "self_suff", "drops")


def _quadrant(cell, board_size: int) -> int:
    half = board_size // 2
    x, y = int(cell[0]), int(cell[1])
    return (y // half) * 2 + (x // half)


def _task_categories(tasks) -> list[str]:
    """Today's category per task row — the same five the signature counts."""
    from agent.world.action import item_of
    from agent.wsr.tasks import ITEM_CODE
    from agent.world.action_rules import CARRIES
    cycle = {ITEM_CODE[item_of(g)] for g in _CYCLE_GOODS}
    cats = []
    for i in range(tasks.n):
        op = str(tasks.ops[i][0])
        yg = int(tasks.yields[i])
        if op == "DROP":
            cats.append("drops")
        elif op in CONSUMERS:
            cats.append("consumer")
        elif yg >= 0 and yg in cycle:
            cats.append("producer")
        elif yg >= 0:
            cats.append("deliver")
        else:
            cats.append("self_suff")
    return cats


def signature(day, tasks) -> dict:
    """The (category x quadrant) density + the hands-per-hour vector.

    Categories come from the TaskArray's own columns, which the engine's
    tables fill: `items` (what must be in the bag) and `yields` (what the
    task puts in). A producer whose yield is a bag-cycle good (WHEAT /
    FERTILIZER) is a `producer`; any other yield is a `deliver` — it goes to
    the shed, it does not feed the day's bag. DROPs are one count: whether a
    drop banks anything is decided by the search, not by the layout.
    """
    from agent.world.action import item_of
    from agent.wsr.tasks import ITEM_CODE
    from agent.world.action_rules import CARRIES

    cycle = {ITEM_CODE[item_of(g)] for g in _CYCLE_GOODS}
    n_q = 4
    board = int(getattr(tasks, "board_size", 10) or 10)
    cats: list[Counter] = [Counter() for _ in range(n_q)]
    tiles = [0] * n_q
    for i in range(tasks.n):
        q = _quadrant(tasks.cells[i], board)
        op = str(tasks.ops[i][0])
        if op == "DROP":
            cats[q]["drops"] += 1
            continue
        tiles[q] += 1
        item = int(tasks.items[i])
        yg = int(tasks.yields[i])
        if op in CONSUMERS or (item >= 0 and str(tasks.ops[i][0]) == "PLACE"):
            cats[q]["consumer"] += 1
        elif yg >= 0 and yg in cycle:
            cats[q]["producer"] += 1
        elif yg >= 0:
            cats[q]["deliver"] += 1
        else:
            cats[q]["self_suff"] += 1
    return {
        "pool": int(len(day.hire_times)) if day.hire_times else 1,
        "hire_times": [int(h) for h in day.hire_times],
        "horizon": int(day.horizon),
        "quadrants": [{c: int(counter[c]) for c in CATEGORIES}
                      for counter in cats],
        "tiles": tiles,
    }


def sig_distance(a: dict, b: dict) -> float:
    """Distance in the signature space: category densities dominate, tiles and
    the hands-per-hour vector break near-ties. All terms are scale-free."""
    if a["pool"] != b["pool"]:
        return float("inf")          # a warm row for another pool is unusable
    n_q = len(a["quadrants"])
    d = 0.0
    for qa, qb in zip(a["quadrants"], b["quadrants"]):
        for c in CATEGORIES:
            d += abs(qa[c] - qb[c])
    for ta, tb in zip(a["tiles"], b["tiles"]):
        d += 0.5 * abs(ta - tb)
    if a["hire_times"] != b["hire_times"]:
        la, lb = len(a["hire_times"]), len(b["hire_times"])
        d += 2.0 * abs(la - lb) + sum(
            abs(x - y) for x, y in zip(a["hire_times"], b["hire_times"]))
    if a["horizon"] != b["horizon"]:
        d += 4.0
    return d


class WarmMemory:
    """The seed store. In-memory during a season; JSON on disk between runs."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else WARM_PATH
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        self._entries = []
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._entries = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries))

    # -- store -------------------------------------------------------------
    def store(self, day, tasks, result, *, verified: bool = True) -> None:
        """Add a solved route as a future seed. `verified=False` stores it
        anyway (offline build), flagged so lookups prefer verified entries."""
        sig = signature(day, tasks)
        cats = _task_categories(tasks)
        id_to_idx = {str(tid): i for i, tid in enumerate(tasks.ids)}
        self._entries.append({
            "sig": sig,
            "pool": int(result.pool),
            "route": [(int(t), str(tid), int(w)) for t, tid, w in result.route],
            "categories": [cats[id_to_idx[str(tid)]] if str(tid) in id_to_idx
                           else "self_suff" for _t, tid, _w in result.route],
            "complete": bool(result.complete),
            "verified": bool(verified),
        })
        if len(self._entries) > 4096:
            self._entries = self._entries[-4096:]

    # -- lookup ------------------------------------------------------------
    def lookup(self, day, tasks) -> list:
        """Up to WARM_K verified, re-timed, compile-checked warm Results for
        this day — nearest signature first. [] when nothing fits."""
        sig = signature(day, tasks)
        pool = sig["pool"]
        scored = []
        for entry in self._entries:
            if entry.get("verified", False):
                d = sig_distance(sig, entry["sig"])
                if d == d and d != float("inf"):
                    scored.append((d, entry))
        if not scored:
            self.misses += 1
            return []
        scored.sort(key=lambda x: x[0])
        out = []
        for _d, entry in scored[:WARM_K]:
            warm = self._project(entry, day, tasks)
            if warm is not None:
                out.append(warm)
        if out:
            self.hits += 1
        else:
            self.misses += 1
        return out

    # -- projection --------------------------------------------------------
    def _project(self, entry: dict, day, tasks):
        """Re-project a stored route onto today's tasks: match each placement
        by CATEGORY (nearest free task of the same category to the worker's
        recorded position), then re-time greedily into legal hours, then
        verify with the compiler. Returns a Result, or None."""
        n = tasks.n
        board = int(getattr(tasks, "board_size", 10) or 10)
        from agent.world.action import item_of
        from agent.wsr.tasks import ITEM_CODE
        from agent.world.action_rules import CARRIES
        cycle = {ITEM_CODE[item_of(g)] for g in _CYCLE_GOODS}
        cats = []
        for i in range(n):
            op = str(tasks.ops[i][0])
            yg = int(tasks.yields[i])
            if op == "DROP":
                cats.append("drops")
            elif op in CONSUMERS:
                cats.append("consumer")
            elif yg >= 0 and yg in cycle:
                cats.append("producer")
            elif yg >= 0:
                cats.append("deliver")
            else:
                cats.append("self_suff")
        by_cat: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            by_cat[cats[i]].append(i)

        pool = entry["pool"]
        claimed: set[int] = set()
        route = []
        cat_seq = entry.get("categories")
        if not cat_seq or len(cat_seq) != len(entry["route"]):
            return None
        worker_at: dict[int, tuple[int, int]] = {}
        hours = day.hire_times
        for (turn, _tid, worker), cat in zip(entry["route"], cat_seq):
            cands = by_cat.get(cat, [])
            best, best_d = None, None
            for i in cands:
                if i in claimed:
                    continue
                cell = (int(tasks.cells[i][0]), int(tasks.cells[i][1]))
                at = worker_at.get(worker, (board // 2, board // 2))
                d = abs(cell[0] - at[0]) + abs(cell[1] - at[1])
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best is None:
                continue
            claimed.add(best)
            worker_at.setdefault(worker, (board // 2, board // 2))
            route.append((turn, best, worker))
        if not route:
            return None
        re_timed = self._retime(route, tasks, day, pool)
        if re_timed is None:
            return None
        warm = re_timed
        if check_route(day, tasks, warm):
            return None
        try:
            compile_route(day, tasks, warm)
        except ValueError:
            return None
        return warm

    def _retime(self, route, tasks, day, pool):
        """Greedy legal-hours pass over a projected route: keep the placement
        ORDER per worker, move each op to the earliest hour that satisfies the
        task's window, the worker's own first hour and the walk from the
        worker's previous cell."""
        from agent.wsr.routing import walk
        board = int(getattr(tasks, "board_size", 10) or 10)
        at: dict[int, tuple[int, int]] = {}
        free: dict[int, int] = {}
        done_at: dict[int, int] = {}
        out = []
        for _turn, idx, worker in sorted(route, key=lambda x: (x[0], x[1])):
            cell = (int(tasks.cells[idx][0]), int(tasks.cells[idx][1]))
            t = free.get(worker, int(day.hire_times[worker]) if worker < len(day.hire_times) else 1)
            t = max(t, int(tasks.earliest[idx]))
            t += abs(at.get(worker, (board // 2, board // 2))[0] - cell[0]) \
                 + abs(at.get(worker, (board // 2, board // 2))[1] - cell[1])
            t = min(t, int(tasks.latest[idx]))
            done_at[idx] = t
            free[worker] = t + 1
            at[worker] = cell
            out.append((t, str(tasks.ids[idx]), int(worker)))
        out.sort()
        return B_Result(pool=pool, route=out, complete=False)


def B_Result(**kw):
    from agent.wsr.beam import Result
    return Result(**kw)


_MEMORY: WarmMemory | None = None


def memory() -> WarmMemory:
    """The season's shared memory (lookup counters included)."""
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = WarmMemory()
    return _MEMORY


def candidates_for(day, tasks) -> list:
    """The entry point `search` calls when the caller passed no warm: up to
    WARM_K verified seeds, nearest signature first."""
    return memory().lookup(day, tasks)


def remember(day, tasks, result, *, verified: bool = True) -> None:
    """The entry point `search` calls with every answer it returns."""
    memory().store(day, tasks, result, verified=verified)
