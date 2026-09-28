"""Self-warm: the artifact-backed seed memory for the beam search.

The artifact (`agent/artifact/warm_routes.npz`) is built OFFLINE by
`offline_lab/build/warm_routes.py` from the winner's own recorded routes in
the September parquet store. Each seed carries:

  pool         the worker count the route was searched with
  hire_hours   the seed's own per-worker first acting hour
  soft         the (category x quadrant) logistic density + tiles
  route        the placements: (hour, task vocab index, worker)
  route_cat    the category of each placement
  doors        the doors the route settled on

The lookup pipeline (hard gates first, then soft similarity):

  1. pool gate: only seeds whose pool equals the query pool (the beam's
     `seed()` accepts `warm.pool == pool` and nothing else);
  2. hire-hour gate: the query's per-worker first hours must equal the
     seed's (the door geometry, F040, is a function of this vector);
  3. soft ranking: L1 distance over the (category x quadrant) density and
     tile counts; the WARM_K nearest seeds are returned;
  4. re-projection: each stored placement claims the nearest FREE task of
     the SAME CATEGORY in today's day, re-timed into today's legal hours;
  5. legality: the re-timed seed must pass `check_route` + `compile_route`
     before it is offered — the memory can only offer seeds that compile.

Self-warm is always consulted (`search` calls `selfwarm.candidates_for`).
The caller's own `warm=` argument is a different, privileged channel and
takes priority; the two are separate by design.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent.wsr.emit import check_route, compile_route

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifact" / "warm_routes.npz"
WARM_K = 8

CATEGORIES = ("consumer", "producer", "deliver", "self_suff", "drops")
QUADRANTS = 4
SOFT_WIDTH = QUADRANTS * (len(CATEGORIES) + 1)
CONSUMERS = ("FEED", "FERTILIZE", "PLACE")


def task_categories(tasks) -> np.ndarray:
    """Today's category per task row, from TaskArray's own columns.

    consumer  = the op needs a good IN THE BAG (`items >= 0`): FEED, FERTILIZE,
                PLACE (the engine's CARRIES table plus the animal placement).
    producer  = the op puts a bag-cycle good in the bag (`yields` is WHEAT or
                FERTILIZER): the wheat HARVEST and COLLECT_FERTILIZER.
    deliver   = the op puts a NON-cycle good in the bag: the melon/strawberry
                HARVESTs that go straight to the shed.
    self_suff = neither (WATER, CARE, DIG, PLANT - the seed comes from the
                seed shed, not the worker's bag).
    drops     = the derived DROP rows (one per deadline-carrying chain).
    """
    from agent.world.action import item_of
    from agent.wsr.tasks import ITEM_CODE
    cycle = {ITEM_CODE[item_of(g)] for g in ("WHEAT", "FERTILIZER")}
    cats = np.empty(tasks.n, dtype=np.int8)
    for i in range(tasks.n):
        op = str(tasks.ops[i][0])
        item = int(tasks.items[i])
        yg = int(tasks.yields[i])
        if op == "DROP":
            cats[i] = 4
        elif item >= 0 or op == "PLACE":
            cats[i] = 0
        elif yg >= 0 and yg in cycle:
            cats[i] = 1
        elif yg >= 0:
            cats[i] = 2
        else:
            cats[i] = 3
    return cats


def soft_vector(day, tasks) -> np.ndarray:
    """The (category x quadrant) density + tiles, as one int32 vector (20)."""
    board = int(getattr(tasks, "board_size", 10) or 10)
    half = board // 2
    cats = task_categories(tasks)
    vec = np.zeros(SOFT_WIDTH, dtype=np.int32)
    for i in range(tasks.n):
        x, y = int(tasks.cells[i][0]), int(tasks.cells[i][1])
        q = (y // half) * 2 + (x // half)
        vec[q * (len(CATEGORIES) + 1) + int(cats[i])] += 1
        if str(tasks.ops[i][0]) != "DROP":
            vec[q * (len(CATEGORIES) + 1) + len(CATEGORIES)] += 1
    return vec


def first_hours_of(day, pool: int) -> np.ndarray:
    from agent.wsr.beam import _start_hours
    return np.asarray(_start_hours(day, pool), dtype=np.int8)


@dataclass
class Seed:
    """One stored route: the payload a lookup projects onto today's day."""
    pool: int
    hire_hours: tuple[int, ...]
    soft: list[int]
    route: list[tuple[int, str, int]]          # (hour, task_id, worker)
    categories: list[int]                      # per placement
    first_hours: dict[int, int]                # per worker
    doors: tuple[tuple[int, int], ...]
    complete: bool
    source: str


class SelfWarm:
    """The seed memory: npz on disk, list of Seed in memory."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else ARTIFACT_PATH
        self.hits = 0
        self.misses = 0
        self.seeds: list[Seed] = []
        self._load()

    def _load(self) -> None:
        self.seeds = []
        if not self.path.exists():
            return
        z = np.load(self.path, allow_pickle=False)
        vocab = [str(v) for v in z["vocab"]]
        n = len(z["pool"])
        for i in range(n):
            length = int(z["route_len"][i])
            route = [(int(z["route_hour"][i, k]),
                      vocab[int(z["route_task"][i, k])],
                      int(z["route_worker"][i, k])) for k in range(length)]
            cats = [int(z["route_cat"][i, k]) for k in range(length)]
            doors = tuple((int(d[0]), int(d[1]))
                          for d in z["doors"][i] if d[0] >= 0)
            first_hours = {int(w): int(h) for w, h in
                           enumerate(z["hire_hours"][i]) if h >= 0}
            self.seeds.append(Seed(
                pool=int(z["pool"][i]),
                hire_hours=tuple(int(h) for h in z["hire_hours"][i]),
                soft=[int(v) for v in z["soft"][i]],
                route=route,
                categories=cats,
                first_hours=first_hours,
                doors=doors,
                complete=True,
                source=f"seed{i}",
            ))

    def save(self) -> None:
        """Write the in-memory seeds to the npz artifact."""
        vocab: list[str] = []
        vidx: dict[str, int] = {}
        for s in self.seeds:
            for _h, tid, _w in s.route:
                if tid not in vidx:
                    vidx[tid] = len(vocab)
                    vocab.append(tid)
        n = len(self.seeds)
        max_len = max((len(s.route) for s in self.seeds), default=0)
        max_hire = max((len(s.hire_hours) for s in self.seeds), default=0)
        max_hands = max((len(s.doors) for s in self.seeds), default=0)
        pool = np.array([s.pool for s in self.seeds], dtype=np.int8)
        hire = np.zeros((n, max_hire), dtype=np.int8)
        for i, s in enumerate(self.seeds):
            hire[i, :len(s.hire_hours)] = s.hire_hours
        soft = np.array([s.soft for s in self.seeds], dtype=np.int32).reshape(n, SOFT_WIDTH) \
            if n else np.zeros((0, SOFT_WIDTH), dtype=np.int32)
        route_hour = np.full((n, max_len), -1, dtype=np.int16)
        route_task = np.full((n, max_len), -1, dtype=np.int16)
        route_worker = np.full((n, max_len), -1, dtype=np.int8)
        route_cat = np.full((n, max_len), -1, dtype=np.int8)
        for i, s in enumerate(self.seeds):
            for k, (h, tid, w) in enumerate(s.route):
                route_hour[i, k] = h
                route_task[i, k] = vidx[tid]
                route_worker[i, k] = w
            route_cat[i, :len(s.categories)] = s.categories
        route_len = np.array([len(s.route) for s in self.seeds], dtype=np.int16)
        doors = np.zeros((n, max_hands, 2), dtype=np.int8)
        for i, s in enumerate(self.seeds):
            for k, d in enumerate(s.doors):
                doors[i, k] = d
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, pool=pool, hire_hours=hire, soft=soft,
                            route_hour=route_hour, route_task=route_task,
                            route_worker=route_worker, route_cat=route_cat,
                            route_len=route_len, doors=doors,
                            vocab=np.array(vocab))

    # -- store -------------------------------------------------------------
    def store(self, day, tasks, result, *, verified: bool = True) -> Seed:
        """Add a solved route as a future seed. The seed carries the pool it
        was searched at, its own per-worker first hours and its settled
        doors: everything the re-projection needs to rebuild it on a similar
        day."""
        from agent.wsr.beam import _start_hours
        soft = soft_vector(day, tasks).tolist()
        pool = int(result.pool)
        cats = task_categories(tasks)
        id_idx = {str(tid): i for i, tid in enumerate(tasks.ids)}
        route = sorted((int(t), str(tid), int(w)) for t, tid, w in result.route)
        seed = Seed(
            pool=pool,
            hire_hours=tuple(int(h) for h in day.hire_times),
            soft=soft,
            route=route,
            categories=[cats[id_idx[str(tid)]] for _t, tid, _w in route
                        if str(tid) in id_idx],
            first_hours={w: int(h) for w, h in enumerate(_start_hours(day, pool))},
            doors=tuple(tuple(int(v) for v in d) for d in result.doors),
            complete=bool(result.complete),
            source="runtime",
        )
        self.seeds.append(seed)
        return seed

    # -- lookup ------------------------------------------------------------
    def lookup(self, day, tasks, hands: int) -> list:
        """Up to WARM_K seeds for `hands` workers on this day, nearest soft
        signature first, each re-projected and compile-verified."""
        if not self.seeds:
            self.misses += 1
            return []
        soft = soft_vector(day, tasks)
        day_hire = tuple(int(h) for h in day.hire_times)
        scored = []
        for s in self.seeds:
            if s.pool != hands:                 # hard gate: pool
                continue
            # hard gate: hands per hour. The npz row is zero-padded to the
            # widest seed; the padding is "no worker", so compare only the
            # first len(day.hire_times) entries and require the seed's own
            # vector to be exactly the day's (no grown workers beyond it).
            if s.hire_hours[:len(day_hire)] != day_hire:
                continue
            if any(h != 0 for h in s.hire_hours[len(day_hire):]):
                continue
            d = sum(abs(a - b) for a, b in zip(s.soft, soft.tolist()))
            scored.append((d, s))
        if not scored:
            self.misses += 1
            return []
        scored.sort(key=lambda x: x[0])
        out = []
        for _d, s in scored[:WARM_K]:
            seed = self._project(s, day, tasks)
            if seed is not None:
                out.append(seed)
        if out:
            self.hits += 1
        else:
            self.misses += 1
        return out

    # -- projection --------------------------------------------------------
    def _project(self, s: Seed, day, tasks):
        """Re-project one stored route onto today's tasks: each stored
        placement claims the nearest FREE task of the SAME CATEGORY to the
        worker's last projected position; a greedy re-timing then puts every
        kept placement inside its own window; the compile/check gate decides.
        A placement that cannot land legally is dropped (a partial seed is
        legal); an empty projection returns None."""
        cats = task_categories(tasks)
        by_cat: dict[int, list[int]] = defaultdict(list)
        for i in range(tasks.n):
            by_cat[int(cats[i])].append(i)

        claimed: set[int] = set()
        route: list[tuple[int, int, int]] = []
        worker_at: dict[int, tuple[int, int]] = {}
        board = int(getattr(tasks, "board_size", 10) or 10)
        for (s_hour, _tid, worker), cat in zip(s.route, s.categories):
            if worker > s.pool:
                continue
            free_tasks = by_cat.get(cat, [])
            best, best_d = None, None
            for i in free_tasks:
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
            route.append((s_hour, best, worker))
        if not route:
            return None
        return self._retime(route, tasks, day, s.pool,
                            {w: int(h) for w, h in s.first_hours.items()})

    def _retime(self, route, tasks, day, pool, first_hours: dict[int, int]):
        """Greedy legal-hours pass over a projected route: keep the placement
        ORDER per worker, move each op to the earliest hour that satisfies the
        task's window, the worker's own first hour and the walk from the
        worker's previous cell. A placement whose window cannot be met is
        DROPPED from the seed (a partial seed is legal; an out-of-day turn is
        not). Predecessor edges are honoured: an op runs after every task its
        chain orders before it — a re-timing that breaks `pred` would seed a
        row whose children all die in `_expand`. A bag consumer is pushed past
        the day's first good (`day_first_good`): its load happens at the door
        before it walks out."""
        board = int(getattr(tasks, "board_size", 10) or 10)
        from agent.wsr.beam import day_first_good
        at: dict[int, tuple[int, int]] = {}
        free: dict[int, int] = {}
        done_at: dict[int, int] = {}
        out = []
        for _turn, idx, worker in sorted(route, key=lambda x: (x[0], x[1])):
            cell = (int(tasks.cells[idx][0]), int(tasks.cells[idx][1]))
            if worker in free:
                base = free[worker] + 1
            elif worker < len(first_hours):
                base = int(first_hours[worker])
            elif worker < len(day.hire_times):
                base = int(day.hire_times[worker])
            else:
                continue
            walk = abs(at.get(worker, (board // 2, board // 2))[0] - cell[0]) \
                 + abs(at.get(worker, (board // 2, board // 2))[1] - cell[1])
            t = max(base + walk, int(tasks.earliest[idx]))
            preds = [int(j) for j in np.nonzero(tasks.pred[idx])[0]]
            if preds:
                if any(j not in done_at for j in preds):
                    continue
                t = max(t, max(done_at[j] for j in preds) + 1)
            if str(tasks.ops[idx][0]) in CONSUMERS:
                t = max(t, int(day_first_good(tasks)) + 1)
            if t > int(tasks.latest[idx]) or t >= int(day.horizon):
                continue          # outside the window or off the day: skip
            done_at[idx] = t
            free[worker] = t
            at[worker] = cell
            out.append((t, str(tasks.ids[idx]), int(worker)))
        out.sort()
        from agent.wsr.beam import Result
        return Result(pool=pool, route=out, complete=False)


_MEMORY: SelfWarm | None = None


def selfwarm() -> SelfWarm:
    """The season's shared self-warm memory."""
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = SelfWarm()
    return _MEMORY


def candidates_for(day, tasks, hands: int) -> list:
    """The entry point `search` calls when the caller passed no warm: up to
    WARM_K verified seeds, nearest soft signature first."""
    return selfwarm().lookup(day, tasks, hands)
