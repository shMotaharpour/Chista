"""Bench the warm memory on the corpus: does a warm start reduce solve time,
and does any xfail day get rescued? Three arms per day:
  cold            no memory
  warm(artifact)  the built artifact loaded
  warm(all)       every OTHER corpus day's solution in memory (the maximum
                  coverage the artifact could ever have)
Also: how many seeds is optimal? Sweep WARM_K over the corpus with the
full memory and report time + placed per k.
"""
import json, pathlib, sys, time
sys.path.insert(0, ".")
from agent.wsr import beam as B, tasks as T, warm as W
from agent.wsr.emit import compile_route

CORPUS = json.loads(pathlib.Path("tests/day_layer/corpus/winner_days.json").read_text())
ARTIFACT = json.loads(pathlib.Path("agent/artifact/warm_routes.json").read_text())

def bench_day(e, memory_entries, k):
    grid = [(tuple(c), tuple(o), en) for c, o, en in e["chains"]]
    n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
    tasks = T.build(grid, available={g: int(h) for g, h in e["available"].items()},
                    drop_by=[12] * n_shed)
    day = B.Day(chains=tuple(grid),
                available={g: int(h) for g, h in e["available"].items()},
                hire_times=tuple(e["hire_times"]) or (1,) * e["hands"])
    old_k = W.WARM_K
    W.WARM_K = k
    m = W.memory()
    m._entries = list(memory_entries)
    m.hits = m.misses = 0
    t0 = time.perf_counter()
    r = B.search(day, tasks, hands=e["hands"] - 1, max_hands=e["hands"])
    dt = time.perf_counter() - t0
    W.WARM_K = old_k
    hit = m.hits > 0
    return dt, len(r.route), tasks.n, r.complete, hit

def main():
    entries = CORPUS[:12]          # the days the artifact covered
    print(f"== arm 1: artifact memory ({len(ARTIFACT)} seeds) vs cold, k=8 ==")
    for e in entries:
        tag = f"{e['episode']}/d{e['day']}"
        tc, pc, nc, cc, _ = bench_day(e, [], 8)
        tw, pw, nw, cw, hit = bench_day(e, ARTIFACT, 8)
        same = (pw == pc)
        print(f"  {tag}: cold {tc:6.1f}s placed={pc}/{nc} | "
              f"warm {tw:6.1f}s placed={pw}/{nw} hit={hit} "
              f"| {'same' if same and cc == cw else 'CHANGED: ' + ('better' if pw > pc else 'worse')}")

    print("\n== arm 2: full memory (every other day as seed), k=8 ==")
    all_but = [b for b in ARTIFACT]
    extra = []
    # add every corpus day's own solution so a same-day hit is possible
    from agent.wsr.emit import compile_route
    for e in CORPUS[:12]:
        grid = [(tuple(c), tuple(o), en) for c, o, en in e["chains"]]
        n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
        tasks = T.build(grid, available={g: int(h) for g, h in e["available"].items()},
                        drop_by=[12] * n_shed)
        day = B.Day(chains=tuple(grid),
                    available={g: int(h) for g, h in e["available"].items()},
                    hire_times=tuple(e["hire_times"]) or (1,) * e["hands"])
        r = B.search(day, tasks, hands=e["hands"] - 1, max_hands=e["hands"])
        if r.complete:
            cats = W._task_categories(tasks)
            idx = {str(t): i for i, t in enumerate(tasks.ids)}
            extra.append({
                "sig": W.signature(day, tasks), "pool": int(r.pool),
                "route": [(int(t), str(tid), int(w)) for t, tid, w in r.route],
                "categories": [cats[idx[str(tid)]] for _t, tid, _w in r.route],
                "complete": True, "verified": True,
                "source": f"own/{e['episode']}/d{e['day']}"})
    mem = ARTIFACT + extra
    print(f"  memory size: {len(mem)} seeds")
    for e in entries:
        tag = f"{e['episode']}/d{e['day']}"
        tc, pc, nc, cc, _ = bench_day(e, [], 8)
        tw, pw, nw, cw, hit = bench_day(e, mem, 8)
        print(f"  {tag}: cold {tc:6.1f}s placed={pc}/{nc} | "
              f"warm {tw:6.1f}s placed={pw}/{nw} hit={hit} "
              f"| {'complete!' if cw and not cc else ''}")

    print("\n== arm 3: WARM_K sweep (full memory, aggregate over 6 days) ==")
    six = entries[:6]
    for k in (1, 2, 4, 8):
        tot = 0.0
        placed = 0
        ntot = 0
        for e in six:
            dt, p, n, _c, _h = bench_day(e, mem, k)
            tot += dt; placed += p; ntot += n
        print(f"  k={k}: total {tot:6.1f}s, placed {placed}/{ntot}")

if __name__ == "__main__":
    main()
