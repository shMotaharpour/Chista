"""Build ONE parquet day's warm partition, end to end, to prove the design.

Hard keys (exact): pool + hire_times vector -> the partition.
Soft key: the (category x quadrant) density -> distance ranking.
Payload: the WINNER's real route from hands_steps (hours + cells), mapped to
wsr task ids per day, compile-verified, stored as a seed.
Encoded compactly (npz-style arrays in a json list; one file per partition
pool so lookups are O(1) by name).
"""
import sys, json, pathlib
sys.path.insert(0, ".")
import duckdb
import numpy as np
from collections import defaultdict, Counter
from agent.wsr import beam as B, tasks as T, warm as W
from agent.wsr.emit import check_route, compile_route

REPO = pathlib.Path("/home/amirelite_ai/Chista/kaggriculture-episodes-analyses/data/replays_parquet")
CORPUS = json.loads(pathlib.Path("tests/day_layer/corpus/winner_days.json").read_text())
BS = 10

def quad(c):
    x, y = int(c[0]), int(c[1])
    return (y // (BS // 2)) * 2 + (x // (BS // 2))

con = duckdb.connect()

def day_rows(dump, ep, day):
    q = f"""
    WITH w AS (
      SELECT episode_id, CASE WHEN reward0 >= reward1 THEN 0 ELSE 1 END AS wp
      FROM read_parquet('{REPO}/{dump}/episodes.parquet') WHERE episode_id = {ep}
    )
    SELECT h.unit, c.hour, h.op, h.arg1, h.arg2, h.x, h.y
    FROM read_parquet('{REPO}/{dump}/hands_steps.parquet') h
    JOIN w ON h.episode_id = w.episode_id AND h.player = w.wp
    JOIN read_parquet('{REPO}/{dump}/city_steps.parquet') c
      ON c.episode_id = h.episode_id AND c.step = h.step
    WHERE c.day = {day} AND h.op IS NOT NULL AND h.op != 'PASS'
    ORDER BY h.unit, c.hour
    """
    return con.execute(q).fetchall()

# pick the corpus day and its entry
e = CORPUS[0]
dump, ep, dnum = e["dump"], e["episode"], e["day"]
grid = [(tuple(c), tuple(o), en) for c, o, en in e["chains"]]
n_shed = sum(1 for _c, o, _e in grid if "FEED" in o or "FERTILIZE" in o)
tasks = T.build(grid, available={g: int(h) for g, h in e["available"].items()},
                drop_by=[12] * n_shed)
day = B.Day(chains=tuple(grid),
            available={g: int(h) for g, h in e["available"].items()},
            hire_times=tuple(e["hire_times"]) or (1,) * e["hands"])
cats = W._task_categories(tasks)

# hard key: pool (workers incl. farmer = hands) + hire vector exactly
pool = e["hands"]
hire_vec = [int(h) for h in (e["hire_times"] or [1] * e["hands"])]
hard_key = {"pool": pool, "hire_times": hire_vec}

rows = day_rows(dump, ep, dnum)
print(f"store rows for winner on {dump}/{ep} d{dnum}: {len(rows)}")

# map recorded (unit, hour, op, x, y) to task ids: same op-name + nearest cell,
# claimed in time order per unit — the game's own order.
by_key = defaultdict(list)
for i in range(tasks.n):
    by_key[(str(tasks.ops[i][0]), int(tasks.cells[i][0]), int(tasks.cells[i][1]))].append(i)

route = []
claimed = set()
worker_at = {}
unmatched = 0
for unit, hour, op, a1, a2, x, y in rows:
    x, y = int(x), int(y)
    cands = [i for i in by_key.get((str(op), x, y), []) if i not in claimed]
    if not cands:
        near = [(abs(int(tasks.cells[i][0]) - x) + abs(int(tasks.cells[i][1]) - y), i)
                for i in range(tasks.n)
                if str(tasks.ops[i][0]) == str(op) and i not in claimed]
        if near:
            i = min(near)[1]
        else:
            unmatched += 1
            continue
    else:
        i = cands[0]
    claimed.add(i)
    worker_at.setdefault(unit, (x, y))
    route.append({"h": int(hour), "i": i, "w": int(unit)})

print(f"matched {len(route)} ops to tasks, unmatched {unmatched}, "
      f"workers {sorted({r['w'] for r in route})}")

# soft key: category x quadrant density of the day
sig_cats = Counter()
tiles = Counter()
for i in range(tasks.n):
    q = quad(tasks.cells[i])
    sig_cats[(quad(tasks.cells[i]), cats[i])] += 1
    if str(tasks.ops[i][0]) != "DROP":
        tiles[q] += 1
soft = {f"Q{q}_{c}": n for (q, c), n in sorted(sig_cats.items())}
soft.update({f"Q{q}_tiles": tiles[q] for q in range(4)})

# verify the seed compiles
seed = B.Result(pool=pool, route=sorted((r["h"], tasks.ids[r["i"]], r["w"]) for r in route),
                complete=False)
viol = check_route(day, tasks, seed)
try:
    compile_route(day, tasks, seed, horizon=24)
    compiles = True
except (ValueError, IndexError) as ex:
    compiles = False
    print("compile note:", str(ex)[:120])
print("check_route:", "OK" if not viol else viol[:3], "| compiles:", compiles)

# encoded payload: compact arrays, one JSON per partition-pool
payload = {
    "pool": pool,
    "hire_times": hire_vec,
    "task_ids": [str(t) for t in tasks.ids],     # vocabulary of this seed
    "soft": soft,
    "seeds": [{
        "episode": ep, "day": dnum,
        "hours": [r["h"] for r in route],
        "task_idx": [r["i"] for r in route],
        "workers": [r["w"] for r in route],
    }],
}
out = pathlib.Path("/tmp/warm_one_day.json")
out.write_text(json.dumps(payload))
print(f"\nwritten {out}: {out.stat().st_size / 1024:.1f} KB")
print("structure: hard_key (pool+hire_times) -> soft vector -> compact seed arrays")
