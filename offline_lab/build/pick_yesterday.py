"""Pick yesterday's episodes for the builder's Kaggle test.

Reads the parquet store's newest partition (2026-09-26) and prints 20
episode ids to download: the 10 highest-scoring episodes and 10 random
others (so the test covers both elite play and ordinary play).
"""

from __future__ import annotations

import glob
import sys

import duckdb

STORE = "/home/amirelite_ai/Chista/kaggriculture-episodes-analyses/data/replays_parquet"
PARTS = sorted(glob.glob(f"{STORE}/2026-09-2*"))
part = PARTS[-1]
print(f"partition: {part}", file=sys.stderr)

con = duckdb.connect()
con.execute("SET memory_limit='600MB'")
q = f"""
SELECT episode_id, greatest(reward0, reward1) AS top_reward
FROM read_parquet('{part}/episodes.parquet')
WHERE status0 = 'DONE' AND status1 = 'DONE'
ORDER BY top_reward DESC
LIMIT 10
"""
top = con.execute(q).fetchall()

q2 = f"""
SELECT episode_id, greatest(reward0, reward1) AS top_reward
FROM read_parquet('{part}/episodes.parquet')
WHERE status0 = 'DONE' AND status1 = 'DONE'
  AND episode_id NOT IN (SELECT episode_id FROM ({q}))
USING SAMPLE 10
"""
rest = con.execute(q2).fetchall()

for eid, r in top + rest:
    print(f"{eid} {int(r)}")
