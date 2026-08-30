# Kaggriculture replays → queryable store

## Why the raw JSON is a bad query target

Measured on `data/raw/Samples/` (16 episodes, 456 MB of JSON):

| Fact | Value | Consequence |
|---|---|---|
| `steps` share of each file | 99.98 % | everything else (`specification`, `configuration`, schema text) is noise repeated per file |
| player-1's `observation.farms` / `market` / `town` | **byte-identical** to player-0 in 720/720 steps | half of every file is a literal duplicate |
| tile cells that change between consecutive steps | ~5 % | the 10×10 board is re-serialised in full 720× though almost nothing moved |
| whole-file gzip ratio | ~66× | the JSON is overwhelmingly repetition |
| actions (both players, whole game) | 278 KB | the actually-interesting signal is tiny |

## Target format: Parquet + DuckDB

- **Parquet** (columnar, zstd) — column-per-field + dictionary/RLE compression collapses the repetition the JSON can't.
- **DuckDB** — reads Parquet straight off disk, out-of-core, spills to disk. On 4 GB RAM it never needs to load a table whole. No server, single file, `pip install duckdb`.
- Avoid: loading JSON into pandas (10–15× RAM blow-up), Postgres/a server (setup + RAM), MongoDB (keeps the redundancy).

## Result

`convert_replays.py` streams **one episode at a time** (peak RSS ~530 MB, well under 4 GB) and writes 7 normalised tables:

| table | grain | notes |
|---|---|---|
| `episodes` | 1 row / game | agents, seed, rewards, config knobs |
| `steps` | 1 row / step | shared state only: day/hour, 9 market prices, 9 inventories, town shops |
| `farm_steps` | 1 row / step / player | money, farmer xy, #hands, quadrants |
| `private_steps` | 1 row / step / player | shed + seed counts |
| `actions` | 1 row / unit-action | farmer (`unit=-1`) + each hand, `op/arg1/arg2` |
| `market_orders` | 1 row / order | `op/item/qty` |
| `tiles_delta` | 1 row / **changed** tile | delta-encoded; reconstruct full board with a window function |

16 sample episodes → **716 KB** of Parquet (~640× smaller than the JSON).

### Projection for the 10 zips
- ~700 episodes/zip → ~7 000 episodes total
- ≈ **300–500 MB** of Parquet total (fits in RAM, let alone on disk)
- ~2 s/episode → ~4 h single-threaded, one-time. Keep it single-threaded to keep RAM flat.

## Run it

```bash
pip install pyarrow duckdb           # conda: already set up in miniforge base
python convert_replays.py data/raw/Samples/ -o data/parquet/
# later, straight from a zip (no extraction):
python convert_replays.py data/raw/raw_zip/ -o data/parquet/
```

Re-running appends nothing — it overwrites `data/parquet/`. To process zips incrementally,
point `-o` at a per-zip folder (`data/parquet/2026-08-15/`) and glob all of them at query time:
`FROM 'data/parquet/*/episodes.parquet'`.

## Query examples

```python
import duckdb
con = duckdb.connect("kaggri.duckdb")          # or connect() for in-memory
con.execute("CREATE VIEW episodes AS SELECT * FROM 'data/parquet/**/episodes.parquet'")
con.execute("CREATE VIEW steps    AS SELECT * FROM 'data/parquet/**/steps.parquet'")
con.execute("CREATE VIEW actions  AS SELECT * FROM 'data/parquet/**/actions.parquet'")
con.execute("CREATE VIEW tiles    AS SELECT * FROM 'data/parquet/**/tiles_delta.parquet'")
```

```sql
-- per-agent win rate
SELECT agent, count(*) games, sum(won) wins, round(avg(m)) avg_money
FROM (SELECT unnest([agent0,agent1]) agent,
             unnest([reward0,reward1]) m,
             unnest([(reward0>reward1)::int,(reward1>reward0)::int]) won
      FROM episodes)
GROUP BY 1 ORDER BY wins DESC;

-- what winners plant vs losers
SELECT a.arg1 crop, count(*)
FROM actions a JOIN episodes e USING(episode_id)
WHERE a.op='PLANT'
  AND ((a.player=0 AND e.reward0>e.reward1) OR (a.player=1 AND e.reward1>e.reward0))
GROUP BY 1 ORDER BY 2 DESC;

-- reconstruct full board at an arbitrary step (last-value-carried-forward)
SELECT x, y, last(kind) OVER w AS kind, last(crop) OVER w AS crop
FROM tiles
WHERE episode_id = 90025221 AND player = 0 AND step <= 400
WINDOW w AS (PARTITION BY x,y ORDER BY step
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
QUALIFY row_number() OVER (PARTITION BY x,y ORDER BY step DESC) = 1;
```

## Tuning knobs (if the store is still bigger than you want)

- **Drop `tiles_delta`** entirely if your questions are economic (money/market/actions). It's the largest table (~340 KB of the 716 KB) because `watered_today` toggles daily.
- **Down-sample tiles to once per day**: only emit a tile row when `hour == turnsPerDay-1`. ~24× fewer rows.
- **Skip `private_steps` / `farm_steps` mid-day**: sample every Nth step; money is monotone-ish between trades.
- **Partition by day** in Parquet (`partition_cols`) if you routinely filter by game phase.
