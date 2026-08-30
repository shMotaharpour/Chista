"""Tiny helper: open a DuckDB connection with views over the Parquet tables.

    from kdb import connect
    con = connect()                      # uses ./data/parquet
    con.sql("select * from episodes").df()
"""
from __future__ import annotations
import os
import duckdb

TABLES = ["episodes", "steps", "farm_steps", "private_steps",
          "actions", "market_orders", "tiles_delta"]


def connect(parquet_dir: str = "data/parquet", database: str = ":memory:"):
    con = duckdb.connect(database)
    for t in TABLES:
        # glob matches both flat (data/parquet/x.parquet) and per-zip subdirs
        pattern = os.path.join(parquet_dir, "**", f"{t}.parquet")
        flat = os.path.join(parquet_dir, f"{t}.parquet")
        src = pattern if not os.path.exists(flat) else f"{flat}"
        con.execute(
            f"CREATE OR REPLACE VIEW {t} AS "
            f"SELECT * FROM read_parquet('{parquet_dir}/**/{t}.parquet', "
            f"union_by_name=true, hive_partitioning=false)"
        )
    # long-form (agent, money, won) one row per player per game — handy for seaborn
    con.execute("""
        CREATE OR REPLACE VIEW agent_games AS
        SELECT episode_id, seed,
               unnest([agent0, agent1])                                   AS agent,
               unnest([0, 1])                                             AS player,
               unnest([reward0, reward1])                                 AS money,
               unnest([(reward0 > reward1), (reward1 > reward0)])         AS won,
               unnest([(reward0 = reward1), (reward0 = reward1)])         AS tied
        FROM episodes
    """)
    return con
