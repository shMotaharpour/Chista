"""Build the offline empirical spatial transition matrix from winning competition replays.

Queries winning player steps across replay parquet dates using DuckDB, aggregates
cell-to-cell transitions (c1 -> c2) across the 10x10 farm grid, normalizes the
frequencies into an empirical transition likelihood matrix, and saves to
`agent/artifact/spatial_transitions.npy`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import duckdb
import numpy as np


DEFAULT_PARQUET_DIR = Path("/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet")
DEFAULT_OUT_PATH = Path(__file__).resolve().parents[2] / "agent" / "artifact" / "spatial_transitions.npy"
BOARD_SIZE = 10


def build_spatial_transitions(
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    out_path: Path = DEFAULT_OUT_PATH,
    sample_dates_count: int = 10,
) -> np.ndarray:
    """Extract winning transitions from parquet replays and save normalized matrix."""
    print(f"Building Spatial Transition Matrix from {parquet_dir}...")
    t0 = time.perf_counter()

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    dates = sorted([d.name for d in parquet_dir.glob("2026-*") if d.is_dir()])
    if not dates:
        raise ValueError(f"No date subdirectories found in {parquet_dir}")

    step = max(1, len(dates) // sample_dates_count)
    sample_dates = dates[::step][:sample_dates_count]
    print(f"Sampling {len(sample_dates)} dates across season: {sample_dates}")

    hands_files = [str(parquet_dir / d / "hands_steps.parquet") for d in sample_dates]
    episodes_files = [str(parquet_dir / d / "episodes.parquet") for d in sample_dates]

    # DuckDB query across the parquet files for winning players
    query = f"""
    WITH wins AS (
        SELECT episode_id, (reward1 > reward0) AS winner_player
        FROM read_parquet({episodes_files})
        WHERE reward0 != reward1
    ),
    winner_steps AS (
        SELECT h.episode_id, h.unit, h.step, h.x, h.y, h.op,
               LAG(h.x) OVER (PARTITION BY h.episode_id, h.unit ORDER BY h.step) as prev_x,
               LAG(h.y) OVER (PARTITION BY h.episode_id, h.unit ORDER BY h.step) as prev_y
        FROM read_parquet({hands_files}) h
        JOIN wins w ON h.episode_id = w.episode_id AND h.player = w.winner_player
        WHERE h.op NOT IN ('PASS', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'HIRE')
    )
    SELECT prev_y * {BOARD_SIZE} + prev_x as from_cell, y * {BOARD_SIZE} + x as to_cell, COUNT(*) as count
    FROM winner_steps
    WHERE prev_x IS NOT NULL AND prev_y IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL
          AND prev_x >= 0 AND prev_x < {BOARD_SIZE} AND prev_y >= 0 AND prev_y < {BOARD_SIZE}
          AND x >= 0 AND x < {BOARD_SIZE} AND y >= 0 AND y < {BOARD_SIZE}
    GROUP BY from_cell, to_cell;
    """

    df = duckdb.query(query).to_df()
    elapsed = time.perf_counter() - t0
    total_count = int(df['count'].sum())
    print(f"Extracted {len(df)} unique cell transitions in {elapsed:.2f}s (total steps: {total_count:,})")

    # Construct 100x100 matrix
    matrix = np.zeros((BOARD_SIZE ** 2, BOARD_SIZE ** 2), dtype=np.float32)
    for _, row in df.iterrows():
        c1, c2, cnt = int(row["from_cell"]), int(row["to_cell"]), float(row["count"])
        matrix[c1, c2] = cnt

    # Log-scaling and row normalization
    log_matrix = np.log1p(matrix)
    row_max = log_matrix.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    norm_matrix = (log_matrix / row_max).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, norm_matrix)
    size_kb = out_path.stat().st_size / 1024
    print(f"Successfully saved normalized transition matrix to {out_path} ({size_kb:.1f} KB)")
    return norm_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline spatial transition matrix.")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR, help="Path to replays parquet directory")
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH, help="Output .npy file path")
    parser.add_argument("--samples", type=int, default=10, help="Number of dates to sample across the season")
    args = parser.parse_args()

    build_spatial_transitions(args.parquet_dir, args.out_path, args.samples)
