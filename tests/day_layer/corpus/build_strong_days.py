"""Rebuild the strong-player corpus: the days a cumulative top-10 player WON, from the winner's side.

The corpus the layer is measured on (`real_days.json`) takes player-0 of an episode, chosen by episode
id plus a seed, with no filter on who played it. Measured over the archive: 5 of its 102 days belong to
a cumulative top-10 player, and player-0 is the LOSING side in 67 of those 102 episodes - so in two
thirds of the cases the target the tests ask for is the weaker player's own day.

This builds the opposite sample, so the corpus test answers the competitive question: can the layer
carry a *good* player's day? Every entry is the winner's side of an episode whose winner is in the
cumulative top ten by wins (the same cut AGT0001 in the analyses repo uses), and each entry carries the
player's name and its win count so the file says who it is built from.

Not a test - a tool, run by hand, and it reads the two-gigabyte archive:

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \\
      python tests/day_layer/corpus/build_strong_days.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import duckdb

from build_real_days import ROOT, dumps, extract

OUT = pathlib.Path(__file__).parent / "strong_days.json"
#: The cut: the ten agents with the most wins across the archive, as AGT0001 ranks them.
STRONG = 10
#: Every dump gives its mid-season day; every fifth gives a season's worth, so the sample covers the
#: season and not one hour of it.
MID_DAY = 5
DAY_EVERY = 5
SEASON_DAYS = (1, 10, 15, 20, 25)


def leaders(con, every: list[str]) -> tuple[list[str], dict[str, int]]:
    """The agents with the most wins across every dump, and how many each has."""
    files = "[" + ", ".join(f"'{ROOT}/{dump}/episodes.parquet'" for dump in every) + "]"
    wins = con.sql(f"""
        SELECT agent, count(*) AS wins FROM (
            SELECT agent0 AS agent FROM read_parquet({files}) WHERE reward0 > reward1
            UNION ALL
            SELECT agent1 AS agent FROM read_parquet({files}) WHERE reward1 > reward0
        ) GROUP BY agent ORDER BY wins DESC, agent
    """).df()
    names = [str(name) for name in wins["agent"].head(STRONG)]
    return names, {str(row.agent): int(row.wins) for row in wins.itertuples()}


def a_win(con, dump: str, strong: list[str]):
    """The first episode of this dump that a strong player won, or None."""
    names = "(" + ", ".join("'" + name.replace("'", "''") + "'" for name in strong) + ")"
    rows = con.sql(f"""
        SELECT episode_id, agent0, agent1, reward0, reward1
        FROM '{ROOT}/{dump}/episodes.parquet'
        WHERE (reward0 > reward1 AND agent0 IN {names})
           OR (reward1 > reward0 AND agent1 IN {names})
        ORDER BY episode_id LIMIT 1
    """).fetchall()
    return rows[0] if rows else None


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    every = dumps()
    strong, wins_of = leaders(con, every)
    print(f"{len(every)} dumps | the strong set: {strong}")

    days = []
    for index, dump in enumerate(every):
        picked = a_win(con, dump, strong)
        if not picked:
            print(f"  {dump}: no strong winner")
            continue
        episode, agent0, agent1, reward0, reward1 = picked
        won_first = int(reward0) > int(reward1)
        winner = str(agent0) if won_first else str(agent1)
        loser = str(agent1) if won_first else str(agent0)
        plan = (MID_DAY,) if index % DAY_EVERY else (MID_DAY, *SEASON_DAYS)
        found_here = 0
        for day in plan:
            found = extract(con, dump, int(episode), day, side=0 if won_first else 1)
            if found:
                found.update({"agent": winner, "wins": wins_of.get(winner, 0), "opponent": loser})
                days.append(found)
                found_here += 1
        print(f"  {dump}: ep {episode} won by {winner} ({wins_of.get(winner, 0)} wins) "
              f"vs {loser} -> {found_here} days")

    OUT.write_text(json.dumps(days, separators=(",", ":")))
    print(f"{len(days)} days from {len({d['dump'] for d in days})} dumps, "
          f"{len({d['agent'] for d in days})} strong players, {OUT.stat().st_size} bytes -> {OUT}")


if __name__ == "__main__":
    main()
