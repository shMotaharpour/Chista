"""Rebuild the winner-day corpus: strong winners, one day each, by bucket.

Replaces the archive-extraction corpora (`real_days.json`, `strong_days.json`,
`last_days.json`) per the owner's order (2026-09-23): every day comes from the
WINNER'S side of a game whose player has a win rate above 60% (at least 10
games across the sampled dumps), 3 days per unlocked-quadrant tier (1..4),
plus 3 self-serve-mix days (FEED after own HARVEST / FERTILIZE after own
COLLECT_FERTILIZER, no shed pickup), plus 3 day-29 days with SELL-backed
drop deadlines, plus the 3 most-op days.

Extraction rules (the two corrections the fork run exposed):
- an op is kept only if its (cell, hour) changed AND every op submitted at
  that (cell, hour) is the same op — when units disagree, no one can be
  credited and the day must not ask for the unattributable op;
- ops on tiles LOCKED at hour 0 are dropped (the game bought that quadrant
  mid-day; the day the solver is given opens with the hour-0 unlock set).

    KAGGLE_REPLAYS_PARQUET=/path/to/replays_parquet \
      python tests/day_layer/corpus/build_winner_days.py
"""
from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict

import duckdb
import pandas as pd

ROOT = os.environ.get(
    "KAGGLE_REPLAYS_PARQUET",
    "/chista/Chista/kaggriculture-episodes-analyses/data/replays_parquet",
)
OUT = pathlib.Path(__file__).parent / "winner_days.json"
DUMPS = ["2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20"]
MIN_GAMES = 10
WIN_RATE = 0.6
PER_TIER = 3
PER_AGENT_EPISODES = 8

MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP"}
PRODUCTS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}


def quadrant_of(x: int, y: int) -> str:
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def _clean(v) -> int | None:
    if v is None or str(v) in ('nan', 'None', 'NA', '<NA>', ''):
        return None
    s = str(v)
    try:
        return int(float(s))
    except ValueError:
        return None


def winners_table(con) -> pd.DataFrame:
    """Per-agent games/wins over the sampled dumps."""
    parts = " UNION ALL ".join(
        f"SELECT agent0 AS agent, reward0 AS r0, reward1 AS r1 "
        f"FROM '{ROOT}/{d}/episodes.parquet' WHERE status0='DONE' AND status1='DONE'"
        f" UNION ALL "
        f"SELECT agent1, reward1, reward0 FROM '{ROOT}/{d}/episodes.parquet' "
        f"WHERE status0='DONE' AND status1='DONE'"
        for d in DUMPS)
    return con.sql(f"""
        SELECT agent, count(*) AS games,
               count(*) FILTER (WHERE r0 > r1) AS wins
        FROM ({parts}) GROUP BY agent
        HAVING count(*) >= {MIN_GAMES}
           AND count(*) FILTER (WHERE r0 > r1) > {WIN_RATE} * count(*)
        ORDER BY wins DESC
    """).df()


def won_episodes(con, agent: str, limit: int) -> list[tuple[str, int, bool, int, int]]:
    """(dump, episode_id, player_side, reward, loser) of games the agent won."""
    rows = []
    for d in DUMPS:
        got = con.execute(f"""
            SELECT episode_id, reward0, reward1, agent0 FROM '{ROOT}/{d}/episodes.parquet'
            WHERE status0='DONE' AND status1='DONE'
              AND ((reward0 > reward1 AND agent0 = $a) OR (reward1 > reward0 AND agent1 = $a))
            ORDER BY episode_id LIMIT {limit}
        """, {"a": agent}).fetchall()
        for ep, r0, r1, a0 in got:
            won_first = a0 == agent
            rows.append((d, int(ep), not won_first,
                         int(r0 if won_first else r1),
                         int(r1 if won_first else r0)))
    return rows


def day_meta(con, dump: str, ep: int, player: bool) -> dict[int, dict]:
    """Per in-game day: the quadrant string at hour 0 and a tile-change count."""
    pl = 'true' if player else 'false'
    qs = con.sql(f"""
        SELECT step // 24 AS day, quadrants FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step % 24 = 0
    """).fetchall()
    quads = {int(d): str(q or '') for d, q in qs}
    proxy = con.sql(f"""
        SELECT step // 24 AS day, count(DISTINCT (x, y, step % 24)) AS changes
        FROM '{ROOT}/{dump}/tiles_delta.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 BETWEEN 1 AND 29
        GROUP BY 1
    """).fetchall()
    changes = {int(d): int(c) for d, c in proxy}
    out = {}
    for day in range(1, 30):
        q = quads.get(day, '')
        out[day] = {'quadrants': q,
                    'n_quad': len([t for t in q.split(',') if t]),
                    'changes': changes.get(day, 0)}
    return out


def extract_winner_day(con, dump: str, ep: int, day: int, player: bool,
                       agent: str, reward: int, loser: int,
                       wins: int, games: int, win_rate: float,
                       selfserve: str = '') -> dict | None:
    """One winner-side day, extracted with attribution and the unlock filter."""
    pl = 'true' if player else 'false'
    step0 = day * 24

    quad_row = con.sql(f"""
        SELECT quadrants, money FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step={step0}
    """).fetchone()
    quads = set(str(quad_row[0] or '').split(',')) - {''}
    money = _clean(quad_row[1])

    submitted = con.sql(f"""
        SELECT step % 24 AS hour, farmer_x AS x, farmer_y AS y, op
        FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        UNION ALL
        SELECT step % 24 AS hour, x, y, op
        FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
    """).df()
    changed = con.sql(f"""
        SELECT DISTINCT x, y, step % 24 AS hour
        FROM '{ROOT}/{dump}/tiles_delta.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
    """).df()
    happened = {(int(r.x), int(r.y), int(r.hour)) for r in changed.itertuples()}

    at_cell_hour: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for r in submitted.itertuples():
        if isinstance(r.op, str) and r.op not in MOVES:
            at_cell_hour[(int(r.x), int(r.y), int(r.hour))].add(r.op)

    chains: dict[tuple[int, int], list[str]] = {}
    op_hours: dict[tuple[int, int], list[int]] = {}
    for r in submitted.itertuples():
        if not isinstance(r.op, str) or r.op in MOVES:
            continue
        cell = (int(r.x), int(r.y))
        if quadrant_of(*cell) not in quads:
            continue                    # locked at hour 0: the day cannot ask for it
        key = (cell[0], cell[1], int(r.hour))
        if key not in happened:
            continue                    # refused in silence (F047)
        if len(at_cell_hour[key]) > 1:
            continue                    # unattributable: units disagree at (cell, hour)
        chains.setdefault(cell, []).append(r.op)
        op_hours.setdefault(cell, []).append(int(r.hour))

    if not chains:
        return None

    last = con.sql(f"""
        SELECT x, y, kind, crop, animal
        FROM '{ROOT}/{dump}/tiles_delta.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        QUALIFY row_number() OVER (PARTITION BY x, y ORDER BY step DESC) = 1
    """).df()
    entity = {}
    for r in last.itertuples():
        if r.crop is not None and r.crop is not pd.NA:
            entity[(int(r.x), int(r.y))] = str(r.crop)
        elif r.animal is not None and r.animal is not pd.NA:
            entity[(int(r.x), int(r.y))] = str(r.animal)

    hours = [int(h) for (h,) in con.sql(f"""
        SELECT min(step % 24) FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        GROUP BY unit ORDER BY unit
    """).fetchall()]

    # The fork's configuration is not in the archive; infer the hire-cost
    # multiplier from the day's own record: a hand that materialized while the
    # purse never DROPPED below its previous hour's value means hiring was
    # free (mult=0) — with mult=1 the n-th hire costs fib(n) >= 1, so the
    # money row always dips when hands appear.
    mh = con.sql(f"""
        SELECT step % 24 AS hour, money, n_hands FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day} ORDER BY step
    """).fetchall()
    hire_mult = 1
    prev_money = None
    prev_n = 0
    for _h, m, n in mh:
        mv = _clean(m)
        if prev_money is not None and n > prev_n and mv is not None \
                and mv >= prev_money:
            hire_mult = 0
            break
        prev_money = mv
        prev_n = n

    shed_row = con.sql(f"""
        SELECT * FROM '{ROOT}/{dump}/private_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day} AND step % 24 = 0
    """).df()
    available = {}
    shed_counts = {}
    seeds = {}
    if not shed_row.empty:
        for k, v in shed_row.iloc[0].to_dict().items():
            c = _clean(v)
            if k.startswith('shed_') and c is not None and c > 0:
                available[k[5:]] = 1          # the layer's hour flag (has it / at 0)
                shed_counts[k[5:]] = c        # the fork's TRUE shed counts
            if k.startswith('seed_') and c is not None and c > 0:
                seeds[k[5:]] = c

    # --- the recorded solution: the day's own actions, replayable on FastSim.
    # units[0] is the farmer, units[k] hand k-1 (hands_steps.unit is 0-based);
    # each is 24 hour slots of [op, args...] with PASS for the idle hours.
    farm_acts = con.sql(f"""
        SELECT step % 24 AS hour, op, arg1, arg2 FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day} ORDER BY hour
    """).fetchall()
    hand_acts = con.sql(f"""
        SELECT unit, step % 24 AS hour, op, arg1, arg2
        FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        ORDER BY unit, hour
    """).fetchall()

    def _act(op, a1, a2) -> list:
        out = [str(op)]
        for a in (a1, a2):
            if a is None or str(a) in ('nan', 'None', 'NA'):
                break
            out.append(int(float(a)) if str(a).replace('.', '', 1).isdigit()
                       else str(a))
        return out

    # --- the recorded crew's SPAWN tiles: each STORE unit's position at hour 2
    # (the first hour every hand of the day stands somewhere; a hand hired
    # this day appears at its first hour instead — the row it acts from).
    # Keyed by STORE unit id: the fork's units[k+1] is store unit k.
    spawn_rows = con.sql(f"""
        SELECT unit, x, y FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
          AND step % 24 = 2
        ORDER BY unit
    """).fetchall()
    spawns: dict[int, tuple[int, int]] = {int(u): (int(x), int(y))
                                          for u, x, y in spawn_rows}
    n_store_units = con.sql(f"""
        SELECT max(unit) + 1 FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
    """).fetchone()[0]
    crew = [list(spawns.get(u, (4, 4))) for u in range(int(n_store_units))]

    units: list[list] = [[] for _ in range(len(hours) + 1)]
    units[0] = [["PASS"]] * 24
    for h, op, a1, a2 in farm_acts:
        units[0][int(h)] = _act(op, a1, a2)
    for u, h, op, a1, a2 in hand_acts:
        if not units[int(u) + 1]:
            units[int(u) + 1] = [["PASS"]] * 24
        units[int(u) + 1][int(h)] = _act(op, a1, a2)
    for i, hs in enumerate(hours):        # a hand acts from its first hour
        for h in range(min(int(hs), 24)):
            units[i + 1][h] = ["PASS"]
    # hands_steps records only MATERIALIZED hands; the hour-0 action rows are
    # planning rows (None ops) — the store already drops them, so a hand whose
    # first real op is at hour h carries PASS through hour h-1. The 'None' op
    # rows (hour 1, pre-hire turn of a hand hired that hour) map to PASS: the
    # engine accepted no op from a unit that did not exist in that turn.
    for k, unit in enumerate(units[1:], start=1):
        if not unit:
            unit[:] = [["PASS"]] * 24
        unit[:] = [["PASS"] if (len(op) == 1 and op[0] == 'None') else op
                   for op in unit]
    # per-unit positions per hour (hands_steps x/y; farmer from farm_steps):
    # lets the golden replay attribute each op to its acting unit without the
    # cell-merged chains
    unit_hours: list[list] = [[None] * 24 for _ in range(len(units))]
    for h, op, x, y in con.sql(f"""
        SELECT step % 24 AS hour, op, farmer_x, farmer_y
        FROM '{ROOT}/{dump}/farm_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        ORDER BY hour
    """).fetchall():
        unit_hours[0][int(h)] = [int(h), int(x), int(y), str(op)]
    for u, h, x, y, op in con.sql(f"""
        SELECT unit, step % 24 AS hour, x, y, op
        FROM '{ROOT}/{dump}/hands_steps.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        ORDER BY unit, hour
    """).fetchall():
        unit_hours[int(u) + 1][int(h)] = [int(h), int(x), int(y),
                                          str(op) if op is not None else None]
    market: dict[int, list] = defaultdict(list)
    for h, idx, op, item, qty in con.sql(f"""
        SELECT step % 24 AS hour, idx, op, item, qty FROM '{ROOT}/{dump}/market_orders.parquet'
        WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
        ORDER BY hour, idx
    """).fetchall():
        order = [str(op)]
        if item is not None and str(item) not in ('nan', 'None'):
            order.append(str(item))
            if qty is not None and str(qty) not in ('nan', 'None'):
                order.append(int(float(qty)))
        market[int(h)].append(order)

    # --- the hour-0 snapshot: board (EMPTY -> None, absent -> LOCKED), shed.
    step0 = day * 24
    board_rows = con.sql(f"""
        SELECT * FROM (
          SELECT *, row_number() OVER (PARTITION BY x, y ORDER BY step DESC) rn
          FROM '{ROOT}/{dump}/tiles_delta.parquet'
          WHERE episode_id={ep} AND player={pl} AND step <= {step0}
        ) WHERE rn = 1
    """).df()
    board = {}
    for r in board_rows.to_dict('records'):
        kind = r['kind']
        kind = None if kind is None or str(kind) in ('nan', 'None', 'NA', '<NA>') \
            else str(kind)
        x, y = int(r['x']), int(r['y'])
        if kind is None or kind == 'EMPTY':
            board[(x, y)] = None
            continue
        t = {'kind': kind}
        for f in ('crop', 'animal'):
            v = r.get(f)
            if v is not None and str(v) not in ('nan', 'None', 'NA'):
                t[f] = str(v)
        for f in ('yield_units', 'consecutive_unwatered', 'fertilized_until_day',
                  'planted_day', 'max_lifespan_step', 'placed_day',
                  'consecutive_unfed', 'pending_care_bonus'):
            v = _clean(r.get(f))
            if v is not None:
                t[f] = v
        for f in ('watered_today', 'fed_today', 'cared_today', 'fertilizer_available'):
            v = r.get(f)
            if v is not None and str(v) not in ('nan', 'None', 'NA'):
                t[f] = bool(v)
        board[(x, y)] = t

    # day-29 drop deadlines: the winner's own SELL hours per good
    sell_hours: dict[str, int] = {}
    if day == 29:
        sells = con.sql(f"""
            SELECT step % 24 AS hour, item FROM '{ROOT}/{dump}/market_orders.parquet'
            WHERE episode_id={ep} AND player={pl} AND step // 24 = {day}
              AND op = 'SELL' AND item IS NOT NULL
        """).fetchall()
        for h, item in sells:
            g = str(item)
            sell_hours[g] = min(sell_hours.get(g, 24), int(h))

    def good_of(e: str | None) -> str | None:
        if e in PRODUCTS:
            return e
        return ANIMAL_PRODUCT.get(e or '', None)

    chains_sorted = sorted(chains.items())
    drop_by = []
    for cell, _ops in chains_sorted:
        g = good_of(entity.get(cell))
        drop_by.append(sell_hours.get(g) if g else None)

    # --- shared market state at step0: the fork's prices must be the
    # recording's prices (city_steps is what makes a day fork replayable).
    town = con.sql(f"""
        SELECT * FROM '{ROOT}/{dump}/city_steps.parquet'
        WHERE episode_id={ep} AND step={step0}
    """).df()
    inv, prices, shops = {}, {}, ''
    if not town.empty:
        r = town.iloc[0].to_dict()
        for k, v in r.items():
            c = _clean(v)
            if k.startswith('inv_') and c is not None:
                inv[k[4:]] = c
            if k.startswith('price_') and c is not None:
                prices[k[6:]] = c
        shops = str(r.get('town_shops') or '')

    # the opponent's own market orders that day: replayed in the fork so the
    # shared town drains as it drained in the recording
    opp = 'true' if player == 'false' else 'false'
    opp_orders: dict[int, list] = defaultdict(list)
    for h, op, item, qty in con.sql(f"""
        SELECT step % 24 AS hour, op, item, qty FROM '{ROOT}/{dump}/market_orders.parquet'
        WHERE episode_id={ep} AND player={opp} AND step // 24 = {day}
        ORDER BY step, idx
    """).fetchall():
        order = [str(op)]
        if item is not None and str(item) not in ('nan', 'None'):
            order.append(str(item))
            if qty is not None and str(qty) not in ('nan', 'None'):
                order.append(int(float(qty)))
        opp_orders[int(h)].append(order)

    n_ops = sum(len(o) for _c, o in chains_sorted)
    return {
        'dump': dump, 'episode': ep, 'day': day,
        'agent': agent, 'wins': wins, 'games': games,
        'win_rate': round(win_rate, 3),
        'reward': reward, 'opponent_reward': loser,
        'quadrants': sorted(quads), 'quadrant_count': len(quads),
        'money': money,
        'hands': len(hours), 'hire_times': sorted(hours),
        'ops': n_ops,
        'available': available,
        'sell_hours': {g: h for g, h in sorted(sell_hours.items())},
        'drop_by': drop_by,
        'selfserve': selfserve,
        'chains': [[list(c), o, entity.get(c)] for c, o in chains_sorted],
        'op_hours': [[list(c), op_hours[c]] for c, _o in chains_sorted],
        'snapshot': {
            'board': [[x, y, t] for (x, y), t in sorted(board.items())],
            'shed': available,
            'shed_counts': shed_counts,
            'seeds': seeds,
            'money': money,
            'town_inv': inv,
            'town_prices': prices,
            'town_shops': shops,
            'hand_spawns': crew,
        },
        'opponent_market': {str(h): v for h, v in opp_orders.items()},
        'config': {'farmHandCostMult': hire_mult},
        'recorded': {'units': units, 'market': {str(h): v for h, v in market.items()},
                     'unit_hours': unit_hours},
    }


def selfserve_pairs(con) -> dict[tuple[str, int, int], set]:
    """(dump, episode, day) -> the adjacent self-serve ops on the WINNER side."""
    out: dict[tuple[str, int, int], set] = defaultdict(set)
    for d in DUMPS:
        rows = con.sql(f"""
            SELECT p.op, p.episode_id, p.day
            FROM read_parquet('/chista/tmp_selfserve/pairs_{d}.parquet') p
            JOIN '{ROOT}/{d}/episodes.parquet' e USING (episode_id)
            WHERE p.player = (e.reward1 > e.reward0)
        """).fetchall()
        for op, ep, day in rows:
            if op in ('FEED', 'FERTILIZE'):
                out[(d, int(ep), int(day))].add(op)
    return out


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    wt = winners_table(con)
    print(f'{len(wt)} agents with win rate > {WIN_RATE:.0%} over {MIN_GAMES}+ games')
    print(wt.to_string(index=False))

    pairs = selfserve_pairs(con)
    print(f'self-serve winner-side (day) pairs: {len(pairs)}')

    wr_of = {r.agent: (int(r.wins), int(r.games)) for r in wt.itertuples()}
    chosen: dict[tuple, dict] = {}
    # candidate day-level pool per agent: up to N won episodes each
    buckets: dict[str, list] = defaultdict(list)
    for agent in wt['agent']:
        wins, games = wr_of[agent]
        rate = wins / games
        for dump, ep, side, reward, loser in won_episodes(con, agent, 24):
            meta = day_meta(con, dump, ep, side)
            for day, m in meta.items():
                if m['n_quad'] == 0 or m['changes'] == 0:
                    continue
                buckets[f'tier{m["n_quad"]}'].append(
                    ((dump, ep, day), agent, side, reward, loser,
                     wins, games, rate, m))

    def add(key, item, bucket, selfserve='') -> bool:
        if key in chosen:
            return False
        dump, ep, day = key
        _k, agent, side, reward, loser, wins, games, rate, _m = item
        e = extract_winner_day(con, dump, ep, day, side, agent, reward, loser,
                               wins, games, rate, selfserve)
        if not e:
            return False
        e['bucket'] = bucket
        chosen[key] = e
        return True

    # 1) three days per quadrant tier
    exclude = set()
    for tier in ('tier1', 'tier2', 'tier3', 'tier4'):
        got = 0
        for key, *rest in buckets.get(tier, []):
            if got >= PER_TIER:
                break
            item = (key, *rest)
            if add(key, item, f'quadrant-{tier[4:]}'):
                exclude.add(key)
                got += 1

    # 2) self-serve mix days: adjacent pairs on the winner side
    got = 0
    for key, ops in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        if got >= PER_TIER:
            break
        dump, ep, day = key
        row = con.execute(f"""
            SELECT CASE WHEN reward1 > reward0 THEN agent1 ELSE agent0 END,
                   reward0, reward1,
                   CASE WHEN reward1 > reward0 THEN true ELSE false END
            FROM '{ROOT}/{dump}/episodes.parquet' WHERE episode_id = {ep}
        """).fetchone()
        if not row or row[0] not in wr_of:
            continue
        agent, r0, r1, side = row
        wins, games = wr_of[agent]
        item = (key, agent, bool(side), int(max(r0, r1)), int(min(r0, r1)),
                wins, games, wins / games, None)
        tag = '+'.join(sorted(ops))
        if add(key, item, 'self-serve-mix', selfserve=tag):
            exclude.add(key)
            got += 1

    # 3) day-29 days with SELL-backed drop deadlines
    got = 0
    for items in buckets.values():
        for item in items:
            if got >= PER_TIER:
                break
            key = item[0]
            if key in exclude or key[2] != 29:
                continue
            dump, ep, day = key
            _k, agent, side, reward, loser, wins, games, rate, _m = item
            e = extract_winner_day(con, dump, ep, day, side, agent, reward,
                                   loser, wins, games, rate)
            if not e or not any(d is not None for d in e['drop_by']):
                continue
            e['bucket'] = 'day-29-drops'
            chosen[key] = e
            exclude.add(key)
            got += 1

    # 4) the most-op days
    flat = [item for items in buckets.values() for item in items]
    got = 0
    for item in sorted(flat, key=lambda it: -it[8]['changes']):
        if got >= PER_TIER:
            break
        key = item[0]
        if key in exclude:
            continue
        if add(key, item, 'most-ops'):
            exclude.add(key)
            got += 1

    days = list(chosen.values())
    OUT.write_text(json.dumps(days, separators=(",", ":")))
    per_bucket = defaultdict(int)
    for e in days:
        per_bucket[e['bucket']] += 1
    print(f'{len(days)} days -> {OUT}')
    print(dict(per_bucket))
    for e in days:
        print(f"  {e['bucket']:18s} {e['dump']} ep{e['episode']} d{e['day']} "
              f"Q{e['quadrant_count']} hands={e['hands']} ops={e['ops']} "
              f"selfserve={e['selfserve'] or '-'} agent={e['agent']} "
              f"({e['wins']}/{e['games']})")


if __name__ == '__main__':
    main()
