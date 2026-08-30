#!/usr/bin/env python3
"""
Kaggriculture replay -> columnar (Parquet) converter for low-RAM machines.

Design goals
------------
* Never hold more than ONE episode (~28 MB JSON, ~300-400 MB parsed) in memory.
* Throw away everything that is redundant or reconstructable:
    - `specification` / schema description blocks (identical in every file)
    - player-1's copy of `farms` / `market` / `town` (byte-identical to player-0)
    - empty `info` dicts, `remainingOverageTime`
    - static tiles: only ~5% of tile cells change per step -> delta-encode them
* Output a handful of tidy tables as Parquet (zstd). DuckDB then queries these
  straight off disk, out-of-core, so 4 GB RAM is plenty.

Usage
-----
    pip install pyarrow            # (duckdb optional, for querying)
    python convert_replays.py  data/raw/Samples/   -o data/parquet/
    python convert_replays.py  data/raw/raw_zip/   -o data/parquet/

Accepts any mix of: directories of *.json, individual *.json, or *.zip archives.
Reads *.json members straight out of a .zip without extracting to disk.
"""
from __future__ import annotations
import argparse, json, os, re, sys, zipfile, glob
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------- schema helpers
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

TILE_FIELDS = [  # union of PLANT + COOP/PASTURE dict keys we keep
    "kind", "crop", "animal", "yield_units", "watered_today",
    "consecutive_unwatered", "fertilized_until_day", "planted_day",
    "max_lifespan_step", "placed_day", "fed_today", "consecutive_unfed",
    "cared_today", "fertilizer_available", "pending_care_bonus",
]

CONFIG_KEYS = ["boardSize", "episodeSteps", "farmHandCostMult",
               "maxMarketOrdersPerTurn", "shedCapacity", "startingMoney",
               "townCenterSellInterval", "townShopSellInterval",
               "townShopUnlockInterval", "turnsPerDay", "weedSpawnChance"]


class TableWriter:
    """Buffered ParquetWriter: append dict rows, flush in row-group batches."""
    def __init__(self, path, schema, batch=200_000):
        self.path, self.schema, self.batch = path, schema, batch
        self._rows = []
        self._w = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, row):
        self._rows.append(row)
        if len(self._rows) >= self.batch:
            self._flush()

    def _flush(self):
        if not self._rows:
            return
        self._w.write_table(pa.Table.from_pylist(self._rows, schema=self.schema))
        self._rows.clear()

    def close(self):
        self._flush()
        self._w.close()


f64, i64, s, b = pa.float64(), pa.int64(), pa.string(), pa.bool_()


def _sch(*pairs):
    return pa.schema([pa.field(n, t) for n, t in pairs])


SCHEMAS = {
    "episodes": _sch(
        ("episode_id", i64), ("seed", i64), ("agent0", s), ("agent1", s),
        ("reward0", f64), ("reward1", f64), ("status0", s), ("status1", s),
        ("n_steps", i64), *[(k, f64) for k in CONFIG_KEYS]),
    "steps": _sch(
        ("episode_id", i64), ("step", i64), ("day", i64), ("hour", i64),
        ("town_shops", s),
        *[(f"price_{k}", f64) for k in PRODUCTS],
        *[(f"inv_{k}", f64) for k in PRODUCTS]),
    "farm_steps": _sch(
        ("episode_id", i64), ("step", i64), ("player", i64), ("money", f64),
        ("farmer_x", i64), ("farmer_y", i64), ("n_hands", i64),
        ("hires_today", i64), ("quadrants", s)),
    "private_steps": _sch(
        ("episode_id", i64), ("step", i64), ("player", i64),
        *[(f"shed_{k}", f64) for k in PRODUCTS + ["GOOSE", "COW", "SHEEP"]],
        *[(f"seed_{k}", f64) for k in
          ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]]),
    "actions": _sch(
        ("episode_id", i64), ("step", i64), ("player", i64), ("unit", i64),
        ("op", s), ("arg1", s), ("arg2", s)),
    "market_orders": _sch(
        ("episode_id", i64), ("step", i64), ("player", i64), ("idx", i64),
        ("op", s), ("item", s), ("qty", f64)),
    "tiles_delta": _sch(
        ("episode_id", i64), ("step", i64), ("player", i64),
        ("x", i64), ("y", i64),
        ("kind", s), ("crop", s), ("animal", s), ("yield_units", f64),
        ("watered_today", b), ("consecutive_unwatered", f64),
        ("fertilized_until_day", f64), ("planted_day", f64),
        ("max_lifespan_step", f64), ("placed_day", f64), ("fed_today", b),
        ("consecutive_unfed", f64), ("cared_today", b),
        ("fertilizer_available", b), ("pending_care_bonus", f64)),
}


def tile_row(t):
    """Flatten a tile value (None / 'LOCKED' / dict) to a comparable tuple + dict."""
    if t is None:
        d = {"kind": "EMPTY"}
    elif t == "LOCKED":
        d = {"kind": "LOCKED"}
    else:
        d = {k: t.get(k) for k in TILE_FIELDS if k in t}
        d["kind"] = t.get("kind")
    return d


def convert_episode(raw, W):
    ep = raw["info"].get("EpisodeId") or raw.get("id")
    agents = [a.get("Name") for a in raw["info"].get("Agents", [{}, {}])]
    cfg = raw.get("configuration", {})
    rewards = raw.get("rewards", [None, None])
    statuses = raw.get("statuses", [None, None])
    steps = raw["steps"]

    W["episodes"].add({
        "episode_id": ep,
        "seed": raw["info"].get("seed"),
        "agent0": agents[0], "agent1": agents[1] if len(agents) > 1 else None,
        "reward0": rewards[0], "reward1": rewards[1],
        "status0": statuses[0], "status1": statuses[1],
        "n_steps": len(steps),
        **{k: cfg.get(k) for k in CONFIG_KEYS},
    })

    prev_tiles = {0: {}, 1: {}}  # (x,y) -> flattened dict, per player
    for i, st in enumerate(steps):
        p0 = st[0]
        obs = p0["observation"]                 # shared state lives here
        farms = obs["farms"]
        mkt = obs.get("market", {})
        inv, pr = mkt.get("inventory", {}), mkt.get("prices", {})

        W["steps"].add({
            "episode_id": ep, "step": i,
            "day": obs.get("day"), "hour": obs.get("hour"),
            "town_shops": ",".join(obs.get("town", {}).get("unlocked_shops", [])),
            **{f"price_{k}": pr.get(k) for k in PRODUCTS},
            **{f"inv_{k}": inv.get(k) for k in PRODUCTS},
        })

        for pl in (0, 1):
            entry = st[pl]
            f = farms[pl]
            W["farm_steps"].add({
                "episode_id": ep, "step": i, "player": pl,
                "money": f.get("money"),
                "farmer_x": f["farmer"][0], "farmer_y": f["farmer"][1],
                "n_hands": len(f.get("hands", [])),
                "hires_today": f.get("hires_today"),
                "quadrants": ",".join(f.get("unlocked_quadrants", [])),
            })

            # private (per player, differs) -> shed + seeds
            prv = entry.get("observation", {}).get("private", {})
            shed, seeds = prv.get("shed", {}), prv.get("seeds", {})
            if shed or seeds:
                W["private_steps"].add({
                    "episode_id": ep, "step": i, "player": pl,
                    **{f"shed_{k}": shed.get(k) for k in PRODUCTS
                       + ["GOOSE", "COW", "SHEEP"]},
                    **{f"seed_{k}": seeds.get(k) for k in
                       ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]},
                })

            # actions: farmer (unit -1) + hands (0..n)
            act = entry.get("action") or {}
            units = [("-1", act.get("farmer"))] + \
                    list(enumerate(act.get("hands") or []))
            for u, op in units:
                if not op:
                    continue
                W["actions"].add({
                    "episode_id": ep, "step": i, "player": pl, "unit": int(u),
                    "op": op[0],
                    "arg1": str(op[1]) if len(op) > 1 else None,
                    "arg2": str(op[2]) if len(op) > 2 else None,
                })
            for j, mo in enumerate(act.get("market") or []):
                W["market_orders"].add({
                    "episode_id": ep, "step": i, "player": pl, "idx": j,
                    "op": mo[0],
                    "item": str(mo[1]) if len(mo) > 1 else None,
                    "qty": mo[2] if len(mo) > 2 else None,
                })

            # tiles: delta-encode (emit only changed cells; kind='EMPTY' etc.)
            tiles = f["tiles"]
            cur = {}
            for y, row in enumerate(tiles):
                for x, t in enumerate(row):
                    d = tile_row(t)
                    cur[(x, y)] = d
                    if prev_tiles[pl].get((x, y)) != d:
                        row = {"episode_id": ep, "step": i, "player": pl,
                               "x": x, "y": y}
                        for k in TILE_FIELDS:
                            v = d.get(k)
                            if v is not None and k in ("kind", "crop", "animal"):
                                v = str(v)
                            row[k] = v
                        W["tiles_delta"].add(row)
            prev_tiles[pl] = cur


def _date_from_source(src):
    """Pull a YYYY-MM-DD tag from a path like 'Kaggriculture Episodes 2026-08-15.zip'."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(src))
    return m.group(1) if m else None


def _iter_items(paths, skip_tags=frozenset()):
    """Lazily yield (date_tag, src_name, blob) ONE json at a time.

    Memory contract: exactly one compressed blob (~tens of MB) is alive at any
    moment.  Each zip is opened, its members streamed out, and the blob is
    released as soon as the caller advances the generator.  Nothing is
    accumulated in a list anywhere.
    """
    for p in paths:
        if os.path.isdir(p):
            yield from _iter_items(
                sorted(glob.glob(os.path.join(p, "*.zip")))
                + sorted(glob.glob(os.path.join(p, "*.json"))), skip_tags)
        elif p.endswith(".zip"):
            tag = _date_from_source(p) or "unknown"
            if tag in skip_tags:          # group already converted -> don't read
                continue
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.endswith(".json"):
                        with z.open(name) as fh:
                            yield tag, f"{os.path.basename(p)}::{name}", fh.read()
        elif p.endswith(".json"):
            with open(p, "rb") as fh:
                yield _date_from_source(p), p, fh.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="dirs / *.json / *.zip")
    ap.add_argument("-o", "--out", default="data/parquet", help="output root dir")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing per-zip output dirs")
    args = ap.parse_args()

    # decide which date-groups to skip BEFORE reading any bytes
    skip_tags = set()
    if not args.force:
        for p in args.paths:
            zips = sorted(glob.glob(os.path.join(p, "*.zip"))) if os.path.isdir(p) \
                else ([p] if p.endswith(".zip") else [])
            for zp in zips:
                tag = _date_from_source(zp) or "unknown"
                out_dir = os.path.join(args.out, tag)
                if os.path.isdir(out_dir) and any(
                        fn.endswith(".parquet") for fn in os.listdir(out_dir)):
                    skip_tags.add(tag)

    # one writer per date-group; switch lazily as the stream crosses zips
    W, cur_tag, out_dir, n = None, None, None, 0
    for tag, src, blob in _iter_items(args.paths, skip_tags):
        if tag != cur_tag:                      # entered a new group
            if W:
                for w in W.values():
                    w.close()
                print(f"done: [{cur_tag or 'loose'}] {n} episodes -> {out_dir}/",
                      file=sys.stderr)
            cur_tag, n = tag, 0
            out_dir = args.out if tag is None else os.path.join(args.out, tag)
            os.makedirs(out_dir, exist_ok=True)
            W = {name: TableWriter(os.path.join(out_dir, f"{name}.parquet"), sch)
                 for name, sch in SCHEMAS.items()}
        try:
            convert_episode(json.loads(blob), W)
            n += 1
            if n % 25 == 0:
                print(f"  [{tag or 'loose'}] {n} episodes...", file=sys.stderr)
        except Exception as e:  # noqa
            print(f"!! {src}: {e}", file=sys.stderr)
        del blob                                # release this json immediately
    if W:
        for w in W.values():
            w.close()
        print(f"done: [{cur_tag or 'loose'}] {n} episodes -> {out_dir}/",
              file=sys.stderr)


if __name__ == "__main__":
    main()
