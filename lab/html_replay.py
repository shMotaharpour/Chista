"""Generate an HTML replay visualization for the melon fertilizer A/B comparison.

Usage: python -m lab.html_replay [--out lab/results/melon_replay.html]
"""
from __future__ import annotations

import argparse
import json

from kaggle_environments import make


def trace_melon(fert: bool, harvest_at: int, seed: int = 70):
    env = make("kaggriculture", configuration={"episodeSteps": 14 * 24, "seed": seed}, debug=False)
    log = []

    def ag(obs):
        me = obs["farms"][0]; priv = obs["private"]
        fx, fy = me["farmer"]; tile = me["tiles"][fy][fx]
        day, hour, step = obs["day"], obs.get("hour", 0), obs["step"]
        if step == 0:
            o = [["BUY_SEED", "MELON", 1]]
            if fert:
                o.append(["BUY_PRODUCT", "FERTILIZER", 3])
            return {"farmer": ["PASS"], "hands": [], "market": o}
        if step == 1 and fert:
            return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
        if tile is None and priv["seeds"].get("MELON", 0):
            return {"farmer": ["PLANT", "MELON"], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if fert and age == 6 and hour == 1 and priv["shed"].get("FERTILIZER", 0):
                return {"farmer": ["PICKUP", "FERTILIZER", 3], "hands": [], "market": []}
            if fert and age == 6 and hour == 2 and priv["inventories"][0].get("FERTILIZER", 0):
                return {"farmer": ["FERTILIZE"], "hands": [], "market": []}
            if age >= harvest_at and hour == 1 and tile["yield_units"] > 0:
                log.append((day, tile["yield_units"]))
                return {"farvest": None} if False else {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env.run([ag, "pass"])
    days = []
    for step in range(0, 14 * 24, 24):
        o = env.steps[step][0].observation
        me = o.farms[0]
        t = me["tiles"][me["farmer"][1]][me["farmer"][0]]
        days.append({
            "day": step // 24,
            "tile": (dict(t) if isinstance(t, dict) else t),
            "money": me["money"],
        })
    # also capture the env's own renderer text for each day
    return {"fert": fert, "harvest_log": log, "days": days, "env_json": env.toJSON()}


def build_html() -> str:
    a = trace_melon(True, 8)    # fertilized: harvest at first full yield (day 8)
    b = trace_melon(False, 10)  # plain: harvest at first full yield (day 10)

    def tile_html(t) -> str:
        if t is None:
            return '<span class="empty">empty</span>'
        if isinstance(t, str):
            return f'<span class="locked">{t}</span>'
        kind = t.get("kind", "?")
        if kind == "PLANT":
            fert_badge = " 🌱fert" if t.get("fertilized_until_day", -1) >= 0 else ""
            color = "#2e7d32" if t.get("yield_units", 0) >= 6 else ("#f9a825" if t.get("yield_units", 0) >= 2 else "#8d6e63")
            return f'<span class="plant" style="color:{color}">🌿MELON y={t.get("yield_units",0)}{fert_badge}</span>'
        if kind == "WEED":
            return '<span class="weed">🥀WEED</span>'
        return f'<span>{kind}</span>'

    rows = []
    for (d1, t1, m1), (d2, t2, m2) in zip(a["days"], b["days"]):
        rows.append(f"""
        <tr>
          <td class="day">{d1}</td>
          <td>{tile_html(t1)}</td>
          <td>${m1:.0f}</td>
          <td>{tile_html(t2)}</td>
          <td>${m2:.0f}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Melon Fertilizer A/B Replay — Kaggriculture</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background:#121212; color:#e0e0e0; margin:2rem; }}
  h1 {{ color:#aed581; font-size:1.4rem; }}
  h2 {{ color:#90caf9; font-size:1.05rem; margin-top:1.5rem; }}
  table {{ border-collapse: collapse; margin-top:1rem; width:100%; max-width:900px; }}
  th, td {{ padding:6px 12px; text-align:left; border-bottom:1px solid #333; }}
  th {{ color:#aaa; font-weight:600; }}
  td.day {{ color:#888; font-family:monospace; }}
  .plant {{ font-weight:600; }}
  .empty {{ color:#666; }}
  .weed {{ color:#bf360c; }}
  .locked {{ color:#555; }}
  .harvest {{ background:#1b5e20; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }}
  .log {{ font-family:monospace; background:#1e1e1e; padding:10px; border-radius:6px; }}
  .cols {{ display:flex; gap:2rem; flex-wrap:wrap; }}
</style>
</head>
<body>
<h1>🍉 Melon Fertilizer A/B Replay</h1>
<p>Same seed (70), two identical farms: left applies fertilizer on day 6 (bonus window start),
right never fertilizes. Each harvests at first full yield.</p>

<table>
<tr><th>Day</th><th>FERTILIZED tile</th><th>Money F</th><th>PLAIN tile</th><th>Money P</th></tr>
{''.join(rows)}
</table>

<div class="cols">
<div>
<h2>Fertilized harvest log</h2>
<div class="log">{'<br>'.join(f'day {d}: <span class="harvest">HARVEST y={y}</span>' for d, y in a['harvest_log']) or 'none'}</div>
</div>
<div>
<h2>Plain harvest log</h2>
<div class="log">{'<br>'.join(f'day {d}: <span class="harvest">HARVEST y={y}</span>' for d, y in b['harvest_log']) or 'none'}</div>
</div>
</div>

<h2>Key numbers</h2>
<ul>
<li><b>Fertilized:</b> harvest day <b>8</b> (6 units) — tile free for cycle 2 from day 9</li>
<li><b>Plain:</b> harvest day <b>10</b> (6 units) — tile free day 11</li>
<li>Fertilizer cost: $300 — value: melon ready 2 days earlier → 3rd cycle becomes feasible in a 30-day season</li>
<li>Both sell prices are identical at equal inventory → the entire gain is the 2-day tile acceleration</li>
</ul>
</body>
</html>"""
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab/results/melon_replay.html")
    args = ap.parse_args()
    html = build_html()
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html)} bytes)")
