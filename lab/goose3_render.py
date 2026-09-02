"""HTML render for the 3-goose care/feeding demo (goose3_demo).

Builds a self-contained HTML page with a per-day summary grid of the three
goose coops + action log, from a live 6-day run.

Usage: python -m lab.goose3_render [--out lab/results/goose3_replay.html]
"""
from __future__ import annotations

import argparse
import json

from lab.goose3_demo import run_goose3, DAYS


def collect(env, days=DAYS + 1):
    days_out = []
    actions = []
    for st, srow in enumerate(env.steps):
        a = srow[0].action or {}
        f = a.get("farmer", ["PASS"])
        m = a.get("market", [])
        if f != ["PASS"] or m:
            actions.append((st // 24, st % 24, " ".join(map(str, f)),
                            ", ".join(" ".join(map(str, o)) for o in m)))
    for d in range(days):
        step = min((d + 1) * 24 - 1, len(env.steps) - 1)
        o = env.steps[step][0].observation
        me = o.farms[0]
        coops = sorted([(x, y, t) for y, row in enumerate(me["tiles"])
                        for x, t in enumerate(row)
                        if isinstance(t, dict) and t.get("kind") == "COOP" and t.get("animal")],
                       key=lambda c: c[0])
        geese = [{"yield": t.get("yield_units", 0),
                  "bank": t.get("pending_care_bonus", 0),
                  "fed": bool(t.get("fed_today")),
                  "cared": bool(t.get("cared_today"))}
                 for _, _, t in coops]
        days_out.append({"day": d, "money": me["money"],
                         "eggs": o.private["shed"].get("EGG", 0), "geese": geese})
    return days_out, actions


def build_html(env, days=DAYS + 1):
    day_rows, actions = collect(env, days)

    def goose_cell(g):
        badges = []
        if g["fed"]:
            badges.append('<span class="ok">fed</span>')
        else:
            badges.append('<span class="no">unfed</span>')
        if g["cared"]:
            badges.append('<span class="ok">cared</span>')
        badges.append(f'yield {g["yield"]}')
        badges.append(f'bank {g["bank"]}')
        return " ".join(badges)

    rows = []
    for d in day_rows:
        cells = "".join(f"<td>{goose_cell(g)}</td>" for g in d["geese"])
        rows.append(f'<tr><td class="day">{d["day"]}</td><td>${d["money"]:.0f}</td>'
                    f'<td>{d["eggs"]}</td>{cells}</tr>')

    act_rows = "".join(
        f'<tr><td class="day">d{d} h{h}</td><td>{op}</td><td>{mk}</td></tr>'
        for d, h, op, mk in actions[:120])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3-Goose Care Schedule — 6-day replay</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background:#121212; color:#e0e0e0; margin:2rem; }}
  h1 {{ color:#aed581; font-size:1.4rem; }}
  h2 {{ color:#90caf9; font-size:1.05rem; margin-top:1.5rem; }}
  table {{ border-collapse: collapse; margin-top:1rem; }}
  th, td {{ padding:6px 12px; text-align:left; border-bottom:1px solid #333; }}
  th {{ color:#aaa; font-weight:600; }}
  td.day {{ color:#888; font-family:monospace; }}
  .ok {{ background:#1b5e20; color:#fff; padding:1px 7px; border-radius:4px; font-size:.8rem; }}
  .no {{ background:#5e1b1b; color:#fff; padding:1px 7px; border-radius:4px; font-size:.8rem; }}
  .log {{ font-family:monospace; font-size:.85rem; background:#1e1e1e; padding:10px; border-radius:6px; max-height:420px; overflow:auto; }}
  .cols {{ display:flex; gap:3rem; flex-wrap:wrap; align-items:flex-start; }}
</style>
</head>
<body>
<h1>🪿 3-Goose Care Schedule Replay (6 days)</h1>
<p>
<b>Goose 1</b>: feed + CARE daily from day 0 ·
<b>Goose 2</b>: feed + CARE daily from day 1 ·
<b>Goose 3</b>: FEED only, every other day from day 1 (no CARE).
Coops at (2,4), (3,4), (4,4). Eggs harvested whenever present.
</p>

<div class="cols">
<div>
<h2>End-of-day state</h2>
<table>
<tr><th>Day</th><th>Money</th><th>Eggs (shed)</th><th>Goose 1</th><th>Goose 2</th><th>Goose 3</th></tr>
{''.join(rows)}
</table>
</div>
<div>
<h2>Action log (non-PASS)</h2>
<div class="log"><table>{act_rows}</table></div>
</div>
</div>

<h2>Verified engine semantics</h2>
<ul>
<li>Base production (<code>yield += 1</code>) is <b>unconditional</b> — feeding gates only the care bank (source line ~828)</li>
<li>First eggs at end-of-day-4 refresh (placed_day 0 + first_yield_day 4); goose 1/2 payout = 1 base + banked care</li>
<li>Goose 3 (alternating feed): still produced its base egg each production day — bank stayed 0</li>
<li>Total shed eggs by day 5: <b>{day_rows[-1]['eggs']}</b></li>
</ul>
</body>
</html>"""
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab/results/goose3_replay.html")
    ap.add_argument("--days", type=int, default=DAYS + 1)
    args = ap.parse_args()
    env = run_goose3(days=args.days)
    html = build_html(env, args.days)
    with open(args.out, "w") as f:
        f.write(html)
    with open("lab/results/goose3_replay.json", "w") as f:
        json.dump(env.toJSON(), f)
    print(f"html -> {args.out}")
    print("json -> lab/results/goose3_replay.json")
