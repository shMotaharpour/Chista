"""Sell-impact analysis: how dumping n units moves each product's price.

Usage: python -m lab.sell_impact
"""
from __future__ import annotations

from lab.prices import price, sell_revenue, MARKET_I0, MARKET_PARAMS, PRODUCTS

def analyze(item: str) -> dict:
    T = MARKET_PARAMS[item]["T"]
    base = MARKET_PARAMS[item]["base"]
    rows = []
    for n in (1, 10, 50, 100, T, 2 * T):
        rev = sell_revenue(item, n)
        avg = rev / n
        rows.append({"n": n, "revenue": rev, "avg_price": avg, "loss_vs_base_pct": 100 * (1 - avg / base)})
    return {"item": item, "base": base, "T": T, "rows": rows}


def optimal_daily_volume(item: str) -> dict:
    """Max units/day selling at >= 90% of base price (starting from I0)."""
    target = int(0.9 * MARKET_PARAMS[item]["base"])
    n = 0
    while price(item, MARKET_I0 + n) >= target:
        n += 1
        if n > 5000:
            break
    return {"item": item, "units_per_day_at_90pct": n}


if __name__ == "__main__":
    print("=== Sell impact (dump n units from I0, one-at-a-time lockstep) ===")
    for item in PRODUCTS:
        r = analyze(item)
        print(f"\n{item} (base ${r['base']}, T={r['T']}):")
        for row in r["rows"]:
            print(f"  n={row['n']:>5}  revenue=${row['revenue']:>7}  avg=${row['avg_price']:>6.2f}  "
                  f"loss vs base: {row['loss_vs_base_pct']:>5.1f}%")
        o = optimal_daily_volume(item)
        print(f"  → safe pace: {o['units_per_day_at_90pct']} units/day keeps price ≥ 90% of base")
