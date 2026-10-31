"""
actions.py
------------
A concrete, explainable suggestion for every overstock product x city on the
serving date, in place of a fixed "Reduce orders". An estimate: stock is
simulated. The risk rule itself (inventory.classify_risk) is unchanged.

  sell window   the days the stock can still sell in: the shelf life for
                perishables (shelf life <= the delivery cycle: dairy, sweets),
                the overstock threshold (LEAD_TIME_DAYS + 2 + shelf life, as in
                classify_risk) for the rest
  excess        max(0, stock - forecast_daily x sell window)

Then, in order:

  a) transfer   the same SKU is short in another city of the scope: at
                stockout risk (short by its reorder_qty) or "Order now" in the
                next festival's readiness (festival_readiness.city_readiness,
                short by its shortfall; the larger of the two if both).
                "Move N units to <city>", N = min(excess here, shortfall there).
                Surpluses are paired largest excess first, each with the city
                that has the largest shortfall still open; a city's shortfall
                shrinks as transfers fill it. Never more than the excess.
  b) discount   the smallest discount d (5% steps, up to 40%) at which the
                stock sells within the sell window, with demand x
                (1 - d) ** -PRICE_ELASTICITY (the elasticity the synthetic
                demand and the analog price adjustment use).
                "20% off for 4 days": the days the discounted rate needs to
                sell the stock.
  c) reduce     otherwise (40% is not enough: "Discount alone won't clear it"):
                "Reduce next order by N units", N = the excess.
"""

import json
import math

import numpy as np
import pandas as pd

from config import DELIVERY_CYCLE_DAYS, LEAD_TIME_DAYS, PRICE_ELASTICITY, SAFETY_STOCK_DAYS, SHELF_LIFE_DAYS

LABEL = "Estimate · simulated stock"
DISCOUNT_STEP = 0.05
MAX_DISCOUNT = 0.40
NO_DISCOUNT = "Discount alone won't clear it"
TYPES = ["transfer", "discount", "reduce_order"]
COLUMNS = ["city", "sku_id", "excess_units", "sell_window_days", "action_type", "action", "action_detail",
           "transfer_to", "transfer_units", "discount_pct", "discount_days", "reduce_units"]
_CACHE = {}


def sell_window(category):
    """Days stock can still sell in: the shelf life (perishables) or the overstock threshold."""
    shelf_life = SHELF_LIFE_DAYS[category]
    # classify_risk's overstock threshold is horizon + 2 + shelf life; the 2 is SAFETY_STOCK_DAYS
    return shelf_life if shelf_life <= DELIVERY_CYCLE_DAYS else LEAD_TIME_DAYS + SAFETY_STOCK_DAYS + shelf_life


def excess_units(stock, daily, window):
    return max(0.0, float(stock) - float(daily) * window)


def demand_lift(discount, elasticity=PRICE_ELASTICITY):
    """Demand multiplier at a price cut of `discount` (0.2 = 20% off)."""
    return (1 - discount) ** -elasticity


def discount_plan(stock, daily, window, elasticity=PRICE_ELASTICITY):
    """(discount, days) - the smallest discount that sells `stock` within `window` days - or None."""
    if daily <= 0:
        return None
    steps = int(round(MAX_DISCOUNT / DISCOUNT_STEP))
    for step in range(1, steps + 1):
        discount = round(step * DISCOUNT_STEP, 2)
        rate = daily * demand_lift(discount, elasticity)
        if rate * window >= stock - 1e-9:
            return discount, max(1, math.ceil(stock / rate - 1e-9))
    return None


def pair_transfers(surplus, short):
    """
    surplus: rows with sku_id, city, excess; short: rows with sku_id, city,
    shortfall and reason. Returns {surplus index: (city, units, reason)}: per
    SKU, largest excess first, each to the other city with the largest
    shortfall still open; units = min(excess, open shortfall).
    """
    plan = {}
    for sku, rows in surplus[surplus["excess"] > 0].groupby("sku_id", sort=False):
        needs = short[(short["sku_id"] == sku) & (short["shortfall"] > 0)]
        open_need = {city: [float(need), reason] for city, need, reason in needs[["city", "shortfall", "reason"]].itertuples(index=False)}
        for index, row in rows.sort_values(["excess", "city"], ascending=[False, True]).iterrows():
            options = sorted((-need, city) for city, (need, _) in open_need.items() if city != row["city"] and need > 0)
            if not options:
                continue
            need, city = -options[0][0], options[0][1]  # largest open shortfall; ties by city name
            units = math.floor(min(row["excess"], need))
            if units <= 0:
                continue
            plan[index] = (city, units, open_need[city][1])
            open_need[city][0] -= units
    return plan


def shortfalls(rows, readiness=None, festival=""):
    """Per city x SKU: stockout reorder_qty and "Order now" readiness shortfall, the larger one with its reason."""
    frames = [rows.loc[rows["risk_status"] == "stockout", ["city", "sku_id", "reorder_qty"]]
              .rename(columns={"reorder_qty": "shortfall"}).assign(reason="stockout risk")]
    if readiness is not None and not readiness.empty:
        now = readiness[readiness["status"] == "Order now"]
        frames.append(now[["city", "sku_id", "shortfall"]].assign(reason=f"{festival} “Order now”".strip()))
    short = pd.concat(frames, ignore_index=True)
    short = short.sort_values("shortfall", ascending=False, kind="stable").drop_duplicates(["city", "sku_id"])
    return short.reset_index(drop=True)


def overstock_actions(rows, short, category_col="category"):
    """One action per overstock row of `rows` (needs city, sku_id, opening_stock, forecast_daily, risk_status)."""
    over = rows[rows["risk_status"] == "overstock"].copy()
    if over.empty:
        return pd.DataFrame(columns=COLUMNS)
    over["sell_window_days"] = over[category_col].map(sell_window)
    over["excess"] = [excess_units(stock, daily, window) for stock, daily, window
                      in over[["opening_stock", "forecast_daily", "sell_window_days"]].itertuples(index=False)]
    transfers = pair_transfers(over, short)
    out = []
    for index, row in over.iterrows():
        excess = row["excess"]
        record = {"city": row["city"], "sku_id": row["sku_id"], "excess_units": math.ceil(excess - 1e-9),
                  "sell_window_days": int(row["sell_window_days"]), "transfer_to": None, "transfer_units": None,
                  "discount_pct": None, "discount_days": None, "reduce_units": None}
        window = int(row["sell_window_days"])
        sells = f"{row['forecast_daily']:.1f}/day forecast sells {row['forecast_daily'] * window:,.0f} in {window} days"
        if index in transfers:
            city, units, reason = transfers[index]
            record.update(action_type="transfer", action=f"Move {units:,} units to {city}", transfer_to=city,
                          transfer_units=units,
                          action_detail=f"{units:,} of {record['excess_units']:,} excess units; {city}: {reason}")
        else:
            plan = discount_plan(row["opening_stock"], row["forecast_daily"], window)
            if plan:
                discount, days = plan
                record.update(action_type="discount", action=f"{discount:.0%} off for {days} day{'s' * (days != 1)}",
                              discount_pct=round(discount * 100), discount_days=days,
                              action_detail=f"{record['excess_units']:,} excess units ({sells}); demand ×{demand_lift(discount):.2f} at {discount:.0%} off")
            else:
                units = record["excess_units"]
                record.update(action_type="reduce_order", action=f"Reduce next order by {units:,} units", reduce_units=units,
                              action_detail=f"{NO_DISCOUNT}: {units:,} excess units ({sells}); 40% off lifts demand only ×{demand_lift(MAX_DISCOUNT):.2f}")
        out.append(record)
    return pd.DataFrame(out, columns=COLUMNS)


def festival_readiness_rows(cities):
    """ "Order now" readiness for the default festival in every city, and its name; (None, "") when unavailable."""
    import festival_readiness

    try:
        tables, festival = [], ""
        for city in cities:
            meta, table = festival_readiness.city_readiness(city)
            tables.append(table)
            festival = meta["festival"]
        return pd.concat(tables, ignore_index=True), festival
    except (FileNotFoundError, LookupError, ValueError):
        return None, ""


def scope_actions(scope):
    """Actions for a stockout_view scope ("model" or "regional"), built once and kept in memory."""
    import stockout_view

    rows, _ = stockout_view.raw_rows(scope)
    if scope not in _CACHE or _CACHE[scope][0] is not rows:  # rebuilt when stockout_view's table is
        category_col = "model_category" if "model_category" in rows else "category"
        readiness, festival = (None, "")
        if (rows["risk_status"] == "overstock").any():
            readiness, festival = festival_readiness_rows(sorted(rows["city"].unique()))
        _CACHE[scope] = rows, overstock_actions(rows, shortfalls(rows, readiness, festival), category_col)
    return _CACHE[scope][1]


def lookup(scope):
    """{(city, sku_id): action record} for a scope, plain Python values (JSON-ready)."""
    records = json.loads(scope_actions(scope).to_json(orient="records"))  # NaN -> null, numpy -> plain types
    return {(row["city"], row["sku_id"]): {**row, "label": LABEL} for row in records}


def clear_cache():
    _CACHE.clear()


def summary(scope):
    """Action counts per city and type."""
    table = scope_actions(scope)
    return table.groupby("city")["action_type"].value_counts().unstack(fill_value=0).reindex(columns=TYPES, fill_value=0)


if __name__ == "__main__":
    for name in ("model", "regional"):
        print(f"\n{name}:\n{summary(name).to_string()}")
