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
  shelf left    days the stock on hand stays sellable, counting today: it came
                with the last delivery, days_until_delivery - R days ago (R =
                the delivery cycle, as in festival_readiness), so
                days_until_delivery - R + shelf life

Then, in order:

  a) transfer   the same SKU is short in another city of the scope: at
                stockout risk (short by its reorder_qty) or "Order now" in the
                next festival's readiness (festival_readiness.city_readiness,
                short by its shortfall; the larger of the two if both).
                "Move N units to <city>", N = min(excess here, shortfall there).
                Surpluses are paired largest excess first, each with the city
                that has the largest shortfall still open; a city's shortfall
                shrinks as transfers fill it. Never more than the excess.
                Freshness: a shortfall only counts if the units are still
                within shelf life when that city needs them, after
                TRANSIT_DAYS in transit -- stockout: shelf left > TRANSIT_DAYS
                + that city's days until its next delivery; "Order now": still
                fresh on the first day of the festival week (shelf left > days
                from today to the week's start). Otherwise b), then c).
  b) discount   the smallest discount d (5% steps, up to 40%) at which the
                stock sells within the sell window, with demand x
                (1 - d) ** -PRICE_ELASTICITY (the elasticity the synthetic
                demand and the analog price adjustment use).
                "20% off for 4 days": the days the discounted rate needs to
                sell the stock.
  normal order  forecast_daily x (R + SAFETY_STOCK_DAYS): one delivery cycle of
                forecast demand plus 2 days, as the reorder rule
                (inventory.classify_risk, simulate_inventory) orders

  otherwise 40% is not enough: "Discount alone won't clear it", with the
  sales multiple the stock would need vs the ~1.4x that 40% off gives, and
  c) write-off  perishables (sell window = shelf life): the excess expires
                before it can sell, and the delivery cycle equals the shelf
                life, so skipping a delivery would empty the shelf. "Expected
                write-off ≈N units (expires before it can sell) · order ~X per
                delivery instead of ~Y from now on", X = the normal order
                (below), Y = recent_avg_order, the
                average delivery over the last RECENT_ORDER_DAYS days (today
                included; it includes any over-order). The detail starts with
                the hint "Mark down or donate before expiry" when N > 0.
  d) reduce     non-perishables, in whole delivery cycles of a normal order
                M: "Skip the next K orders (≈N units
                excess; a normal order is ~M)", K = floor(N / M); "Reduce next
                order by N units" when N is less than one normal order.
"""

import json
import math

import numpy as np
import pandas as pd

from config import DELIVERY_CYCLE_DAYS, LEAD_TIME_DAYS, PRICE_ELASTICITY, SAFETY_STOCK_DAYS, SHELF_LIFE_DAYS
from inventory import review_days_for

LABEL = "Estimate · simulated stock"
DISCOUNT_STEP = 0.05
MAX_DISCOUNT = 0.40
NO_DISCOUNT = "Discount alone won't clear it"
TRANSIT_DAYS = 1
FESTIVAL_WEEK_DAYS = 7  # festival_readiness.FESTIVAL_WEEK_DAYS
TYPES = ["transfer", "discount", "write_off", "reduce_order"]
WRITE_OFF_HINT = "Mark down or donate before expiry"
RECENT_ORDER_DAYS = 28  # build_bigbasket_regional_risk.WARMUP_DAYS
COLUMNS = ["city", "sku_id", "excess_units", "sell_window_days", "action_type", "action", "action_detail",
           "shelf_left_days", "transfer_to", "transfer_units", "discount_pct", "discount_days", "reduce_units",
           "normal_order", "skip_orders", "recent_avg_order"]
_CACHE = {}


def is_perishable(category):
    """Stock that expires within one delivery cycle (dairy, sweets): written off at the next delivery."""
    return SHELF_LIFE_DAYS[category] <= DELIVERY_CYCLE_DAYS


def sell_window(category):
    """Days stock can still sell in: the shelf life (perishables) or the overstock threshold."""
    shelf_life = SHELF_LIFE_DAYS[category]
    # classify_risk's overstock threshold is horizon + 2 + shelf life; the 2 is SAFETY_STOCK_DAYS
    return shelf_life if is_perishable(category) else LEAD_TIME_DAYS + SAFETY_STOCK_DAYS + shelf_life


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
    surplus: rows with sku_id, city, excess and shelf_left; short: rows with
    sku_id, city, shortfall, reason and need_by (the day offset the units must
    still be sellable on). Returns ({surplus index: (city, units, reason)},
    {surplus index: [(city, reason, need_by)] ruled out by freshness}): per
    SKU, largest excess first, each to the other city with the largest fresh
    shortfall still open; units = min(excess, open shortfall).
    """
    plan, blocked = {}, {}
    for sku, rows in surplus[surplus["excess"] > 0].groupby("sku_id", sort=False):
        needs = short[(short["sku_id"] == sku) & (short["shortfall"] > 0)]
        filled = {}  # units already sent to each city
        for index, row in rows.sort_values(["excess", "city"], ascending=[False, True]).iterrows():
            other = needs[needs["city"] != row["city"]]
            fresh = other[other["need_by"] < row["shelf_left"]]
            stale = other[other["need_by"] >= row["shelf_left"]]
            if not stale.empty:
                blocked[index] = list(stale[["city", "reason", "need_by"]].itertuples(index=False, name=None))
            options = []
            for city, group in fresh.groupby("city"):
                best = group.loc[group["shortfall"].idxmax()]  # the larger need if a city is short both ways
                open_units = float(best["shortfall"]) - filled.get(city, 0)
                if open_units > 0:
                    options.append((-open_units, city, best["reason"]))
            if not options:
                continue
            need, city, reason = min(options)  # largest open shortfall; ties by city name
            units = math.floor(min(row["excess"], -need))
            if units <= 0:
                continue
            plan[index] = (city, units, reason)
            filled[city] = filled.get(city, 0) + units
    return plan, blocked


def shortfalls(rows, readiness=None, festival="", week_start=None):
    """
    Short city x SKU rows (both, if a city is short both ways): stockout
    (reorder_qty, needed until its next delivery after transit) and "Order
    now" readiness (shortfall, needed from the festival week's first day,
    `week_start` days after today).
    """
    stockout = rows[rows["risk_status"] == "stockout"]
    frames = [pd.DataFrame({"city": stockout["city"], "sku_id": stockout["sku_id"], "shortfall": stockout["reorder_qty"],
                            "reason": "stockout risk", "need_by": TRANSIT_DAYS + stockout["days_until_delivery"]})]
    if readiness is not None and not readiness.empty and week_start is not None:
        now = readiness[readiness["status"] == "Order now"]
        frames.append(pd.DataFrame({"city": now["city"], "sku_id": now["sku_id"], "shortfall": now["shortfall"],
                                    "reason": f"{festival} “Order now”".strip(), "need_by": week_start}))
    return pd.concat(frames, ignore_index=True)


def multiple(value):
    return f"{value:.1f}" if value < 10 else f"{value:,.0f}"


def reduce_text(units, normal_order):
    """The reduce-order action in whole delivery cycles: (text, orders skipped)."""
    if normal_order <= 0 or units < normal_order:
        return f"Reduce next order by {units:,} units", 0
    skip = math.floor(units / normal_order)
    orders = "the next order" if skip == 1 else f"the next {skip} orders"
    return f"Skip {orders} (≈{units:,} units excess; a normal order is ~{normal_order:,.0f})", skip


def day_text(offset, counted=None):
    """A day offset from the stock-counting morning, as a date when that day is known ("2 Nov") or "day 3"."""
    if counted is None:
        return f"day {offset}"
    day = pd.Timestamp(counted) + pd.Timedelta(days=int(offset))
    return f"{day.day} {day:%b}"


def write_off_text(units, normal_order, recent_order=None):
    """The perishable action: expected write-off and the right order size from now on."""
    usual = "" if recent_order is None or pd.isna(recent_order) else f" instead of ~{recent_order:,.0f}"
    return (f"Expected write-off ≈{units:,} units (expires before it can sell) · "
            f"order ~{normal_order:,.0f} per delivery{usual} from now on")


def blocked_text(shelf_left, blocked, counted=None):
    """e.g. 'Transfer ruled out: stock here is sellable until 1 Nov; Delhi (Diwali “Order now”) needs it sellable on 2 Nov'."""
    needs = "; ".join(f"{city} ({reason}) needs it sellable on {day_text(need_by, counted)}" for city, reason, need_by in blocked)
    return f"Transfer ruled out: stock here is sellable until {day_text(int(shelf_left) - 1, counted)}; {needs}"


def overstock_actions(rows, short, category_col="category", counted=None):
    """
    One action per overstock row of `rows` (needs city, sku_id, opening_stock,
    forecast_daily, days_until_delivery, risk_status); `counted` (the
    stock-counting day) turns day offsets into dates in the text.
    """
    over = rows[rows["risk_status"] == "overstock"].copy()
    if over.empty:
        return pd.DataFrame(columns=COLUMNS)
    over["sell_window_days"] = over[category_col].map(sell_window)
    over["excess"] = [excess_units(stock, daily, window) for stock, daily, window
                      in over[["opening_stock", "forecast_daily", "sell_window_days"]].itertuples(index=False)]
    cycle = review_days_for(over[category_col])
    over["normal_order"] = over["forecast_daily"] * (cycle + SAFETY_STOCK_DAYS)  # the reorder rule's order size
    over["shelf_left"] = over["days_until_delivery"] - cycle + over[category_col].map(SHELF_LIFE_DAYS)
    transfers, blocked = pair_transfers(over, short)
    out = []
    for index, row in over.iterrows():
        excess = row["excess"]
        record = {"city": row["city"], "sku_id": row["sku_id"], "excess_units": math.ceil(excess - 1e-9),
                  "sell_window_days": int(row["sell_window_days"]), "shelf_left_days": int(row["shelf_left"]),
                  "transfer_to": None, "transfer_units": None, "discount_pct": None, "discount_days": None,
                  "reduce_units": None, "normal_order": round(float(row["normal_order"]), 1), "skip_orders": None,
                  "recent_avg_order": None if pd.isna(row.get("recent_avg_order")) else round(float(row["recent_avg_order"]), 1)}
        window = int(row["sell_window_days"])
        sells = f"{row['forecast_daily']:.1f}/day forecast sells {row['forecast_daily'] * window:,.0f} in {window} days"
        ruled_out = f" · {blocked_text(row['shelf_left'], blocked[index], counted)}" if index in blocked else ""
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
                              action_detail=f"{record['excess_units']:,} excess units ({sells}); demand ×{demand_lift(discount):.2f} at {discount:.0%} off{ruled_out}")
            else:
                units = record["excess_units"]
                needed = row["opening_stock"] / max(row["forecast_daily"] * window, 1e-9)
                perishable = is_perishable(row[category_col])
                why = (f"{NO_DISCOUNT}: needs ~{multiple(needed)}× normal sales "
                       f"{'before expiry' if perishable else f'within {window} days'}; "
                       f"{MAX_DISCOUNT:.0%} off gives ~{demand_lift(MAX_DISCOUNT):.1f}×{ruled_out}")
                if perishable:
                    record.update(action_type="write_off", reduce_units=units,
                                  action=write_off_text(units, row["normal_order"], row.get("recent_avg_order")),
                                  action_detail=f"{WRITE_OFF_HINT} · {why}" if units > 0 else why)
                else:
                    text, skip = reduce_text(units, row["normal_order"])
                    record.update(action_type="reduce_order", action=text, reduce_units=units, skip_orders=skip,
                                  action_detail=why)
        out.append(record)
    return pd.DataFrame(out, columns=COLUMNS)


def model_recent_orders(rows, counted):
    """
    recent_avg_order for the model scope's perishable rows, from the POS data
    (data/pos_synthetic.csv) and this morning's stock: the mean stock on
    delivery mornings (days_until_delivery == the cycle) over the last
    RECENT_ORDER_DAYS days, today included. Perishable stock is written off at
    each delivery, so that stock is exactly the delivery.
    """
    from config import DATA_DIR

    first = pd.Timestamp(counted) - pd.Timedelta(days=RECENT_ORDER_DAYS - 1)
    cols = ["date", "city", "sku_id", "category", "opening_stock", "days_until_delivery"]
    pos = pd.read_csv(DATA_DIR / "pos_synthetic.csv", usecols=cols, parse_dates=["date"])
    pos = pos[(pos["date"] >= first) & pos["category"].map(is_perishable)]
    today = rows.loc[rows["category"].map(is_perishable), cols]
    days = pd.concat([pos, today[today["date"] == pd.Timestamp(counted)]], ignore_index=True)
    deliveries = days[days["days_until_delivery"] == review_days_for(days["category"])]
    return deliveries.groupby(["city", "sku_id"])["opening_stock"].mean().rename("recent_avg_order").reset_index()


def festival_readiness_rows(cities):
    """
    Readiness for the default festival in every city, its name and the days
    from the stock-counting day to the festival week's first day; (None, "",
    None) when unavailable.
    """
    import festival_readiness

    try:
        tables, festival, week_start = [], "", None
        for city in cities:
            meta, table = festival_readiness.city_readiness(city)
            tables.append(table)
            festival = meta["festival"]
            counted = meta["data_date"] + pd.Timedelta(days=1)  # stock is counted the morning after the data date
            week_start = (meta["date"] - counted).days - (FESTIVAL_WEEK_DAYS - 1)
        return pd.concat(tables, ignore_index=True), festival, week_start
    except (FileNotFoundError, LookupError, ValueError):
        return None, "", None


def scope_actions(scope):
    """Actions for a stockout_view scope ("model" or "regional"), built once and kept in memory."""
    import stockout_view

    rows, counted = stockout_view.raw_rows(scope)
    if scope not in _CACHE or _CACHE[scope][0] is not rows:  # rebuilt when stockout_view's table is
        category_col = "model_category" if "model_category" in rows else "category"
        over = rows
        if scope == "model" and (rows["risk_status"] == "overstock").any():
            try:
                over = rows.merge(model_recent_orders(rows, counted), on=["city", "sku_id"], how="left")
            except (FileNotFoundError, ValueError):
                pass
        readiness, festival, week_start = (None, "", None)
        if (rows["risk_status"] == "overstock").any():
            readiness, festival, week_start = festival_readiness_rows(sorted(rows["city"].unique()))
        _CACHE[scope] = rows, overstock_actions(over, shortfalls(rows, readiness, festival, week_start), category_col,
                                                     counted)
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
