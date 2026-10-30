"""
festival_readiness.py
-----------------------
"<Festival> readiness" for the admin dashboard and the vendor alerts, as a
top-up above normal orders (how stores plan festivals): for every product in
a city (one row per product), how much EXTRA must be ordered on top of the
regular deliveries to cover the festival-week demand? A planning estimate;
stock is simulated.

  serving date D        config.SERVING_DATE; stock is counted on S = D + 1
  festival week         the 7 days ending on the festival day F
  festival-week demand  the festival planning estimate (festival_uplift.planning_rows):
                        7-day forecast x the category's measured festival uplift
  regular deliveries    on S + d, S + d + R, ... (d = days until the next delivery,
                        R = the delivery cycle, inventory.review_days_for), each
                        bringing one cycle of NORMAL forecast demand
                        (forecast_daily x R), as the replenishment policy does
  before the week       forecast demand from S to the day before the festival week
                        is taken from the stock on hand and the deliveries before
                        the week, oldest first
  usable at week start  what is left of that stock, if still within shelf life on
                        the festival day (a batch delivered on day t keeps until
                        t + shelf life - 1; the stock on hand came with the last
                        delivery, S + d - R). Non-perishables keep for months.
  week deliveries       the regular deliveries arriving during the festival week
                        (up to F). They count in full: each is one cycle of demand,
                        sold within its cycle (the cycle is never longer than the
                        shelf life)
  available             usable at week start + week deliveries
  shortfall             max(0, festival-week demand - available)
  small gaps            a shortfall under max(MIN_GAP_UNITS, MIN_GAP_SHARE x
                        festival-week demand) is ignored: "Ready"
  order by              the last regular delivery that arrives before the festival,
                        minus the LEAD_TIME_DAYS order lead time
  status                "Order now"  a shortfall, and that delivery arrives within
                                     ORDER_NOW_DAYS of D (its order-by date has come)
                        "At risk"    a shortfall otherwise
                        "Ready"      no shortfall, or a small gap
  extra order           the quantity on top of the regular orders: shortfall +
                        SAFETY_STOCK_DAYS days of festival demand (festival-week
                        demand / 7 a day), rounded up
  value at risk         shortfall x MRP (the email ranks "Order now" rows by it)
"""

import json

import numpy as np
import pandas as pd

from config import LEAD_TIME_DAYS, SAFETY_STOCK_DAYS, SHELF_LIFE_DAYS
from inventory import review_days_for

ORDER_NOW_DAYS = 7
MIN_GAP_UNITS = 5
MIN_GAP_SHARE = 0.10
STATUSES = ["Order now", "At risk", "Ready"]
FESTIVAL_WEEK_DAYS = 7
SORT_COLUMNS = {"stock": "opening_stock", "demand": "festival_week_units", "shortfall": "shortfall",
                "order_by": "order_by", "extra": "extra_order", "value": "value_at_risk"}
COLUMNS = ["city", "sku_id", "product_name", "brand", "category", "model_category", "price", "forecast_source",
           "opening_stock", "forecast_daily", "forecast_7d", "uplift", "festival_week_units", "usable_at_week_start",
           "week_deliveries", "available", "shortfall", "value_at_risk", "last_delivery", "order_by", "extra_order",
           "status"]
_CACHE = {}


def readiness(frame, festival_date, serving_date):
    """
    Adds usable_at_week_start, week_deliveries, available, shortfall,
    value_at_risk, last_delivery, order_by, extra_order and status to `frame`
    (needs forecast_daily, festival_week_units, opening_stock,
    days_until_delivery and model_category; price for value_at_risk).
    """
    frame = frame.copy()
    serving = pd.Timestamp(serving_date)
    start = serving + pd.Timedelta(days=1)  # stock is counted this morning
    festival_offset = (pd.Timestamp(festival_date) - start).days  # day offsets from the counting day
    week_offset = festival_offset - (FESTIVAL_WEEK_DAYS - 1)

    cycle = review_days_for(frame["model_category"]).astype(int)
    shelf_life = frame["model_category"].map(SHELF_LIFE_DAYS).to_numpy()
    until = frame["days_until_delivery"].to_numpy().astype(int)
    rate = frame["forecast_daily"].to_numpy(dtype=float)
    normal_delivery = (rate * cycle)[:, None]  # one cycle of normal forecast demand

    slots = int(max(festival_offset, 0) // cycle.min()) + 2
    deliveries = until[:, None] + np.arange(slots)[None, :] * cycle[:, None]  # arrival offsets

    # stock carried into the week: batch 0 is the stock on hand (it came with the last
    # delivery), then each regular delivery before the week
    arrival = np.concatenate([(until - cycle)[:, None], deliveries], axis=1)
    carried = np.concatenate([frame["opening_stock"].to_numpy(dtype=float)[:, None],
                              np.where(deliveries < week_offset, normal_delivery, 0.0)], axis=1)
    before_week = rate * max(week_offset, 0)  # demand before the week, oldest stock first
    filled = np.cumsum(carried, axis=1)
    taken = np.clip(np.minimum(filled, before_week[:, None]) - (filled - carried), 0, None)
    fresh = arrival + shelf_life[:, None] > festival_offset  # still within shelf life on the festival day
    frame["usable_at_week_start"] = ((carried - taken) * fresh).sum(axis=1)
    in_week = (deliveries >= week_offset) & (deliveries <= festival_offset)
    frame["week_deliveries"] = (in_week * normal_delivery).sum(axis=1)
    frame["available"] = frame["usable_at_week_start"] + frame["week_deliveries"]

    frame["shortfall"] = (frame["festival_week_units"] - frame["available"]).clip(lower=0)
    frame["value_at_risk"] = frame["shortfall"] * frame["price"] if "price" in frame else np.nan
    last_slot = np.where(deliveries < festival_offset, deliveries, -10 ** 6).max(axis=1)
    has_slot = last_slot > -10 ** 6
    frame["last_delivery"] = [start + pd.Timedelta(days=int(offset)) if ok else pd.NaT
                              for offset, ok in zip(last_slot, has_slot)]
    frame["order_by"] = frame["last_delivery"] - pd.Timedelta(days=LEAD_TIME_DAYS)
    threshold = np.maximum(MIN_GAP_UNITS, MIN_GAP_SHARE * frame["festival_week_units"])
    short = frame["shortfall"] >= threshold  # smaller gaps are ignored
    soon = frame["last_delivery"] <= serving + pd.Timedelta(days=ORDER_NOW_DAYS)
    frame["status"] = np.select([short & soon, short], STATUSES[:2], default=STATUSES[2])
    daily_festival_demand = frame["festival_week_units"] / FESTIVAL_WEEK_DAYS
    frame["extra_order"] = np.where(short, np.ceil(frame["shortfall"] + SAFETY_STOCK_DAYS * daily_festival_demand), 0).astype(int)
    return frame


def city_readiness(city, festival="", next_after_data_date=False):
    """(meta, table) for one city and festival (default: the festival-planning default), kept in memory."""
    import festival_uplift

    meta, table = festival_uplift.planning_rows(city, festival, next_after_data_date)
    key = (meta["city"], meta["festival"], meta["data_date"])
    if key not in _CACHE:
        if len(_CACHE) > 32:
            _CACHE.clear()
        _CACHE[key] = readiness(table, meta["date"], meta["data_date"])[COLUMNS]
    return meta, _CACHE[key]


def clear_cache():
    _CACHE.clear()


def filter_and_sort(table, params):
    """status / category / brand filters (exact, case-insensitive) and a SORT_COLUMNS sort; default: status, then shortfall."""
    rows = table
    for name, column in [("status", "status"), ("category", "category"), ("brand", "brand")]:
        value = params.get(name, "").strip()
        if value:
            rows = rows[rows[column].str.casefold() == value.casefold()]
    sort, order = params.get("sort", ""), params.get("order", "")
    if order and order not in {"asc", "desc"}:
        raise ValueError("order must be asc or desc")
    if sort and sort not in SORT_COLUMNS:
        raise ValueError(f"sort must be one of {', '.join(SORT_COLUMNS)}")
    if not sort:
        rank = rows["status"].map({status: i for i, status in enumerate(STATUSES)})
        return rows.assign(_rank=rank).sort_values(["_rank", "shortfall", "sku_id"], ascending=[True, False, True],
                                                   kind="stable").drop(columns="_rank")
    column = SORT_COLUMNS[sort]
    ascending = (order or ("asc" if sort == "order_by" else "desc")) == "asc"
    return rows.sort_values([column, "sku_id"], ascending=[ascending, True], na_position="last", kind="stable")


ROUNDING = {"forecast_daily": 2, "forecast_7d": 1, "uplift": 3, "festival_week_units": 1, "usable_at_week_start": 1,
            "week_deliveries": 1, "available": 1, "shortfall": 1, "value_at_risk": 0}


def to_records(rows):
    out = rows.assign(last_delivery=rows["last_delivery"].dt.strftime("%Y-%m-%d"),
                      order_by=rows["order_by"].dt.strftime("%Y-%m-%d"))
    out = out.round(ROUNDING)
    return json.loads(out.to_json(orient="records"))


def to_csv(rows):
    return rows.round(ROUNDING).to_csv(index=False, date_format="%Y-%m-%d")


def readiness_payload(params):
    """/api/readiness: one city's readiness for a festival, filtered, sorted and paged (50 rows)."""
    meta, table = city_readiness(params.get("city", ""), params.get("festival", ""))
    rows = filter_and_sort(table, params)
    offset = int(params.get("offset", "0") or 0)
    limit = min(max(int(params.get("limit", "50") or 50), 1), 500)
    as_day = lambda value: pd.Timestamp(value).date().isoformat()
    return rows, {
        "city": meta["city"], "festival": meta["festival"], "date": as_day(meta["date"]),
        "days_until": int((meta["date"] - meta["data_date"]).days), "data_date": as_day(meta["data_date"]),
        "stock_date": as_day(meta["forecast_start"]),
        "counts": {status: int((table["status"] == status).sum()) for status in STATUSES},
        "categories": sorted(table["category"].dropna().unique().tolist()),
        "brands": table[table["status"] != "Ready"]["brand"].value_counts().head(300).index.tolist(),
        "total": len(rows), "products": len(table),
        "rows": to_records(rows.iloc[max(offset, 0):max(offset, 0) + limit]),
    }


def alert_rows(city, within_days):
    """
    For the vendor email: the next festival within `within_days` of the
    serving date that has measured uplifts, and its "Order now" rows by value
    at risk (shortfall x MRP; no MRP last); None when there is no such festival.
    """
    import festival_uplift
    import stockout_view

    _, forecast_start = stockout_view.raw_rows("regional")
    serving = forecast_start - pd.Timedelta(days=1)
    known = set(festival_uplift.load_uplifts()["festival"])
    festivals = festival_uplift.festival_df()
    festivals = festivals[(festivals["date"] > serving) & (festivals["date"] <= serving + pd.Timedelta(days=within_days))
                          & festivals["festival_name"].isin(known)]
    if festivals.empty:
        return None
    meta, table = city_readiness(city, festivals["festival_name"].iloc[0], next_after_data_date=True)
    rows = table[table["status"] == "Order now"].sort_values(["value_at_risk", "shortfall", "sku_id"],
                                                             ascending=[False, False, True], na_position="last")
    return {"festival": meta["festival"], "date": meta["date"].date().isoformat(),
            "days_until": int((meta["date"] - serving).days), "rows": rows}
