"""
festival_uplift.py
--------------------
Festival-week demand uplift per city x model category x festival, measured
from past years' history only. The admin "Festival planning" list uses it:

    festival-week demand = product's current 7-day forecast x uplift

    uplift = mean over the festival's past occurrences of
             mean daily demand in the festival week
             / mean daily demand on the 4 normal weeks before it

  festival week  the 7 days ending on the festival day
  normal days    days with no festival in the next 7 days or the last 7
                 (festival_calendar's days_until_next_festival /
                 days_since_last_festival); the last NORMAL_DAYS of them
                 before the festival week, within LOOKBACK_DAYS
  past           the festival day is on or before the data date (the last
                 day of history), so nothing after it feeds an uplift

Demand is build_features' unconstrained `demand` (stockout days
un-censored), summed over a city's tracked products in each category. This
is a planning estimate, not a model forecast: the model only forecasts
LEAD_TIME_DAYS ahead.

Output: data/festival_uplift.csv (one row per city x category x festival)
Usage : python festival_uplift.py
"""

import json

import numpy as np
import pandas as pd

from config import DATA_DIR, LEAD_TIME_DAYS
from festival_calendar import build_calendar_table, festival_df

FEATURES_PATH = DATA_DIR / "features.csv"
UPLIFT_PATH = DATA_DIR / "festival_uplift.csv"
FESTIVAL_WEEK_DAYS = 7
NORMAL_DAYS = 28  # 4 normal weeks
MIN_NORMAL_DAYS = 14
LOOKBACK_DAYS = 120
QUIET_DAYS = 7  # a normal day has no festival within this many days either side


def occurrence_ratios(daily, data_date):
    """
    One row per past festival occurrence x city x category: the festival
    week's mean daily demand over the normal days' mean. `daily` has date,
    city, category and demand (any granularity; summed per day).
    """
    data_date = pd.Timestamp(data_date)
    daily = daily[pd.to_datetime(daily["date"]) <= data_date]  # history only
    totals = daily.assign(date=pd.to_datetime(daily["date"])).pivot_table(
        index="date", columns=["city", "category"], values="demand", aggfunc="sum")
    if totals.empty:
        return pd.DataFrame(columns=["festival", "date", "city", "category", "ratio", "normal_days"])
    calendar = build_calendar_table(totals.index.min(), totals.index.max())
    quiet = (calendar["days_until_next_festival"] > QUIET_DAYS) & (calendar["days_since_last_festival"] > QUIET_DAYS)
    normal_dates = calendar.loc[quiet, "date"]
    rows = []
    for festival in festival_df().itertuples(index=False):
        week = pd.date_range(end=festival.date, periods=FESTIVAL_WEEK_DAYS)
        if festival.date > data_date or week[0] < totals.index.min():
            continue
        before = normal_dates[(normal_dates < week[0]) & (normal_dates >= week[0] - pd.Timedelta(days=LOOKBACK_DAYS))]
        before = before.tail(NORMAL_DAYS)
        if len(before) < MIN_NORMAL_DAYS:
            continue
        ratio = totals.loc[week].mean() / totals.loc[before].mean()
        for (city, category), value in ratio.items():
            if np.isfinite(value):
                rows.append({"festival": festival.festival_name, "date": festival.date, "city": city,
                             "category": category, "ratio": value, "normal_days": len(before)})
    return pd.DataFrame(rows)


def summarise(ratios):
    """Mean ratio per festival x city x category, with how many past occurrences it averages."""
    if ratios.empty:
        return pd.DataFrame(columns=["festival", "city", "category", "uplift", "occurrences", "first_year", "last_year"])
    year = pd.to_datetime(ratios["date"]).dt.year
    summary = ratios.assign(year=year).groupby(["festival", "city", "category"]).agg(
        uplift=("ratio", "mean"), occurrences=("ratio", "size"), first_year=("year", "min"),
        last_year=("year", "max")).reset_index()
    summary["uplift"] = summary["uplift"].round(3)
    return summary


_UPLIFTS = {}


def load_uplifts():
    """data/festival_uplift.csv, kept in memory."""
    path = DATA_DIR / "festival_uplift.csv"
    if path not in _UPLIFTS:
        _UPLIFTS.clear()
        _UPLIFTS[path] = pd.read_csv(path)
    return _UPLIFTS[path]


def upcoming_festivals(data_date, window_end):
    """Festivals after the forecast window, within a year of the data date; the next date of each name."""
    festivals = festival_df()
    festivals = festivals[(festivals["date"] > window_end) & (festivals["date"] <= data_date + pd.Timedelta(days=365))]
    return festivals.drop_duplicates("festival_name").reset_index(drop=True)


def planning_rows(city, festival="", next_after_data_date=False):
    """
    The festival-planning estimate for every product in one city (one row per
    product): festival-week units = 7-day forecast x the measured uplift for
    its model category in that city. Products whose category has no measured
    uplift are left out. Returns (meta, table); `festival` defaults to the
    first festival after the 7-day forecast window. With
    next_after_data_date, a named festival is its next occurrence after the
    data date, even inside the forecast window (for the vendor alerts).
    """
    import stockout_view

    rows, forecast_start = stockout_view.raw_rows("regional")
    data_date = forecast_start - pd.Timedelta(days=1)
    window_end = data_date + pd.Timedelta(days=LEAD_TIME_DAYS)
    options = upcoming_festivals(data_date, window_end)
    if next_after_data_date:
        options = festival_df()
        options = options[options["date"] > data_date].drop_duplicates("festival_name").reset_index(drop=True)
    chosen = options[options["festival_name"].str.casefold() == festival.strip().casefold()] if festival else options.head(1)
    if chosen.empty:
        raise ValueError(f"no upcoming festival called {festival!r}")
    name, day = chosen["festival_name"].iloc[0], chosen["date"].iloc[0]
    in_city = rows[rows["city"].str.casefold() == city.strip().casefold()]
    if in_city.empty:
        raise stockout_view.UnknownCity(city)
    city = in_city["city"].iloc[0]

    uplifts = load_uplifts()
    uplifts = uplifts[(uplifts["festival"] == name) & (uplifts["city"] == city)].set_index("category")
    key = ("products", city)
    if key not in _UPLIFTS:  # one row per product, per city (stockout_view.unique_products)
        table = in_city.copy()
        for col in ["product_name", "brand", "pack_size_value", "pack_size_unit"]:
            table[col] = table[col].fillna("").astype(str)
        _UPLIFTS[key] = stockout_view.one_row_per_product(table)
    table = _UPLIFTS[key].copy()
    table["uplift"] = table["model_category"].map(uplifts["uplift"])
    table = table[table["uplift"].notna()]
    table["forecast_7d"] = table["forecast_daily"] * LEAD_TIME_DAYS
    table["festival_week_units"] = table["forecast_7d"] * table["uplift"]
    table["revenue_estimate"] = table["festival_week_units"] * table["price"]
    meta = {"city": city, "festival": name, "date": day, "data_date": data_date, "forecast_start": forecast_start,
            "window_end": window_end, "options": options, "uplifts": uplifts}
    return meta, table


def festival_planning(city, festival="", rank="units", limit=10):
    """
    The admin "Festival planning" list for one city: the top `limit`
    products of planning_rows by festival-week units or revenue (units x
    MRP). A planning estimate, not a model forecast.
    """
    if rank not in {"units", "revenue"}:
        raise ValueError("rank must be units or revenue")
    meta, table = planning_rows(city, festival)
    name, day, data_date, uplifts = meta["festival"], meta["date"], meta["data_date"], meta["uplifts"]
    sort = "festival_week_units" if rank == "units" else "revenue_estimate"
    top = table.dropna(subset=[sort]).sort_values([sort, "sku_id"], ascending=[False, True]).head(limit)
    columns = ["sku_id", "city", "product_name", "brand", "category", "model_category", "price", "forecast_source",
               "forecast_7d", "uplift", "festival_week_units", "revenue_estimate"]
    top = top[columns].round({"forecast_7d": 1, "festival_week_units": 1, "revenue_estimate": 0})
    as_day = lambda value: pd.Timestamp(value).date().isoformat()
    return {
        "city": meta["city"], "festival": name, "date": as_day(day), "days_until": int((day - data_date).days),
        "data_date": as_day(data_date), "window_start": as_day(meta["forecast_start"]),
        "window_end": as_day(meta["window_end"]), "rank": rank,
        "options": [{"festival": row.festival_name, "date": as_day(row.date), "days_until": int((row.date - data_date).days),
                     "has_uplift": bool((load_uplifts()["festival"] == row.festival_name).any())}
                    for row in meta["options"].itertuples(index=False)],
        "uplifts": [{"category": category, "uplift": float(row["uplift"]), "occurrences": int(row["occurrences"]),
                     "first_year": int(row["first_year"]), "last_year": int(row["last_year"])}
                    for category, row in uplifts.sort_values("uplift", ascending=False).iterrows()],
        "rows": json.loads(top.to_json(orient="records")),
    }


def build():
    features = pd.read_csv(FEATURES_PATH, usecols=["date", "city", "category", "demand"])
    data_date = pd.to_datetime(features["date"]).max()
    summary = summarise(occurrence_ratios(features, data_date))
    summary.to_csv(UPLIFT_PATH, index=False)
    return summary, data_date


if __name__ == "__main__":
    summary, data_date = build()
    print(f"Saved {len(summary):,} festival uplifts (history up to {data_date.date()}) to {UPLIFT_PATH}")
    for festival in ["Diwali", "Ganesh Chaturthi"]:
        rows = summary[summary["festival"] == festival]
        if not rows.empty:
            print(f"{festival} (x festival-week demand vs the 4 normal weeks before):")
            print(rows.pivot(index="category", columns="city", values="uplift").round(2).to_string())
