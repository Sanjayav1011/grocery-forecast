"""
product_detail.py
-------------------
Data for the admin dashboard's product panel (GET /api/product): one product
in one city, around "today" (the last POS day the regional forecast was made
from).

  history    the CHART_DAYS up to today: units sold, true demand and stockout
             days from the synthetic POS data (tracked products only)
  forecast   the LEAD_TIME_DAYS after today. Tracked products: LightGBM per day
             (data/bigbasket_tracked_daily_forecast.csv, the values the
             regional forecast averages). Analog products: the same weighted,
             price-adjusted average of their analogs' DAILY forecasts, scaled so
             the 7-day mean is exactly the saved forecast_daily (the rescale
             only absorbs rounding of the saved weights).
  festivals  festival dates in that window (festival_calendar.festival_df)
  details    MRP, stock and risk from the scope's table (model risk report or
             regional), opening stock simulated
  analogs    for analog products: the K tracked products the forecast was built
             from, with normalised inverse-distance weights and price factors
  festival_planning  with festival=<name>: the "Festival planning" estimate,
             7-day forecast x the city's measured uplift for the category
             (festival_uplift.py)
"""

import numpy as np
import pandas as pd

import festival_uplift
import stockout_view
from config import DATA_DIR, LEAD_TIME_DAYS
from festival_calendar import festival_df

CHART_DAYS = 60
DIRECT = stockout_view.DIRECT
NOTES = {
    "history": "Units sold, true demand and stockout days are synthetic POS data (simulated).",
    "stock": "Opening stock, days of cover, next delivery and reorder qty come from simulated stock.",
    "direct": "Forecast: LightGBM, trained on the simulated sales history.",
    "analog": "Forecast: estimated, from 5 analogs (weighted, price-adjusted average of their daily forecasts). "
              "This product has no sales history.",
}
_CACHE = {}


class ProductNotFound(LookupError):
    pass


def tracked_daily():
    """Per-day forecasts of the tracked products, (city, sku_id) x date (kept in memory)."""
    if "daily" not in _CACHE:
        daily = pd.read_csv(DATA_DIR / "bigbasket_tracked_daily_forecast.csv")
        _CACHE["daily"] = daily.pivot_table(index=["city", "sku_id"], columns="date", values="prediction").sort_index(axis=1)
    return _CACHE["daily"]


def analog_daily(row, daily):
    """
    An analog product's forecast per day: sum of weight x price factor x the
    analog's daily forecast, rescaled so the mean is exactly forecast_daily.
    Returns (dates, values).
    """
    ids = str(row["analog_ids"]).split(";")
    weights = np.array([float(value) for value in str(row["analog_weights"]).split(";")])
    price_adj = np.array([float(value) for value in str(row["analog_price_adj"]).split(";")])
    matrix = daily.loc[[(row["city"], sku) for sku in ids]].to_numpy()  # (K, days)
    shape = (weights * price_adj) @ matrix / weights.sum()
    target = float(row["forecast_daily"])
    values = shape * (target / shape.mean()) if shape.mean() > 0 else np.full(len(shape), target)
    return list(daily.columns), values


def recent_pos(today):
    """POS rows for the forecast_regional.HISTORY_DAYS up to today (kept in memory)."""
    from forecast_regional import HISTORY_DAYS

    if ("pos", today) not in _CACHE:
        pos = pd.read_csv(DATA_DIR / "pos_synthetic.csv", parse_dates=["date"])
        _CACHE["pos", today] = pos[(pos["date"] > today - pd.Timedelta(days=HISTORY_DAYS)) & (pos["date"] <= today)]
    return _CACHE["pos", today]


def clear_cache():
    _CACHE.clear()


def _text(value):
    return "" if pd.isna(value) else str(value)


def _number(value, digits=2):
    return None if pd.isna(value) else round(float(value), digits)


def tracked_series(city, sku_id, today):
    """History rows for the chart and the per-day forecast after today."""
    pos = recent_pos(today)
    series = pos[(pos["city"] == city) & (pos["sku_id"] == sku_id)]
    shown = series[series["date"] > today - pd.Timedelta(days=CHART_DAYS)].sort_values("date")
    history = [{"date": day.date().isoformat(), "units_sold": int(sold), "true_demand": _number(true),
                "stockout": bool(flag)}
               for day, sold, true, flag in shown[["date", "units_sold", "true_demand", "stockout_flag"]].itertuples(index=False)]
    daily = tracked_daily().loc[(city, sku_id)]
    forecast = [{"date": day, "units": round(float(units), 4)} for day, units in daily.items()]
    return history, forecast


def analog_list(row, regional):
    """The analogs behind an analog forecast, with weights, price factors and their own forecast in the city."""
    ids = str(row["analog_ids"]).split(";")
    weights = [float(value) for value in str(row["analog_weights"]).split(";")]
    price_adj = [float(value) for value in str(row["analog_price_adj"]).split(";")]
    in_city = regional[(regional["city"] == row["city"]) & regional["sku_id"].isin(ids)].set_index("sku_id")
    return [{
        "sku_id": sku, "product_name": _text(in_city.at[sku, "product_name"]) if sku in in_city.index else "",
        "weight": weight, "price_adjustment": adj,
        "forecast_daily": _number(in_city.at[sku, "forecast_daily"]) if sku in in_city.index else None,
    } for sku, weight, adj in zip(ids, weights, price_adj)]


def festival_note(city, category, forecast_daily, festival, data_date):
    """The festival-planning calculation for one product, or None without a measured uplift."""
    uplifts = festival_uplift.load_uplifts()
    match = uplifts[(uplifts["festival"] == festival) & (uplifts["city"] == city) & (uplifts["category"] == category)]
    upcoming = festival_df()
    upcoming = upcoming[(upcoming["festival_name"] == festival) & (upcoming["date"] > data_date)]
    if match.empty or upcoming.empty:
        return None
    uplift = match.iloc[0]
    seven_day = float(forecast_daily) * LEAD_TIME_DAYS
    return {
        "festival": festival, "date": upcoming["date"].iloc[0].date().isoformat(), "category": category,
        "uplift": float(uplift["uplift"]), "occurrences": int(uplift["occurrences"]),
        "first_year": int(uplift["first_year"]), "last_year": int(uplift["last_year"]),
        "seven_day_units": round(seven_day, 1), "festival_week_units": round(seven_day * float(uplift["uplift"]), 1),
    }


def product_detail(city, sku_id, scope="regional", festival=""):
    """Everything the product panel shows; ProductNotFound for an unknown city / SKU."""
    rows, stock_day = stockout_view.raw_rows(scope)
    regional, forecast_start = stockout_view.raw_rows("regional")
    match = rows[(rows["city"].str.casefold() == city.strip().casefold()) & (rows["sku_id"] == sku_id.strip())]
    if match.empty:
        raise ProductNotFound(sku_id)
    row = match.iloc[0]
    city, sku_id = row["city"], row["sku_id"]
    today = forecast_start - pd.Timedelta(days=1)
    tracked = row["forecast_source"] == DIRECT

    if tracked:
        history, forecast = tracked_series(city, sku_id, today)
        analogs = []
    else:
        history = []
        dates, values = analog_daily(row, tracked_daily())
        forecast = [{"date": day, "units": round(float(units), 4)} for day, units in zip(dates, values)]
        analogs = analog_list(row, regional)

    first = today - pd.Timedelta(days=CHART_DAYS - 1)
    last = today + pd.Timedelta(days=LEAD_TIME_DAYS)
    all_festivals = festival_df()
    festivals = all_festivals[all_festivals["date"].between(first, last)]
    after = all_festivals[(all_festivals["date"] > last) & (all_festivals["date"] <= last + pd.Timedelta(days=14))].head(1)
    return {
        "city": city, "sku_id": sku_id, "scope": scope, "today": today.date().isoformat(),
        "details": {
            "product_name": _text(row["product_name"]), "brand": _text(row["brand"]), "category": _text(row["category"]),
            "price": _number(row["price"]), "forecast_source": row["forecast_source"],
            "forecast_daily": _number(row["forecast_daily"]), "days_of_cover": _number(row["days_of_cover"]),
            "days_until_delivery": int(row["days_until_delivery"]), "reorder_qty": int(row["reorder_qty"]),
            "risk_status": row["risk_status"],
        },
        "stock": {"units": int(row["opening_stock"]), "date": pd.Timestamp(stock_day).date().isoformat(),
                  "simulated": True},
        "history": history,
        "forecast": forecast,
        "festivals": [{"date": day.date().isoformat(), "name": name}
                      for day, name in festivals[["date", "festival_name"]].itertuples(index=False)],
        # a festival just past the chart (e.g. Diwali 3 days after the forecast window), shown at its edge
        "festival_after": {"date": after["date"].iloc[0].date().isoformat(), "name": after["festival_name"].iloc[0]}
                          if not after.empty else None,
        "analogs": analogs,
        "festival_planning": festival_note(city, row["model_category"] if "model_category" in row else row["category"],
                                           row["forecast_daily"], festival, today) if festival else None,
        "notes": [NOTES["history"]] * tracked + [NOTES["stock"], NOTES["direct" if tracked else "analog"]],
    }
