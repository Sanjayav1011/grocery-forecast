"""
product_detail.py
-------------------
Data for the admin dashboard's product panel (GET /api/product): one product
in one city, around "today" (the last POS day the regional forecast was made
from).

  history    the CHART_DAYS up to today: units sold, true demand and stockout
             days from the synthetic POS data (tracked products only)
  forecast   the LEAD_TIME_DAYS after today. Tracked products: LightGBM per day
             (forecast_regional.daily_forecasts, the numbers the regional
             forecast averages). Analog products: their 7-day average, so flat.
  festivals  festival dates in that window (festival_calendar.festival_df)
  details    MRP, stock and risk from the scope's table (model risk report or
             regional), opening stock simulated
  analogs    for analog products: the K tracked products the forecast was built
             from, with normalised inverse-distance weights and price factors
"""

import pandas as pd

import stockout_view
from config import DATA_DIR, LEAD_TIME_DAYS
from festival_calendar import festival_df

CHART_DAYS = 60
DIRECT = stockout_view.DIRECT
NOTES = {
    "history": "Units sold, true demand and stockout days are synthetic POS data (simulated).",
    "stock": "Opening stock, days of cover, next delivery and reorder qty come from simulated stock.",
    "direct": "Forecast: LightGBM, trained on the simulated sales history.",
    "analog": "Forecast: weighted average of the analog products' forecasts, adjusted for price (estimated). "
              "This product has no sales history.",
}
_CACHE = {}


class ProductNotFound(LookupError):
    pass


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
    from forecast_regional import daily_forecasts  # loads LightGBM; only needed here

    pos = recent_pos(today)
    series = pos[(pos["city"] == city) & (pos["sku_id"] == sku_id)]
    shown = series[series["date"] > today - pd.Timedelta(days=CHART_DAYS)].sort_values("date")
    history = [{"date": day.date().isoformat(), "units_sold": int(sold), "true_demand": _number(true),
                "stockout": bool(flag)}
               for day, sold, true, flag in shown[["date", "units_sold", "true_demand", "stockout_flag"]].itertuples(index=False)]
    daily, _ = daily_forecasts(series, today)
    forecast = [{"date": day.date().isoformat(), "units": round(float(units), 2)}
                for day, units in daily.sort_values("date")[["date", "prediction"]].itertuples(index=False)]
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


def product_detail(city, sku_id, scope="regional"):
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
        days = pd.date_range(forecast_start, periods=LEAD_TIME_DAYS)
        forecast = [{"date": day.date().isoformat(), "units": _number(row["forecast_daily"])} for day in days]
        analogs = analog_list(row, regional)

    first = today - pd.Timedelta(days=CHART_DAYS - 1)
    last = today + pd.Timedelta(days=LEAD_TIME_DAYS)
    festivals = festival_df()
    festivals = festivals[festivals["date"].between(first, last)]
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
        "analogs": analogs,
        "notes": [NOTES["history"]] * tracked + [NOTES["stock"], NOTES["direct" if tracked else "analog"]],
    }
