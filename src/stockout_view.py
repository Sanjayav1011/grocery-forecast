"""
stockout_view.py
------------------
Tables behind the admin dashboard's "stocks out first" views, one row per
product x city:

  scope "model"     the tracked products on the latest day of the test
                    window, scored with train_model.build_risk_report (the
                    same inventory.classify_risk rule and inputs as
                    outputs/stockout_risk_report.csv), every risk status
  scope "regional"  the forecast catalog, data/bigbasket_regional_inventory.csv

Tracked products get sales history from data/pos_synthetic.csv over the days
before the planning day (the day stock is counted): units sold in the last 7
and 28 days, average daily sales over 28 days and trend = last 28 days vs the
28 before. Analog catalog products have no history; their sales columns stay
empty and the forecast stands in for the daily rate.

Products listed under several SKUs are shown once per city
(unique_products). All sales and stock figures come from synthetic data.

filter_rows / sort_rows / brand_summary serve /api/stockout and
/api/stockout/brands in dashboard.py.
"""

import json
import re
import threading

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUTS_DIR

SCOPES = ("model", "regional")
DIRECT = "lightgbm_direct"  # forecast_source values written by forecast_regional.py
ANALOG_PREFIX = "lightgbm_analog"
HISTORY_DAYS = 28
PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# sort parameter -> column
SORT_COLUMNS = {
    "cover": "days_of_cover", "units_28d": "units_28d", "avg_daily_sales": "avg_daily_sales",
    "price": "price", "reorder_qty": "reorder_qty", "trend": "trend_pct",
}
COLUMNS = [
    "city", "sku_id", "product_name", "brand", "category", "price", "units_7d", "units_28d",
    "avg_daily_sales", "trend_pct", "forecast_daily", "opening_stock", "days_of_cover", "days_until_delivery",
    "reorder_qty", "risk_status", "forecast_source", "has_history", "seller_rank_pct",
    "pack_size_value", "pack_size_unit",
]
EXPORT_COLUMNS = [col for col in COLUMNS if col not in {"seller_rank_pct", "pack_size_value", "pack_size_unit"}]

_TABLES = {}
_LOCK = threading.RLock()


def product_key(row):
    """Normalised product name + pack size: the catalog lists some products under several SKUs."""
    name = " ".join(re.sub(r"[^0-9a-z]+", " ", row.get("product_name", "").casefold()).split())
    return name, row.get("pack_size_value", ""), row.get("pack_size_unit", "")


def unique_products(rows):
    """One row per product_key: the directly forecast SKU if there is one, else the first (lowest) SKU id."""
    best = {}
    for row in rows:
        rank = (row["forecast_source"] != DIRECT, row["sku_id"])
        key = product_key(row)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, row)
    return [row for _, row in best.values()]


def sales_history(planning_day):
    """Per city x sku: units sold in the 7 / 28 days before planning_day, average daily sales and trend %."""
    pos = pd.read_csv(DATA_DIR / "pos_synthetic.csv", usecols=["date", "city", "sku_id", "units_sold"])
    age = (pd.Timestamp(planning_day) - pd.to_datetime(pos["date"])).dt.days  # 1 = the day before
    pos = pos.assign(age=age)[(age >= 1) & (age <= 2 * HISTORY_DAYS)]
    window = lambda first, last: pos["units_sold"].where(pos["age"].between(first, last), 0)
    sales = pos.assign(units_7d=window(1, 7), units_28d=window(1, HISTORY_DAYS),
                       previous_28d=window(HISTORY_DAYS + 1, 2 * HISTORY_DAYS))
    sales = sales.groupby(["city", "sku_id"])[["units_7d", "units_28d", "previous_28d"]].sum().reset_index()
    sales["avg_daily_sales"] = sales["units_28d"] / HISTORY_DAYS
    previous = sales.pop("previous_28d")
    sales["trend_pct"] = ((sales["units_28d"] - previous) / previous.where(previous > 0) * 100)
    return sales


def one_row_per_product(table):
    """unique_products() within each city; keeps the table's own row order."""
    keep = []
    cols = ["sku_id", "product_name", "pack_size_value", "pack_size_unit", "forecast_source"]
    for _, rows in table.groupby("city", sort=False):
        records = rows[cols].assign(index=rows.index).to_dict("records")
        keep += [record["index"] for record in unique_products(records)]
    return table.loc[sorted(keep)]


def finish(table, planning_day):
    """Join sales history to tracked rows, de-duplicate, rank sellers and keep COLUMNS."""
    table = table.copy()
    for col in ["product_name", "brand", "category", "pack_size_value", "pack_size_unit"]:
        table[col] = table[col].fillna("").astype(str)
    table["has_history"] = table["forecast_source"] == DIRECT
    sales = sales_history(planning_day)
    table = table.merge(sales, on=["city", "sku_id"], how="left")
    history_cols = ["units_7d", "units_28d", "avg_daily_sales", "trend_pct"]
    table.loc[~table["has_history"], history_cols] = np.nan
    table = one_row_per_product(table.reset_index(drop=True))
    # share of the city's sellers (products with history) that sold more in 28 days than this one
    by_city = table.groupby("city")["units_28d"]
    table["seller_rank_pct"] = (by_city.rank(ascending=False, method="min") - 1) / by_city.transform("count")
    table = table[COLUMNS].reset_index(drop=True)
    table.attrs["planning_day"] = pd.Timestamp(planning_day).date().isoformat()
    return table


def model_rows():
    """Every tracked product x city on the latest test-window day, before de-duplication; and that day."""
    from train_model import build_risk_report  # heavy imports (lightgbm, optuna), so only on first use

    scored = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv", parse_dates=["date"])
    report = build_risk_report(scored.rename(columns={"pred": "predicted_units"}))
    latest = report["date"].max()
    rows = report[report["date"] == latest]
    products = pd.read_csv(DATA_DIR / "sampled_products.csv", dtype={"pack_size_value": str},
                           usecols=["sku_id", "product_name", "brand", "mrp", "pack_size_value", "pack_size_unit"])
    table = rows.merge(products, on="sku_id", how="left").rename(
        columns={"mrp": "price", "avg_daily_pred": "forecast_daily"})
    table["forecast_source"] = DIRECT
    return table, latest


def regional_rows():
    """Every forecast catalog product x city, before de-duplication; and the planning day."""
    inventory = pd.read_csv(
        DATA_DIR / "bigbasket_regional_inventory.csv", dtype={"pack_size_value": str},
        usecols=["region", "sku_id", "product_name", "brand", "pack_size_value", "pack_size_unit", "category",
                 "price", "forecast_daily", "forecast_source", "analog_ids", "analog_weights", "analog_price_adj",
                 "opening_stock", "days_of_cover", "days_until_delivery", "reorder_qty", "risk_status"])
    # stock is simulated for the morning of the first forecast day
    start = pd.read_csv(DATA_DIR / "bigbasket_regional_forecast.csv", usecols=["forecast_start"], nrows=1)
    return inventory.rename(columns={"region": "city"}), pd.Timestamp(start["forecast_start"].iloc[0])


def raw_rows(scope):
    """(rows, planning day) for a scope before de-duplication, built once and kept in memory."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    with _LOCK:
        if ("raw", scope) not in _TABLES:
            _TABLES["raw", scope] = model_rows() if scope == "model" else regional_rows()
        return _TABLES["raw", scope]


def table_for(scope):
    """The merged, de-duplicated table for a scope, built once and kept in memory."""
    with _LOCK:
        if scope not in _TABLES:
            _TABLES[scope] = finish(*raw_rows(scope))
        return _TABLES[scope]


def clear_cache():
    _TABLES.clear()


class UnknownCity(LookupError):
    pass


def _number(params, name):
    value = params.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None


def filter_rows(table, params, skip=()):
    """
    Rows matching the query parameters (strings; empty = no filter):
    city (or "all"), brand, category, min_price / max_price (inclusive),
    top (top N% sellers by units sold in 28 days, within each city),
    trend (rising / falling), max_cover (days), source (direct / analog),
    q (every word in the product name). Names in `skip` are ignored.
    """
    params = {key: value for key, value in params.items() if key not in skip}
    mask = pd.Series(True, index=table.index)
    city = params.get("city", "").strip()
    if city and city.casefold() != "all":
        city_mask = table["city"].str.casefold() == city.casefold()
        if not city_mask.any():
            raise UnknownCity(city)
        mask &= city_mask
    for name in ["brand", "category"]:
        value = params.get(name, "").strip()
        if value:
            mask &= table[name].str.casefold() == value.casefold()
    min_price, max_price = _number(params, "min_price"), _number(params, "max_price")
    if min_price is not None:
        mask &= table["price"] >= min_price
    if max_price is not None:
        mask &= table["price"] <= max_price
    top = _number(params, "top")
    if top is not None:
        if not 0 < top <= 100:
            raise ValueError("top must be a percentage between 0 and 100")
        mask &= table["seller_rank_pct"] < top / 100
    trend = params.get("trend", "").strip().casefold()
    if trend:
        if trend not in {"rising", "falling"}:
            raise ValueError("trend must be rising or falling")
        mask &= table["trend_pct"] > 0 if trend == "rising" else table["trend_pct"] < 0
    max_cover = _number(params, "max_cover")
    if max_cover is not None:
        mask &= table["days_of_cover"] <= max_cover
    source = params.get("source", "").strip().casefold()
    if source:
        if source not in {"direct", "analog"}:
            raise ValueError("source must be direct or analog")
        direct = table["forecast_source"] == DIRECT
        mask &= direct if source == "direct" else table["forecast_source"].str.startswith(ANALOG_PREFIX)
    names = table["product_name"].str.casefold()
    for term in params.get("q", "").casefold().split():
        mask &= names.str.contains(term, regex=False)
    return table[mask]


def sort_rows(rows, sort="", order=""):
    """
    Default: lowest cover first, then the fastest daily rate (sales, or the
    forecast for products without history). Otherwise `sort` (a
    SORT_COLUMNS key) in `order`, ties by cover; missing values go last.
    """
    if order and order not in {"asc", "desc"}:
        raise ValueError("order must be asc or desc")
    if sort and sort not in SORT_COLUMNS:
        raise ValueError(f"sort must be one of {', '.join(SORT_COLUMNS)}")
    rows = rows.assign(_rate=rows["avg_daily_sales"].fillna(rows["forecast_daily"]))
    tie_breaks = {"days_of_cover": True, "_rate": False, "sku_id": True}  # column -> ascending
    column = SORT_COLUMNS.get(sort, "days_of_cover")
    first = (order or ("asc" if sort in {"", "cover", "price"} else "desc")) == "asc"
    keys = {column: first, **{key: up for key, up in tie_breaks.items() if key != column}}
    return rows.sort_values(list(keys), ascending=list(keys.values()), na_position="last",
                            kind="stable").drop(columns="_rate")


def page(rows, offset="", limit=""):
    offset = int(_number({"offset": offset}, "offset") or 0)
    limit = int(_number({"limit": limit}, "limit") or PAGE_SIZE)
    return rows.iloc[max(0, offset):max(0, offset) + min(max(1, limit), MAX_PAGE_SIZE)]


def to_records(rows):
    rounded = rows.round({"price": 2, "avg_daily_sales": 2, "trend_pct": 1, "forecast_daily": 2,
                          "days_of_cover": 2, "seller_rank_pct": 3})
    return json.loads(rounded.to_json(orient="records"))  # NaN -> null, numpy -> plain types


def to_csv(rows):
    return rows[EXPORT_COLUMNS].round({"avg_daily_sales": 2, "trend_pct": 1, "forecast_daily": 2,
                                       "days_of_cover": 2}).to_csv(index=False)


def brand_summary(rows):
    """
    Per brand: units over 28 days, revenue (units x MRP), products, products
    at stockout risk, total reorder units and average cover, by units.
    Products without history count 28 x forecast daily (estimated);
    `estimated_products` says how many of a brand's products that is.
    """
    units = rows["units_28d"].fillna(rows["forecast_daily"] * HISTORY_DAYS)
    frame = rows.assign(units=units, revenue=units * rows["price"], stockout=rows["risk_status"] == "stockout",
                        estimated=~rows["has_history"])
    brands = frame.groupby("brand").agg(
        units_28d=("units", "sum"), revenue_28d=("revenue", "sum"), products=("sku_id", "count"),
        stockout_risk=("stockout", "sum"), reorder_units=("reorder_qty", "sum"), avg_cover=("days_of_cover", "mean"),
        estimated_products=("estimated", "sum"),
    ).reset_index()
    brands = brands.sort_values(["units_28d", "brand"], ascending=[False, True], kind="stable")
    return brands.round({"units_28d": 0, "revenue_28d": 0, "avg_cover": 2})
