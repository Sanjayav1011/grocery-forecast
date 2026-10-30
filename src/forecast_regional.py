"""
forecast_regional.py
----------------------
Demand forecasts for every catalog product x city, from the trained models.

  - Tracked products (the ~300 with POS history) are scored directly by the
    LightGBM models over the next LEAD_TIME_DAYS after the latest data date.
  - Every other catalog product mapped to a model category (data/category_map.csv)
    borrows from its K nearest tracked products in the same category, by
    standardized (log MRP, log pack size, log brand popularity), preferring
    the same BigBasket sub-category: an inverse-distance weighted mean of the
    analogs' forecasts, each adjusted for price with the same elasticity the
    generator uses: x exp(PRICE_ELASTICITY x (log_mrp_analog - log_mrp_product)),
    clipped to PRICE_ADJ_RANGE so a near-zero price cannot blow it up.
  - Unmapped products and non-food items mis-filed in food categories are
    excluded and counted.
  - Untracked free samples, BigBasket's own packaging supplies ("MCIP
    Consumables" boxes, bags, labels, tape in household) and products with an
    MRP below MIN_MRP get no analog forecast; they are listed in
    data/bigbasket_not_forecast.csv with reason "not forecast: free sample",
    "not forecast: packaging" or "not forecast: no price".

Also writes recent_daily, the trailing RECENT_DAYS mean demand (actual for
tracked products, analog-estimated otherwise), which the regional risk
report uses to simulate stock ordered before the forecast week.

Output: data/bigbasket_regional_forecast.csv, and
        data/bigbasket_tracked_daily_forecast.csv (each tracked product x
        city's forecast per day, the values forecast_daily averages)
Usage : python forecast_regional.py [--as-of YYYY-MM-DD] [--output PATH]
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_features import MAPPINGS_PATH, RAW_POS_PATH, apply_categorical_mappings, engineer_features
from config import DATA_DIR, FEATURE_COLS, LEAD_TIME_DAYS, MODELS_DIR, PRICE_ELASTICITY
from sample_products import SAMPLED_PRODUCTS_PATH, prepare_catalog

OUT_PATH = DATA_DIR / "bigbasket_regional_forecast.csv"
NOT_FORECAST_PATH = DATA_DIR / "bigbasket_not_forecast.csv"
DAILY_PATH = DATA_DIR / "bigbasket_tracked_daily_forecast.csv"
K_ANALOGS = 5
MIN_MRP = 1.0  # rupees; below this the price says nothing about demand
PRICE_ADJ_RANGE = (0.5, 2.0)
FREE_SAMPLE_PATTERN = r"(?i)\bfree\s*sample\b"
PACKAGING_PATTERN = r"(?i)\bmcip\s+consumables?\b"  # store packaging supplies, not products for sale
PACKAGING_CATEGORIES = {"household"}
HISTORY_DAYS = 400  # covers lag_364 plus the 28-day rolling windows
RECENT_DAYS = 28
DIRECT, ANALOG = "lightgbm_direct", f"lightgbm_analog_k{K_ANALOGS}"
OUTPUT_COLS = ["sku_id", "product_name", "brand", "pack_size_value", "pack_size_unit", "category", "city",
               "forecast_daily", "forecast_horizon",
               "forecast_source", "analog_ids", "analog_weights", "analog_price_adj", "recent_daily",
               "forecast_start", "forecast_end"]
_BOOSTERS = {}


def booster(segment):
    if segment not in _BOOSTERS:
        _BOOSTERS[segment] = lgb.Booster(model_file=str(MODELS_DIR / f"lgbm_{segment}.txt"))
    return _BOOSTERS[segment]


def score_tracked(as_of=None):
    """
    Per tracked product x city: mean daily LightGBM forecast for the
    LEAD_TIME_DAYS after as_of, and mean demand over the RECENT_DAYS up to as_of;
    also as_of and the per-day forecasts (daily_forecasts).
    """
    pos = pd.read_csv(RAW_POS_PATH, parse_dates=["date"])
    as_of = pos["date"].max() if as_of is None else pd.Timestamp(as_of)
    history = pos[(pos["date"] > as_of - pd.Timedelta(days=HISTORY_DAYS)) & (pos["date"] <= as_of)]
    daily, recent = daily_forecasts(history, as_of)
    forecast = daily.groupby(["sku_id", "city"])["prediction"].mean().rename("forecast_daily")
    return pd.concat([forecast, recent], axis=1).reset_index(), as_of, daily


def daily_forecasts(history, as_of):
    """
    LightGBM forecast for each of the LEAD_TIME_DAYS after as_of (columns
    sku_id, city, date, prediction) for every series in `history` (POS rows
    for the HISTORY_DAYS up to as_of), and each series' mean demand over the
    RECENT_DAYS up to as_of. Features are per series, so scoring one series
    alone gives the same numbers as scoring the whole history.
    """
    # Future rows carry what is known in advance (product, price); sales and
    # stock are unknown and promotions are assumed off. Every lag feature is
    # >= LEAD_TIME_DAYS, so these days only need history up to as_of.
    last = history[history["date"] == as_of]
    future = pd.concat([last.assign(date=as_of + pd.Timedelta(days=d)) for d in range(1, LEAD_TIME_DAYS + 1)])
    future = future.assign(units_sold=np.nan, opening_stock=np.nan, true_demand=np.nan,
                           days_until_delivery=np.nan, on_promotion=0, stockout_flag=0)

    df = engineer_features(pd.concat([history, future], ignore_index=True))
    recent = (df[(df["date"] > as_of - pd.Timedelta(days=RECENT_DAYS)) & (df["date"] <= as_of)]
              .groupby(["sku_id", "city"])["demand"].mean().rename("recent_daily"))
    df = df[df["date"] > as_of].copy()
    df = apply_categorical_mappings(df, json.loads(MAPPINGS_PATH.read_text()))
    df["prediction"] = np.nan
    for segment, is_perishable in [("perishable", 1), ("nonperishable", 0)]:
        rows = df["is_perishable"] == is_perishable
        if rows.any():
            df.loc[rows, "prediction"] = np.clip(booster(segment).predict(df.loc[rows, FEATURE_COLS]), 0, None)
    return df[["sku_id", "city", "date", "prediction"]].reset_index(drop=True), recent


def standardized_attributes(products, reference):
    """log MRP, log pack size and log brand popularity, scaled by the tracked products' mean/std."""
    attrs = pd.DataFrame({
        "log_mrp": np.log(products["mrp_filled"]),
        "log_pack": np.log1p(products["pack_filled"]),
        "log_brand": np.log(products["brand_catalog_count"].clip(lower=1)),
    }, index=products.index)
    ref = attrs.loc[reference]
    return (attrs - ref.mean()) / ref.std(ddof=0)


def analog_forecasts(products, tracked_forecast, value_cols=("forecast_daily", "recent_daily")):
    """
    Each value column per product x city: tracked products use their own
    values, the rest the price-adjusted inverse-distance mean of K analogs.
    """
    cities = sorted(tracked_forecast["city"].unique())
    by_sku = {col: tracked_forecast.pivot(index="sku_id", columns="city", values=col)[cities] for col in value_cols}
    is_tracked = products["sku_id"].isin(by_sku[value_cols[0]].index)
    attrs = standardized_attributes(products, products.index[is_tracked])

    values = {col: np.empty((len(products), len(cities))) for col in value_cols}
    analog_ids = np.empty(len(products), dtype=object)
    analog_weights = np.full(len(products), "", dtype=object)  # normalised inverse-distance weights
    analog_price_adj = np.full(len(products), "", dtype=object)
    position = {index: i for i, index in enumerate(products.index)}
    for category, rows in products.groupby("category"):
        pool = rows[is_tracked.loc[rows.index]]
        pool_x, rows_x = attrs.loc[pool.index].to_numpy(), attrs.loc[rows.index].to_numpy()
        distance = np.sqrt(((rows_x[:, None, :] - pool_x[None, :, :]) ** 2).sum(axis=2))
        # same sub-category first (nearest of those), then the rest of the category
        other_sub = rows["bb_subcategory"].to_numpy()[:, None] != pool["bb_subcategory"].to_numpy()[None, :]
        order = np.argsort(other_sub * 1e6 + distance, axis=1, kind="stable")[:, :K_ANALOGS]
        nearest = np.take_along_axis(distance, order, axis=1)
        weights = 1.0 / (nearest + 1e-3)
        pool_log_mrp = np.log(pool["mrp_filled"].to_numpy())[order]
        price_adj = np.exp(PRICE_ELASTICITY * (pool_log_mrp - np.log(rows["mrp_filled"].to_numpy())[:, None]))
        price_adj = np.clip(price_adj, *PRICE_ADJ_RANGE)
        targets = [position[index] for index in rows.index]
        for col in value_cols:
            pool_values = by_sku[col].loc[pool["sku_id"]].to_numpy()[order]  # (rows, K, cities)
            values[col][targets] = ((weights[:, :, None] * price_adj[:, :, None] * pool_values).sum(axis=1)
                                    / weights.sum(axis=1)[:, None])
        analog_ids[targets] = [";".join(ids) for ids in pool["sku_id"].to_numpy()[order]]
        joined = lambda matrix: [";".join(f"{value:.4f}" for value in row) for row in matrix]
        analog_weights[targets] = joined(weights / weights.sum(axis=1, keepdims=True))
        analog_price_adj[targets] = joined(price_adj)

    tracked_rows = np.flatnonzero(is_tracked.to_numpy())
    for col in value_cols:
        values[col][tracked_rows] = by_sku[col].loc[products["sku_id"].iloc[tracked_rows]].to_numpy()
    analog_ids[tracked_rows] = ""
    analog_weights[tracked_rows] = analog_price_adj[tracked_rows] = ""
    source = np.where(is_tracked, DIRECT, ANALOG)

    out = pd.DataFrame({
        "sku_id": np.repeat(products["sku_id"].to_numpy(), len(cities)),
        "product_name": np.repeat(products["SKU Name"].to_numpy(), len(cities)),
        "brand": np.repeat(products["brand"].to_numpy(), len(cities)),
        "pack_size_value": np.repeat(products["pack_size_value"].to_numpy(), len(cities)),
        "pack_size_unit": np.repeat(products["pack_size_unit"].to_numpy(), len(cities)),
        "category": np.repeat(products["category"].to_numpy(), len(cities)),
        "city": np.tile(cities, len(products)),
        "forecast_source": np.repeat(source, len(cities)),
        "analog_ids": np.repeat(analog_ids, len(cities)),
        "analog_weights": np.repeat(analog_weights, len(cities)),
        "analog_price_adj": np.repeat(analog_price_adj, len(cities)),
        **{col: values[col].reshape(-1).round(3) for col in value_cols},
    })
    out["forecast_horizon"] = (out["forecast_daily"] * LEAD_TIME_DAYS).round(2)
    return out


def not_forecast_reasons(products, tracked_ids):
    """
    "not forecast: ..." reason per untracked product that must not be
    forecast by analogs, empty string otherwise. Free samples and packaging
    are matched by name (packaging only in PACKAGING_CATEGORIES), whatever
    their price. "No price" uses the parsed MRP, so a missing MRP (filled
    with the category median) is not "no price", but an MRP of 0 is.
    """
    untracked = ~products["sku_id"].isin(tracked_ids)
    name = products["SKU Name"].fillna("")
    free_sample = name.str.contains(FREE_SAMPLE_PATTERN)
    packaging = name.str.contains(PACKAGING_PATTERN) & products["category"].isin(PACKAGING_CATEGORIES)
    no_price = products["mrp"] < MIN_MRP
    reasons = np.select([untracked & free_sample, untracked & packaging, untracked & no_price],
                        ["not forecast: free sample", "not forecast: packaging", "not forecast: no price"],
                        default="")
    return pd.Series(reasons, index=products.index)


def catalog_products():
    """
    Mapped, food-checked catalog products with filled MRP and pack size, and
    the products left out of analog forecasting (with a reason); prints
    exclusion counts.
    """
    catalog = prepare_catalog().rename(columns={"model_category": "category", "Sub-Category": "bb_subcategory"})
    unmapped = catalog["category"].isna()
    excluded = catalog["excluded_non_food"]
    print(f"Catalog: {len(catalog):,} products; {unmapped.sum():,} unmapped (no model category) and "
          f"{(excluded & ~unmapped).sum():,} non-food items in food categories excluded")
    products = catalog[~unmapped & ~excluded].copy()
    # tracked products must line up with the generator's view of them
    tracked = pd.read_csv(SAMPLED_PRODUCTS_PATH, usecols=["sku_id"])["sku_id"]
    missing = set(tracked) - set(products["sku_id"])
    assert not missing, f"tracked products missing from the catalog: {sorted(missing)[:5]}"

    by_category = products.groupby("category")
    products["mrp_filled"] = products["mrp"].where(products["mrp"] > 0).fillna(by_category["mrp"].transform("median"))
    products["pack_filled"] = products["pack_size_value"].fillna(by_category["pack_size_value"].transform("median"))

    products["reason"] = not_forecast_reasons(products, tracked)
    skipped = products[products["reason"] != ""]
    not_forecast = skipped.rename(columns={"SKU Name": "product_name"})[
        ["sku_id", "product_name", "brand", "category", "mrp", "reason"]]
    for reason, count in not_forecast["reason"].value_counts().items():
        print(f"  {count:,} {reason}")
    products = products.drop(index=skipped.index, columns="reason")
    return products, not_forecast


def main():
    parser = argparse.ArgumentParser(description="Forecast every mapped catalog product x city.")
    parser.add_argument("--as-of", help="forecast from this date (default: latest POS date)")
    parser.add_argument("--output", default=str(OUT_PATH))
    args = parser.parse_args()

    tracked_forecast, as_of, daily = score_tracked(args.as_of)
    daily.to_csv(DAILY_PATH, index=False, date_format="%Y-%m-%d")
    products, not_forecast = catalog_products()
    forecast = analog_forecasts(products, tracked_forecast)
    not_forecast.to_csv(NOT_FORECAST_PATH, index=False)
    forecast["forecast_start"] = (as_of + pd.Timedelta(days=1)).date().isoformat()
    forecast["forecast_end"] = (as_of + pd.Timedelta(days=LEAD_TIME_DAYS)).date().isoformat()
    forecast[OUTPUT_COLS].to_csv(args.output, index=False)
    counts = forecast.drop_duplicates("sku_id")["forecast_source"].value_counts()
    print(f"Saved {len(forecast):,} product x city forecasts ({counts.get(DIRECT, 0):,} tracked, "
          f"{counts.get(ANALOG, 0):,} analog products) for {forecast['forecast_start'].iloc[0]}.."
          f"{forecast['forecast_end'].iloc[0]} to {args.output}")
    print(f"Saved {len(daily):,} tracked daily forecasts to {DAILY_PATH}")
    print(f"Saved {len(not_forecast):,} products not forecast (free sample / packaging / no price) "
          f"to {NOT_FORECAST_PATH}")


if __name__ == "__main__":
    main()
