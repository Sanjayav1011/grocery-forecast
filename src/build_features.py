"""
build_features.py
-------------------
Joins raw POS data (real or synthetic) with the 5-year festival calendar and
engineers the feature set used for training.

Input : data/pos_synthetic.csv   (swap for your real POS export -- same columns)
Output: data/features.csv

>>> TO USE YOUR REAL DATA: point RAW_POS_PATH at your export. As long as it
>>> has the columns date, city, store_id, sku_id, category, is_perishable,
>>> unit_price, on_promotion, units_sold, opening_stock -- this script needs
>>> no changes. <<<
"""

import json

import pandas as pd
import numpy as np
from config import DATA_DIR, LEAD_TIME_DAYS, MODELS_DIR
from festival_calendar import build_calendar_table, festival_uplift
from sample_products import SAMPLED_PRODUCTS_PATH

RAW_POS_PATH = DATA_DIR / "pos_synthetic.csv"
OUT_PATH = DATA_DIR / "features.csv"
MAPPINGS_PATH = MODELS_DIR / "categorical_mappings.json"
CATEGORICAL_COLS = ["city", "category", "sku_id", "brand"]
TOP_BRANDS = 30  # brands outside the 30 most popular share one "other" code


def add_product_features(df):
    """
    Product attributes from the product master (data/sampled_products.csv):
    log_mrp, pack_size_value, and brand grouped to the TOP_BRANDS most
    popular brands (by catalog product count), the rest as "other".
    """
    products = pd.read_csv(SAMPLED_PRODUCTS_PATH, usecols=["sku_id", "brand", "mrp", "pack_size_value", "brand_catalog_count"])
    popularity = products.drop_duplicates("brand").sort_values(["brand_catalog_count", "brand"], ascending=[False, True])
    top_brands = set(popularity["brand"].head(TOP_BRANDS))
    products["brand"] = products["brand"].where(products["brand"].isin(top_brands), "other")
    df = df.merge(products.drop(columns="brand_catalog_count"), on="sku_id", how="left")
    df["log_mrp"] = np.log(df["mrp"].fillna(df["unit_price"]))
    return df.drop(columns="mrp")


def add_lag_and_rolling_features(df, group_cols, target_col="demand", lead=LEAD_TIME_DAYS):
    """
    Every feature is shifted by at least `lead` days, so a forecast made
    `lead` days ahead never sees sales it wouldn't have yet (no lag_1 /
    shift(1) leakage).
    """
    df = df.sort_values(group_cols + ["date"]).copy()
    g = df.groupby(group_cols)[target_col]

    for lag in (lead, 2 * lead, 3 * lead, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    for window in (7, 14, 28):
        df[f"roll_mean_{window}_lag{lead}"] = g.transform(
            lambda s: s.shift(lead).rolling(window, min_periods=1).mean())
    df[f"roll_std_14_lag{lead}"] = g.transform(lambda s: s.shift(lead).rolling(14, min_periods=2).std())
    return df


def build_categorical_mappings(df, cols=CATEGORICAL_COLS):
    """Stable value -> code mapping (sorted), saved so inference uses the same codes."""
    return {col: {value: code for code, value in enumerate(sorted(df[col].astype(str).unique()))} for col in cols}


def apply_categorical_mappings(df, mappings):
    """Adds <col>_code columns; values not seen in training map to -1."""
    for col, mapping in mappings.items():
        df[col + "_code"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df


def add_unconstrained_demand(df, group_cols):
    """
    units_sold is censored on stockout days (sales can't exceed stock). Build
    a `demand` target that un-censors those days: max(units_sold, baseline x
    festival uplift), where baseline is the trailing 14-day mean of
    festival-adjusted units_sold (units_sold / uplift) over NON-stockout days
    only (shifted 1, so today's sale isn't used). Adjusting first matters
    because stockouts cluster in festival run-ups, when the trailing window
    already contains part of the spike. Non-stockout days keep demand = units_sold.
    """
    df = df.sort_values(group_cols + ["date"]).copy()
    # festival_uplift() returns 1 + uplift, the same signal the calendar uses
    uplift = pd.Series(1.0, index=df.index)
    for category, idx in df.groupby("category").groups.items():
        uplift.loc[idx] = festival_uplift(df.loc[idx, "date"], category)
    clean_sales = (df["units_sold"] / uplift).where(df["stockout_flag"] == 0)
    baseline = clean_sales.groupby([df[c] for c in group_cols]).transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).mean().ffill()
    )
    estimate = (baseline * uplift).fillna(df["units_sold"])
    stockout = df["stockout_flag"] == 1
    df["demand"] = df["units_sold"].astype(float)
    df.loc[stockout, "demand"] = np.maximum(df.loc[stockout, "units_sold"], estimate[stockout]).round(2)
    return df


def report_demand_recovery(df):
    """Synthetic data only: how close is the rebuilt `demand` to `true_demand` on stockout days?"""
    if "true_demand" not in df.columns:
        return
    so = df[df["stockout_flag"] == 1]
    true = so["true_demand"]
    for label, est in [("units_sold (censored)", so["units_sold"]), ("demand (rebuilt)", so["demand"])]:
        err = est - true
        print(f"  {label:<22} MAE={err.abs().mean():.2f}  bias={err.mean():+.2f}  "
              f"MAPE={(err.abs() / true).mean() * 100:.1f}%  share of true demand={est.sum() / true.sum() * 100:.1f}%")


def engineer_features(pos, verbose=False):
    """
    All features for a POS frame (calendar, product, demand target, lags),
    before categorical encoding. Also used by forecast_regional.py to score
    future dates appended to recent history.
    """
    cal = build_calendar_table(pos["date"].min(), pos["date"].max())

    df = pos.merge(cal, on="date", how="left")
    df = add_product_features(df)

    # calendar / seasonality features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    df = add_unconstrained_demand(df, group_cols=["city", "sku_id"])
    if verbose:
        print("Stockout-day demand recovery vs true_demand:")
        report_demand_recovery(df)

    # lag/rolling features, computed PER city+sku so one SKU's history
    # never leaks into another's, and per-city demand levels stay separate
    return add_lag_and_rolling_features(df, group_cols=["city", "sku_id"])


def build():
    pos = pd.read_csv(RAW_POS_PATH, parse_dates=["date"])
    df = engineer_features(pos, verbose=True)

    # Explicit, saved encodings so codes don't shift when the data changes;
    # LightGBM receives these columns as categorical features during training.
    mappings = build_categorical_mappings(df)
    MAPPINGS_PATH.write_text(json.dumps(mappings, indent=2))
    df = apply_categorical_mappings(df, mappings)

    # drop the first weeks per SKU/city where the short-history features are
    # NaN; lag_364 stays NaN for the first year (LightGBM handles missing values)
    df = df.dropna(subset=[f"lag_{3 * LEAD_TIME_DAYS}", f"roll_std_14_lag{LEAD_TIME_DAYS}"]).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} feature rows to {OUT_PATH}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns ({len(df.columns)}):", list(df.columns))
    return df


if __name__ == "__main__":
    build()
