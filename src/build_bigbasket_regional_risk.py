"""Build the four-region inventory-risk layer for the BigBasket catalog.

Demand comes from the model forecasts in data/bigbasket_regional_forecast.csv
(forecast_regional.py: LightGBM direct for tracked products, K-nearest
analogs for the rest). The archive has no stock feed, so opening stock is
SIMULATED (stock_source = "simulated") by running the same replenishment
policy as the synthetic POS data (inventory.simulate_inventory) over a
warm-up period at each product's RECENT demand level (recent_daily) with
the generator's day-to-day noise -- i.e. stock ordered from history, which
is then checked against the forecast for the coming week. Risk uses the same rule as the model risk report
(inventory.classify_risk).
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED
from forecast_regional import OUT_PATH as FORECAST_PATH
from inventory import classify_risk, simulate_inventory
from sample_products import load_catalog

WARMUP_DAYS = 28   # enough for several delivery cycles before "today"
DEMAND_NOISE = 0.12  # day-to-day demand noise, as in generate_synthetic_data.py

REPORT_COLS = [
    "region", "sku_id", "category", "subcategory", "model_category", "product_name", "brand",
    "pack_size_value", "pack_size_unit", "price",
    "forecast_units", "forecast_daily", "forecast_horizon", "forecast_source", "analog_ids",
    "opening_stock", "stock_source", "days_until_delivery", "days_of_cover", "shelf_life_days",
    "reorder_qty", "risk_status", "ean_code", "product_url", "image_url", "source_file",
]


def simulate_opening_stock(forecast, rng):
    """Opening stock and days until the next delivery on the first forecast day, per row."""
    daily = forecast["recent_daily"].to_numpy()
    demand = np.clip(daily[:, None] * rng.normal(1.0, DEMAND_NOISE, (len(forecast), WARMUP_DAYS + 1)), 0, None)
    opening, _, days_until_delivery = simulate_inventory(demand, daily, forecast["model_category"], rng)
    return opening[:, -1].round().astype(int), days_until_delivery[:, -1]


def build():
    forecast = pd.read_csv(FORECAST_PATH).rename(columns={"city": "region", "category": "model_category"})
    catalog = load_catalog()[["sku_id", "Category", "Sub-Category", "MRP", "EAN Code", "Product Link", "Image Link", "Source File"]]
    report = forecast.merge(catalog, on="sku_id", how="left")

    rng = np.random.default_rng(RANDOM_SEED)
    report["opening_stock"], report["days_until_delivery"] = simulate_opening_stock(report, rng)
    report["stock_source"] = "simulated"
    report["avg_daily_pred"] = report["forecast_daily"]
    report = classify_risk(report, category_col="model_category")

    report["category"] = report["Category"].fillna("Other")
    report["subcategory"] = report["Sub-Category"].fillna("")
    report["price"] = pd.to_numeric(report["MRP"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce").round(2)
    report["forecast_units"] = report["forecast_daily"].round(1)
    report["days_of_cover"] = report["days_of_cover"].round(2)
    report = report.rename(columns={"EAN Code": "ean_code", "Product Link": "product_url",
                                    "Image Link": "image_url", "Source File": "source_file"})[REPORT_COLS]

    report.to_csv(DATA_DIR / "bigbasket_regional_inventory.csv", index=False)
    report[report["risk_status"] == "stockout"].to_csv(OUTPUTS_DIR / "bigbasket_stockout_risk_report.csv", index=False)
    report[report["risk_status"] == "overstock"].to_csv(OUTPUTS_DIR / "bigbasket_overstock_risk_report.csv", index=False)
    print(f"Saved {len(report):,} region-product rows across {report['region'].nunique()} regions")
    split = report.groupby("region")["risk_status"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(split.reindex(columns=["stockout", "normal", "overstock"], fill_value=0).round(1).to_string())


if __name__ == "__main__":
    build()
