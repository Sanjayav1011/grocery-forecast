"""
serving_risk_report.py
------------------------
Model risk report for the serving date (config.SERVING_DATE, the last day
of data): every tracked product x city, with stock counted on the morning
of the first forecast day (data/serving_opening_stock.csv, simulated by
generate_synthetic_data.extend) against its forecast for that day and the
next LEAD_TIME_DAYS - 1 (data/bigbasket_tracked_daily_forecast.csv, from
the saved models), classified by the same inventory.classify_risk rule as
the evaluation report.

Writes outputs/serving_stockout_risk_report.csv and
serving_overstock_risk_report.csv, with the evaluation reports' columns.
The evaluation reports (outputs/stockout_risk_report.csv /
overstock_risk_report.csv, 1 Apr - 25 Aug 2026, from train_model.py) are
left as they are. The dashboard's model risk report and the vendor alerts
read the serving reports.
"""

import pandas as pd

from config import DATA_DIR, OUTPUTS_DIR
from inventory import classify_risk

STOCKOUT_PATH = OUTPUTS_DIR / "serving_stockout_risk_report.csv"
OVERSTOCK_PATH = OUTPUTS_DIR / "serving_overstock_risk_report.csv"
# the columns of train_model.REPORT_COLS (checked by tests/test_serving.py)
REPORT_COLS = ["date", "city", "sku_id", "product_name", "brand", "category", "predicted_units", "opening_stock",
               "days_of_cover", "horizon_demand", "avg_daily_pred", "days_until_delivery", "shelf_life_days",
               "reorder_qty"]


def build_report(data_dir=DATA_DIR):
    """Every tracked product x city on the first forecast day, with risk_status (all statuses)."""
    daily = pd.read_csv(data_dir / "bigbasket_tracked_daily_forecast.csv", parse_dates=["date"])
    stock = pd.read_csv(data_dir / "serving_opening_stock.csv", parse_dates=["date"])
    products = pd.read_csv(data_dir / "sampled_products.csv", usecols=["sku_id", "product_name", "brand", "category"])
    forecast = daily.sort_values("date").groupby(["city", "sku_id"]).agg(
        date=("date", "min"), predicted_units=("prediction", "first"), avg_daily_pred=("prediction", "mean")).reset_index()
    if set(forecast["date"]) != set(stock["date"]):
        raise ValueError(f"serving stock is for {sorted(set(stock['date'].dt.date))}, the forecast starts "
                         f"{sorted(set(forecast['date'].dt.date))}; re-run generate_synthetic_data.py and forecast_regional.py")
    report = (forecast.merge(stock[["city", "sku_id", "opening_stock", "days_until_delivery"]], on=["city", "sku_id"])
              .merge(products, on="sku_id", how="left"))
    return classify_risk(report)


def main():
    report = build_report()
    stockout = report[report["risk_status"] == "stockout"].sort_values(["date", "days_of_cover"])
    overstock = report[report["risk_status"] == "overstock"].sort_values(["date", "days_of_cover"], ascending=[True, False])
    stockout[REPORT_COLS].to_csv(STOCKOUT_PATH, index=False, date_format="%Y-%m-%d")
    overstock[REPORT_COLS].to_csv(OVERSTOCK_PATH, index=False, date_format="%Y-%m-%d")
    day = report["date"].iloc[0].date()
    print(f"Serving risk report for {day} (stock counted that morning, forecast for the next 7 days): "
          f"{len(stockout)} stockout, {len(overstock)} overstock of {len(report):,} tracked product x city rows")
    print(report.groupby("city")["risk_status"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
