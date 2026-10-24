"""
Pipeline checks. Run from the project root after `python run_pipeline.py`:

    python -m pytest

Tests that need generated data skip (rather than fail) when it's missing.
"""

import json
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import DATA_DIR, FEATURE_COLS, LEAD_TIME_DAYS, MODELS_DIR, SHELF_LIFE_DAYS  # noqa: E402
from festival_calendar import build_calendar_table, festival_df  # noqa: E402
from inventory import classify_risk  # noqa: E402

RANDOM_COLUMNS = {"history_days", "avg_daily_sales_90d", "avg_daily_sales_30d", "avg_daily_sales_7d", "sales_trend"}


def require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not generated; run run_pipeline.py first")
    return path


def test_feature_cols_are_leak_free():
    assert not {"lag_1", "roll_mean_7", "store_id_code"} & set(FEATURE_COLS)
    for col in FEATURE_COLS:
        lag = re.fullmatch(r"lag_(\d+)", col) or re.search(r"_lag(\d+)$", col)
        if lag:
            assert int(lag.group(1)) >= LEAD_TIME_DAYS, f"{col} looks back less than the lead time"


def test_festival_calendar_covers_2021_to_2026():
    years = festival_df()["date"].dt.year
    for year in range(2021, 2027):
        assert (years == year).sum() >= 10, f"too few festivals in {year}"
    cal = build_calendar_table("2021-08-26", "2026-12-31")
    assert cal["days_until_next_festival"].between(0, 60).all()
    assert cal["days_since_last_festival"].between(0, 60).all()


def test_demand_is_never_below_units_sold():
    features = pd.read_csv(require(DATA_DIR / "features.csv"), usecols=["units_sold", "demand"])
    assert (features["demand"] >= features["units_sold"]).all()


def test_categorical_mappings_cover_every_model_category():
    mappings = json.loads(require(MODELS_DIR / "categorical_mappings.json").read_text())
    assert {"city", "category", "sku_id", "brand"} <= set(mappings)
    assert set(mappings["category"]) == set(SHELF_LIFE_DAYS)


def test_regional_forecast_is_model_based():
    forecast = pd.read_csv(require(DATA_DIR / "bigbasket_regional_forecast.csv"))
    assert not RANDOM_COLUMNS & set(forecast.columns)
    assert set(forecast["forecast_source"]) <= {"lightgbm_direct", "lightgbm_analog_k5"}
    assert (forecast["forecast_daily"] >= 0).all()

    inventory = pd.read_csv(require(DATA_DIR / "bigbasket_regional_inventory.csv"),
                            usecols=lambda col: col not in {"analog_ids", "product_url", "image_url"})
    assert not RANDOM_COLUMNS & set(inventory.columns)
    assert set(inventory["stock_source"]) == {"simulated"}


def test_risk_rule_on_a_hand_made_frame():
    frame = pd.DataFrame({
        "category":            ["snacks", "snacks", "dairy", "dairy"],
        "avg_daily_pred":      [10.0,     10.0,     10.0,    0.0],
        "opening_stock":       [30,       100,      200,     5],
        "days_until_delivery": [5,        7,        3,       3],
    })
    risk = classify_risk(frame, horizon=7)
    # 3 days of cover < 5 until delivery; 10 days is fine for snacks;
    # 20 days > 7 + 2 + 3 (dairy shelf life); zero forecast -> cover 5 / 0.1 = 50 days
    assert list(risk["risk_status"]) == ["stockout", "normal", "overstock", "overstock"]
    assert list(risk["days_of_cover"]) == [3.0, 10.0, 20.0, 50.0]
    # reorder = ceil(max(0, 7 x 10 + 2 x 10 - stock))
    assert list(risk["reorder_qty"]) == [60, 0, 0, 0]


def test_dashboard_unknown_dataset_is_json_404():
    import dashboard

    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/dataset?name=does_not_exist.csv"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url, timeout=10)
        assert error.value.code == 404
        assert error.value.headers["Content-Type"] == "application/json"
        assert json.loads(error.value.read()) == {"error": "Dataset not found", "name": "does_not_exist.csv"}
    finally:
        server.shutdown()
        server.server_close()
