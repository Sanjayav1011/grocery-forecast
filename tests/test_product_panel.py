"""
Product detail panel: GET /api/product (src/product_detail.py). Uses the
hand-made data set from test_dashboard_filters.py (planning day 2026-08-26,
so "today" is 2026-08-25), plus one check on the generated data.
"""

import json
import urllib.error

import pandas as pd
import pytest

import forecast_regional
import product_detail
import stockout_view
from config import DATA_DIR, LEAD_TIME_DAYS
from test_dashboard_filters import fake_data, get, server  # noqa: F401  (pytest fixtures)

TODAY = pd.Timestamp("2026-08-25")


def fake_daily_forecasts(history, as_of):
    """Stand-in for the LightGBM scorer: 1, 2, ... 7 units on the days after as_of."""
    days = pd.date_range(as_of + pd.Timedelta(days=1), periods=LEAD_TIME_DAYS)
    sku, city = history["sku_id"].iloc[0], history["city"].iloc[0]
    return pd.DataFrame({"sku_id": sku, "city": city, "date": days, "prediction": range(1, LEAD_TIME_DAYS + 1)}), None


def test_tracked_product_has_60_days_of_history_and_7_forecast_days(server, monkeypatch):
    monkeypatch.setattr(forecast_regional, "daily_forecasts", fake_daily_forecasts)
    product = get(server, "/api/product", sku_id="T1")  # scope regional, city Delhi
    history, forecast = product["history"], product["forecast"]
    assert product["today"] == "2026-08-25" and product["details"]["forecast_source"] == "lightgbm_direct"
    assert len(history) == product_detail.CHART_DAYS == 60
    assert [h["date"] for h in history] == [d.date().isoformat() for d in pd.date_range(end=TODAY, periods=60)]
    assert history[-1] == {"date": "2026-08-25", "units_sold": 10, "true_demand": 10.5, "stockout": False}
    assert [h["date"] for h in history if h["stockout"]] == ["2026-08-16"]  # age 10 in the fixture
    assert [f["date"] for f in forecast] == [d.date().isoformat() for d in pd.date_range("2026-08-26", periods=7)]
    assert [f["units"] for f in forecast] == [1, 2, 3, 4, 5, 6, 7]
    assert product["analogs"] == []
    assert product["stock"] == {"units": 18, "date": "2026-08-26", "simulated": True}
    details = product["details"]
    assert (details["brand"], details["price"], details["days_of_cover"], details["reorder_qty"]) == ("Amul", 25.0, 1.5, 108)
    # festivals in the window, from the festival calendar
    assert {"date": "2026-08-15", "name": "Independence Day"} in product["festivals"]
    assert all("2026-06-27" <= f["date"] <= "2026-09-01" for f in product["festivals"])
    assert any("simulated" in note for note in product["notes"])


def test_analog_product_lists_its_5_analogs_and_a_flat_forecast(server):
    product = get(server, "/api/product", sku_id="A1")
    assert product["history"] == []
    assert [f["units"] for f in product["forecast"]] == [30.0] * 7
    analogs = product["analogs"]
    assert len(analogs) == 5
    assert [a["sku_id"] for a in analogs] == ["T1", "T2", "T3", "T4", "T1"]
    assert [a["weight"] for a in analogs] == [0.4, 0.3, 0.1, 0.1, 0.1]
    assert [a["price_adjustment"] for a in analogs] == [1.0, 1.2, 0.8, 2.0, 0.5]
    assert analogs[0] == {"sku_id": "T1", "product_name": "Amul Lassi 200 ml", "weight": 0.4,
                          "price_adjustment": 1.0, "forecast_daily": 12.0}  # Delhi's T1, not Mumbai's
    assert any("estimated" in note for note in product["notes"])


def test_duplicate_sku_hidden_from_tables_still_opens(server):
    # A4 is dropped from the de-duplicated tables, but the occasion lists may still link to it
    assert get(server, "/api/product", sku_id="A4")["details"]["forecast_daily"] == 99.0


def test_model_scope_details_come_from_the_risk_report(server, fake_data, monkeypatch):
    monkeypatch.setattr(forecast_regional, "daily_forecasts", fake_daily_forecasts)
    pd.DataFrame([{"date": "2026-08-25", "city": "Delhi", "sku_id": "T1", "category": "dairy", "pred": 10.0,
                   "opening_stock": 15, "days_until_delivery": 3}]).to_csv(fake_data / "test_predictions.csv", index=False)
    pd.DataFrame([("T1", "Amul Lassi 200 ml", "Amul", 25.0, "200.0", "ml")],
                 columns=["sku_id", "product_name", "brand", "mrp", "pack_size_value", "pack_size_unit"]).to_csv(
        fake_data / "sampled_products.csv", index=False)
    product = get(server, "/api/product", sku_id="T1", scope="model")
    assert product["stock"] == {"units": 15, "date": "2026-08-25", "simulated": True}
    assert product["details"]["risk_status"] == "stockout" and product["details"]["days_of_cover"] == 1.5


@pytest.mark.parametrize("params", [{"sku_id": "NOPE"}, {"sku_id": "T1", "city": "Atlantis"}, {"sku_id": ""}])
def test_unknown_product_is_404_json(server, params):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(server, "/api/product", **params)
    assert error.value.code == 404
    assert error.value.headers["Content-Type"] == "application/json"
    assert json.loads(error.value.read())["error"] == "Product not found"


def test_bad_scope_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(server, "/api/product", sku_id="T1", scope="everything")
    assert error.value.code == 400


def test_generated_forecast_line_averages_to_the_regional_forecast():
    """On the real data: the panel's per-day LightGBM line is what the regional forecast averages."""
    needed = [DATA_DIR / "bigbasket_regional_inventory.csv", DATA_DIR / "pos_synthetic.csv",
              forecast_regional.MODELS_DIR / "lgbm_perishable.txt"]
    if not all(path.exists() for path in needed):
        pytest.skip("pipeline outputs not generated; run run_pipeline.py first")
    stockout_view.clear_cache()
    product_detail.clear_cache()
    rows, _ = stockout_view.raw_rows("regional")
    tracked = rows[rows["forecast_source"] == "lightgbm_direct"].iloc[0]
    product = product_detail.product_detail(tracked["city"], tracked["sku_id"])
    assert len(product["history"]) == 60 and len(product["forecast"]) == 7
    mean = sum(f["units"] for f in product["forecast"]) / 7
    assert mean == pytest.approx(tracked["forecast_daily"], abs=2e-3)
    analog = rows[rows["forecast_source"] != "lightgbm_direct"].iloc[0]
    assert len(product_detail.product_detail(analog["city"], analog["sku_id"])["analogs"]) == 5
