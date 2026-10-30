"""
Forecast explanations (src/explain_forecast.py, GET /api/explain): LightGBM
SHAP values on the saved models add up to the raw prediction, every model
feature belongs to exactly one driver, the units/day conversion adds up to
the forecast, and the endpoint (on the hand-made data set from
test_dashboard_filters.py) explains tracked and analog products and answers
404 JSON for an unknown SKU.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

import explain_forecast
from config import DATA_DIR, FEATURE_COLS, MODELS_DIR
from explain_forecast import DRIVERS, FEATURE_LABELS, by_driver, contributions, to_units
from test_dashboard_filters import fake_data, get, server  # noqa: F401  (pytest fixtures)


def test_every_feature_is_in_exactly_one_driver():
    grouped = [feature for features in DRIVERS.values() for feature in features]
    assert len(grouped) == len(set(grouped))
    assert sorted(grouped) == sorted(FEATURE_COLS)
    assert set(FEATURE_LABELS) == set(FEATURE_COLS)


def test_units_add_up_to_the_forecast_and_keep_their_sign():
    contrib = pd.DataFrame({"a": [0.5, 0.0, 0.3], "b": [-0.2, 0.0, -0.3]})  # the last row: no net change
    units, baseline, forecast = to_units(contrib, np.log([20.0, 20.0, 20.0]))
    assert np.allclose(baseline + units.sum(axis=1), forecast)
    assert np.allclose(forecast, 20 * np.exp([0.3, 0.0, 0.0]))
    assert (np.sign(units) == np.sign(contrib)).all().all()
    assert units.loc[2, "a"] == pytest.approx(0.3 * 20)  # zero log gap: scaled by the baseline


@pytest.mark.parametrize("segment,is_perishable", list(explain_forecast.SEGMENTS.items()))
def test_saved_model_contributions_plus_base_equal_the_raw_prediction(segment, is_perishable):
    features = DATA_DIR / "features.csv"
    if not features.exists() or not (MODELS_DIR / f"lgbm_{segment}.txt").exists():
        pytest.skip("features or saved models not generated; run run_pipeline.py first")
    for chunk in pd.read_csv(features, usecols=["is_perishable"] + FEATURE_COLS, chunksize=50_000):
        rows = chunk[chunk["is_perishable"] == is_perishable].head(500)
        if len(rows) == 500:
            break
    model = explain_forecast.booster(segment)
    contrib, base = contributions(model, rows)
    raw = model.predict(rows[FEATURE_COLS], raw_score=True)
    assert np.abs(contrib.sum(axis=1).to_numpy() + base - raw).max() < 1e-6
    # grouped into drivers and converted to units, they add up to the model's forecast
    units, baseline, forecast = to_units(contrib, base)
    assert np.abs(baseline + by_driver(units).sum(axis=1).to_numpy() - model.predict(rows[FEATURE_COLS])).max() < 1e-6


def test_generated_serving_explanations_match_the_saved_forecast():
    path, daily = explain_forecast.SERVING_PATH, DATA_DIR / "bigbasket_tracked_daily_forecast.csv"
    if not path.exists() or not daily.exists():
        pytest.skip("explanations not generated; run run_pipeline.py first")
    table = pd.read_csv(path).set_index(["city", "sku_id"])
    saved = pd.read_csv(daily).groupby(["city", "sku_id"])["prediction"].mean()
    assert len(table) == len(saved)
    assert np.abs(table["forecast"] - saved.loc[table.index]).max() < 1e-6
    units = table[[f"units_{driver}" for driver in DRIVERS]].sum(axis=1)
    assert np.abs(table["baseline"] + units - table["forecast"]).max() < 1e-6


@pytest.fixture
def explained(fake_data, monkeypatch):  # noqa: F811  (pytest fixture)
    """Explanations for Delhi T1 and T2 (forecast 26 Aug - 1 Sep 2026, Diwali-free: Janmashtami ahead)."""
    rows = []
    for sku, units in [("T1", [1.0, -0.5, 3.0, 0.2, -0.4, 0.1, 0.6, -0.1]), ("T2", [0.0] * 8)]:
        row = {"city": "Delhi", "sku_id": sku, "segment": "perishable", "category": "dairy", "baseline": 10.0,
               "raw_base": np.log(10.0), "recent_28": 11.6, "last_year": 9.2, "unit_price": 25.0,
               "festival": "Janmashtami", "festival_days": 9, "forecast_start": "2026-08-26", "forecast_end": "2026-09-01"}
        row.update({f"units_{driver}": value for driver, value in zip(DRIVERS, units)})
        row["forecast"] = 10.0 + sum(units)
        rows.append(row)
    pd.DataFrame(rows).to_csv(fake_data / "serving_explanations.csv", index=False)
    monkeypatch.setattr(explain_forecast, "SERVING_PATH", fake_data / "serving_explanations.csv")
    explain_forecast.clear_cache()
    yield fake_data
    explain_forecast.clear_cache()


def test_explain_endpoint_for_a_tracked_product(server, explained):  # noqa: F811
    payload = get(server, "/api/explain", sku_id="T1")
    assert payload["source"] == "direct" and payload["analog"] is None
    e = payload["explanation"]
    assert (e["sku_id"], e["baseline"], e["forecast"]) == ("T1", 10.0, pytest.approx(13.9))
    assert len(e["drivers"]) == 6 and e["rest"]["count"] == 2
    assert [d["driver"] for d in e["drivers"]][:2] == ["recent_demand", "festival"]  # largest |units| first
    labels = {d["driver"]: d["label"] for d in e["all_drivers"]}
    assert labels["festival"] == "Janmashtami in 9 days"
    assert labels["day_of_week"] == "Day of week (weekend 29 Aug–30 Aug)"
    assert labels["recent_demand"] == "Recent sales (~12/day over 4 weeks)" and labels["price"] == "Price ₹25"
    assert 10.0 + sum(d["units"] for d in e["drivers"]) + e["rest"]["units"] == pytest.approx(e["forecast"])


def test_explain_endpoint_for_an_analog_product_uses_its_top_weighted_analog(server, explained):  # noqa: F811
    payload = get(server, "/api/explain", sku_id="A1")  # analogs T1;T2;... weights 0.4;0.3;...
    assert payload["source"] == "analog"
    assert payload["analog"] == {"count": 5, "sku_id": "T1", "product_name": "Amul Lassi 200 ml", "weight": 0.4,
                                 "price_adjustment": 1.0}
    assert payload["explanation"]["sku_id"] == "T1" and payload["forecast_daily"] == 30.0


def test_explain_endpoint_unknown_sku_is_404_json(server, explained):  # noqa: F811
    for sku in ["NOPE", "T1x"]:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, "/api/explain", sku_id=sku)
        assert error.value.code == 404
        assert error.value.headers["Content-Type"] == "application/json"
        assert json.loads(error.value.read()) == {"error": "No explanation for this product", "name": sku}


def test_importance_endpoint_returns_the_top_10_per_segment(server, monkeypatch, tmp_path):  # noqa: F811
    features = [{"feature": f, "label": FEATURE_LABELS[f], "driver": "x", "mean_abs_shap": 1.0 / (i + 1),
                 "typical_effect_pct": 1.0} for i, f in enumerate(FEATURE_COLS)]
    path = tmp_path / "shap_importance.json"
    path.write_text(json.dumps({"window": ["2026-04-01", "2026-08-25"], "note": "n",
                                "segments": {"perishable": {"rows": 5, "features": features}}}))
    monkeypatch.setattr(explain_forecast, "IMPORTANCE_PATH", path)
    explain_forecast.clear_cache()
    with urllib.request.urlopen(f"{server}/api/explain/importance", timeout=30) as response:
        payload = json.loads(response.read())
    explain_forecast.clear_cache()
    assert [f["feature"] for f in payload["segments"]["perishable"]["features"]] == FEATURE_COLS[:10]
