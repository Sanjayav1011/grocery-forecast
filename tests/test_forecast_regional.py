"""
Analog forecasting guards (src/forecast_regional.py): free samples and
near-zero prices are not forecast, and the price adjustment is clipped so an
analog forecast stays within 0.5x-2x its neighbours' weighted mean.
"""

import numpy as np
import pandas as pd
import pytest

import forecast_regional
from config import DATA_DIR
from forecast_regional import (ANALOG, K_ANALOGS, NOT_FORECAST_PATH, PRICE_ADJ_RANGE, analog_forecasts,
                               not_forecast_reasons)

CITIES = ["Bengaluru", "Delhi"]


def products_and_tracked():
    """6 tracked and 4 untracked products in one category, with extreme prices among the untracked."""
    rng = np.random.default_rng(0)
    tracked_mrp = [20, 40, 60, 80, 100, 150]
    untracked_mrp = [0.01, 3.0, 50.0, 10_000.0]  # near-zero, cheap, normal, very expensive
    mrp = tracked_mrp + untracked_mrp
    products = pd.DataFrame({
        "sku_id": [f"T{i}" for i in range(6)] + [f"U{i}" for i in range(4)],
        "SKU Name": [f"product {i}" for i in range(10)],
        "brand": "Brand",
        "category": "snacks",
        "bb_subcategory": ["a", "b"] * 5,
        "mrp_filled": mrp,
        "pack_filled": rng.uniform(50, 500, 10),
        "brand_catalog_count": rng.integers(1, 200, 10),
    })
    tracked_forecast = pd.DataFrame([
        {"sku_id": f"T{i}", "city": city, "forecast_daily": rng.uniform(1, 30), "recent_daily": rng.uniform(1, 30)}
        for i in range(6) for city in CITIES
    ])
    return products, tracked_forecast


def test_analog_forecast_stays_within_2x_of_neighbours_weighted_mean(monkeypatch):
    products, tracked_forecast = products_and_tracked()
    adjusted = analog_forecasts(products, tracked_forecast)
    # elasticity 0 -> the same analogs and weights with no price adjustment,
    # i.e. exactly the neighbours' inverse-distance weighted mean
    monkeypatch.setattr(forecast_regional, "PRICE_ELASTICITY", 0.0)
    weighted_mean = analog_forecasts(products, tracked_forecast)

    analog = adjusted["forecast_source"] == ANALOG
    assert analog.sum() == 4 * len(CITIES)
    assert (adjusted["analog_ids"] == weighted_mean["analog_ids"]).all()
    low, high = PRICE_ADJ_RANGE
    tolerance = 2e-3  # values are rounded to 3 decimals
    for col in ["forecast_daily", "recent_daily"]:
        ratio_ok = adjusted.loc[analog, col] <= high * weighted_mean.loc[analog, col] + tolerance
        assert ratio_ok.all(), adjusted.loc[analog & ~ratio_ok]
        assert (adjusted.loc[analog, col] >= low * weighted_mean.loc[analog, col] - tolerance).all()
    # the ₹0.01 product sits exactly at the 2x cap, the ₹10,000 one at 0.5x
    for sku, factor in [("U0", high), ("U3", low)]:
        rows = adjusted["sku_id"] == sku
        np.testing.assert_allclose(adjusted.loc[rows, "forecast_daily"],
                                   factor * weighted_mean.loc[rows, "forecast_daily"], atol=tolerance)


def test_free_samples_and_sub_5_rupee_products_are_not_forecast():
    products = pd.DataFrame({
        "sku_id": ["A", "B", "C", "D", "E", "F", "G"],
        "SKU Name": ["Chutney, 50 g (Free Sample)", "Shampoo 7 ml Pouch", "Paper bag", "Biscuits",
                     "No MRP listed", "Tracked cheap item", "FREE SAMPLE tea"],
        "mrp": [0.01, 4.0, 0.0, 5.0, np.nan, 3.0, 120.0],
    })
    reasons = not_forecast_reasons(products, tracked_ids=["F"])
    assert list(reasons) == [
        "not forecast: free sample",  # free sample wins over its ₹0.01 price
        "not forecast: no price",     # below ₹5
        "not forecast: no price",     # MRP 0
        "",                           # exactly ₹5 is kept
        "",                           # missing MRP is median-filled, as before
        "",                           # tracked products are scored directly, never excluded
        "not forecast: free sample",  # case-insensitive, whatever the price
    ]


def test_generated_forecast_excludes_free_samples_and_stays_near_its_analogs():
    forecast_path = DATA_DIR / "bigbasket_regional_forecast.csv"
    if not forecast_path.exists() or not NOT_FORECAST_PATH.exists():
        pytest.skip("regional forecast not generated; run run_pipeline.py first")
    forecast = pd.read_csv(forecast_path, usecols=["sku_id", "product_name", "city", "forecast_daily",
                                                   "forecast_source", "analog_ids"])
    not_forecast = pd.read_csv(NOT_FORECAST_PATH)
    assert set(not_forecast["reason"]) <= {"not forecast: free sample", "not forecast: no price"}
    assert not set(not_forecast["sku_id"]) & set(forecast["sku_id"])
    assert not forecast["product_name"].str.contains(forecast_regional.FREE_SAMPLE_PATTERN, na=False).any()

    # weights aren't saved, but a weighted mean is never above the largest
    # neighbour, so 2x the largest neighbour bounds every analog forecast
    own = forecast.set_index(["sku_id", "city"])["forecast_daily"]
    analog = forecast[forecast["forecast_source"] == ANALOG]
    neighbours = analog.assign(analog_id=analog["analog_ids"].str.split(";")).explode("analog_id")
    assert (neighbours.groupby(["sku_id", "city"])["analog_id"].count() == K_ANALOGS).all()
    neighbour_values = own.reindex(pd.MultiIndex.from_arrays([neighbours["analog_id"], neighbours["city"]]))
    largest = pd.Series(neighbour_values.to_numpy(), index=neighbours.index).groupby(level=0).max()
    assert (analog["forecast_daily"] <= PRICE_ADJ_RANGE[1] * largest.loc[analog.index] + 2e-3).all()
