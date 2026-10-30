"""
Analog forecasting guards (src/forecast_regional.py): free samples, store
packaging and sub-₹1 prices are not forecast, and the price adjustment is
clipped so an analog forecast stays within 0.5x-2x its neighbours' weighted
mean. Also the dashboard's one-row-per-product lists and not-forecast note.
"""

import numpy as np
import pandas as pd
import pytest

import dashboard
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
        "pack_size_value": 100.0,
        "pack_size_unit": "g",
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


def test_free_samples_packaging_and_sub_1_rupee_products_are_not_forecast():
    rows = [  # sku, name, model category, MRP, expected reason
        ("A", "Chutney, 50 g (Free Sample)", "snacks", 0.01, "not forecast: free sample"),  # name wins over price
        ("B", "Sunsilk Shampoo, 7 ml Pouch", "personal_care", 3.0, ""),  # real cheap products stay
        ("C", "Marino Marie Crunch, 43 g", "snacks", 3.75, ""),
        ("D", "Scrub Pad, 5x1 pc", "household", 0.0, "not forecast: no price"),  # MRP 0
        ("E", "Toffee", "sweets", 0.5, "not forecast: no price"),  # below ₹1
        ("F", "Biscuits", "snacks", 1.0, ""),  # exactly ₹1 is kept
        ("G", "No MRP listed", "snacks", np.nan, ""),  # median-filled, as before
        ("H", "Tracked cheap item", "snacks", 0.5, ""),  # tracked: scored directly, never excluded
        ("I", "FREE SAMPLE tea", "beverages", 120.0, "not forecast: free sample"),  # any case, any price
        ("J", "MCIP Consumable MCIP Consumables Brown Paper Bag, 1 pc", "household", 2.55, "not forecast: packaging"),
        ("K", "Mcip Consumables Transparent Cello Tape, 1 PC", "household", 24.23, "not forecast: packaging"),
        ("L", "MCIP Consumables Fresho Egg Barcode Label, 1 pc", "household", 0.54, "not forecast: packaging"),
        ("M", "MCIP Consumables lookalike", "snacks", 50.0, ""),  # packaging only in household
        ("N", "DP Paper Gift Bags, 3 pcs", "household", 129.0, ""),  # consumer gift bags stay
    ]
    products = pd.DataFrame(rows, columns=["sku_id", "SKU Name", "category", "mrp", "expected"])
    reasons = not_forecast_reasons(products, tracked_ids=["H"])
    assert list(reasons) == list(products["expected"])


def test_generated_forecast_excludes_free_samples_and_stays_near_its_analogs():
    forecast_path = DATA_DIR / "bigbasket_regional_forecast.csv"
    if not forecast_path.exists() or not NOT_FORECAST_PATH.exists():
        pytest.skip("regional forecast not generated; run run_pipeline.py first")
    forecast = pd.read_csv(forecast_path, usecols=["sku_id", "product_name", "city", "forecast_daily",
                                                   "forecast_source", "analog_ids"])
    not_forecast = pd.read_csv(NOT_FORECAST_PATH)
    assert set(not_forecast["reason"]) <= {"not forecast: free sample", "not forecast: packaging",
                                           "not forecast: no price"}
    assert (not_forecast.loc[not_forecast["reason"] == "not forecast: no price", "mrp"] < 1).all()
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


def demand_row(sku, name, source="lightgbm_analog_k5", forecast=10.0, pack=("32.0", "g")):
    return {"region": "Delhi", "sku_id": sku, "product_name": name, "pack_size_value": pack[0],
            "pack_size_unit": pack[1], "forecast_source": source, "forecast_units": str(forecast),
            "forecast_horizon": str(forecast * 7),
            "category": "snacks", "subcategory": "", "risk_status": "normal", "days_of_cover": "5",
            "reorder_qty": "0"}


def test_unique_products_keeps_the_direct_sku_else_the_first():
    rows = [
        demand_row("BB052007", "Tong Garden Salted Peanuts, 32 g Pouch", forecast=12),
        demand_row("BB048328", "tong garden  salted peanuts 32 g pouch"),  # same after normalising
        demand_row("BB000900", "Muesli, 300 g", forecast=5, pack=("300.0", "g")),
        demand_row("BB000100", "Muesli 300g!", forecast=9, pack=("300.0", "g")),
        demand_row("BB000500", "Muesli, 300 g", source="lightgbm_direct", forecast=3, pack=("300.0", "g")),
        demand_row("BB000200", "Muesli, 300 g", forecast=8, pack=("500.0", "g")),  # other pack size
    ]
    kept = {row["sku_id"] for row in dashboard.unique_products(rows)}
    # peanuts: no direct SKU, so the first SKU id; muesli 300 g: the direct one,
    # even though duplicates forecast more; "Muesli 300g!" normalises differently
    assert kept == {"BB048328", "BB000500", "BB000100", "BB000200"}


def test_regional_panel_lists_each_product_once_and_counts_not_forecast(tmp_path, monkeypatch):
    rows = [demand_row(f"BB{i:06d}", "Tong Garden Salted Peanuts, 32 g Pouch", forecast=100 - i) for i in range(3)]
    rows += [demand_row(f"BC{i:06d}", f"Snack {i}", forecast=50 - i) for i in range(12)]
    pd.DataFrame(rows).to_csv(tmp_path / "bigbasket_regional_inventory.csv", index=False)
    pd.DataFrame({"date": ["2099-01-01"], "next_festival": ["Diwali"], "days_until_next_festival": ["0"]}).to_csv(
        tmp_path / "festival_calendar_5yr.csv", index=False)
    pd.DataFrame({"sku_id": ["X", "Y"], "reason": ["not forecast: free sample", "not forecast: packaging"]}).to_csv(
        tmp_path / dashboard.NOT_FORECAST_FILE, index=False)
    monkeypatch.setattr(dashboard, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "REGIONAL_INTELLIGENCE_CACHE", None)

    [delhi] = dashboard.regional_intelligence()
    names = [row["product_name"] for row in delhi["demand"]]
    assert len(names) == 10 and len(set(names)) == 10
    assert delhi["demand"][0]["sku_id"] == "BB000000"
    assert delhi["sku_count"] == 15  # counts stay per SKU
    assert delhi["not_forecast_count"] == 2
    assert delhi["not_forecast_download"] == "/download/bigbasket_not_forecast.csv"
    assert 'id="not-forecast"' in dashboard.ADMIN_PAGE
