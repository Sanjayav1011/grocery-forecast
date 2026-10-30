"""
Name corrections to the catalog -> model category mapping
(src/sample_products.py NAME_CORRECTIONS): baked flat bread, pita crisps and
cookies filed under "sweets" are forecast as snacks, only for untracked
products, so the tracked set and its models are untouched.
"""

import numpy as np
import pandas as pd
import pytest

from config import ARCHIVE_CSV, DATA_DIR
from sample_products import SAMPLED_PRODUCTS_PATH, name_corrections, prepare_catalog

MOVED = ["BB037804", "BB048315", "BB049809", "BB051994", "BB053488"]  # cookies, flat bread x2, pita crisps x2


def test_name_corrections_move_only_clear_untracked_sweets():
    rows = [  # name, model category, sampling category, expected
        ("Wingreens Farms Crispy Baked Multi-grain Flat Bread - With Flax Seeds, 2 x 100 g", "sweets", np.nan, "snacks"),
        ("Wingreens Farms Bread - Baked Pita Crisps Chilli & Garlic, 2 x 100 g", "sweets", np.nan, "snacks"),
        ("fresho! Signature Pinwheel Cookies, 150 g", "sweets", np.nan, "snacks"),
        ("Britannia Treat Croissant - Cocoa Creme Filling, 6x45 g", "sweets", np.nan, "sweets"),  # not clear: kept
        ("fresho! Signature Red Velvet Cake, 200 g", "sweets", np.nan, "sweets"),
        ("Some Flatbread Crisps, 100 g", "sweets", "sweets", "sweets"),  # tracked: never moved
        ("Amul Choco Crisps Shake, 200 ml", "dairy", np.nan, "dairy"),  # only sweets are corrected
    ]
    catalog = pd.DataFrame(rows, columns=["SKU Name", "model_category", "sampling_category", "expected"])
    assert list(name_corrections(catalog)) == list(catalog["expected"])


def test_catalog_moves_exactly_the_listed_products_and_keeps_the_tracked_set():
    if not ARCHIVE_CSV.exists() or not SAMPLED_PRODUCTS_PATH.exists():
        pytest.skip("catalog not imported; run run_pipeline.py first")
    catalog = prepare_catalog()
    assert (catalog.loc[catalog["sku_id"].isin(MOVED), "model_category"] == "snacks").all()
    # undo the corrections for the listed products: applying them again moves those and nothing else
    reverted = catalog.assign(model_category=catalog["model_category"].mask(catalog["sku_id"].isin(MOVED), "sweets"))
    changed = name_corrections(reverted).fillna("") != reverted["model_category"].fillna("")
    assert sorted(reverted.loc[changed, "sku_id"]) == MOVED
    # tracked products keep the category their POS history and models were built with
    tracked = pd.read_csv(SAMPLED_PRODUCTS_PATH).set_index("sku_id")
    assert (catalog.set_index("sku_id").loc[tracked.index, "model_category"] == tracked["category"]).all()

    forecast_path = DATA_DIR / "bigbasket_regional_forecast.csv"
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path, usecols=["sku_id", "category"])
        assert set(forecast.loc[forecast["sku_id"].isin(MOVED), "category"]) == {"snacks"}
