"""
The model comparison must fit and score every model on identical data:
the same CV folds (train_model.CV_FOLDS) and the same final test window
(train_model.FINAL_TEST). Runs the real comparison on a small synthetic
frame with one tuning trial, so it needs no generated data.
"""

import numpy as np
import pandas as pd

from compare_models import compare_segment
from config import CATEGORICAL_FEATURES, FEATURE_COLS, LEAD_TIME_DAYS
from train_model import CV_FOLDS, FINAL_TEST


def small_frame():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-11-01", FINAL_TEST[1])
    series = [("Bengaluru", "SKU1", "dairy"), ("Mumbai", "SKU2", "snacks"), ("Delhi", "SKU3", "dairy")]
    frames = []
    for code, (city, sku, category) in enumerate(series):
        n = len(dates)
        frame = pd.DataFrame({col: rng.random(n) for col in FEATURE_COLS})
        for col in CATEGORICAL_FEATURES:
            frame[col] = code
        frame["is_festival_week"] = rng.integers(0, 2, n)
        demand = rng.poisson(20 + 10 * frame["is_festival_week"]).astype(float)
        frame = frame.assign(
            date=dates, city=city, sku_id=sku, category=category, demand=demand,
            true_demand=demand + rng.normal(0, 1, n).clip(-demand), stockout_flag=rng.integers(0, 2, n),
            opening_stock=rng.integers(0, 200, n), days_until_delivery=rng.integers(1, 8, n),
            lag_7=pd.Series(demand).shift(LEAD_TIME_DAYS),
            roll_mean_28_lag7=pd.Series(demand).shift(LEAD_TIME_DAYS).rolling(28, min_periods=1).mean(),
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_every_model_uses_identical_folds_and_test_window():
    results = compare_segment(small_frame(), "test", n_trials=1)

    entries = {(r["model"], r["setting"]) for r in results}
    assert entries == {("Seasonal naive", "baseline"), ("Moving average 28d", "baseline"),
                       *((m, s) for m in ("XGBoost", "CatBoost", "LightGBM") for s in ("default", "tuned"))}

    reference = results[0]["splits"]
    for result in results:
        assert result["splits"] == reference, f"{result['model']} {result['setting']} used different data"

    # and those splits are exactly train_model's folds and test window
    for split, (train_end, valid_start, valid_end) in zip(reference["cv"], CV_FOLDS):
        assert split["train"]["end"] == train_end
        assert (split["valid"]["start"], split["valid"]["end"]) == (valid_start, valid_end)
        gap = (pd.Timestamp(split["valid"]["start"]) - pd.Timestamp(split["train"]["end"])).days - 1
        assert gap == LEAD_TIME_DAYS
    final = reference["final_test"]
    assert (final["valid"]["start"], final["valid"]["end"]) == FINAL_TEST
    assert pd.Timestamp(final["train"]["end"]) == pd.Timestamp(FINAL_TEST[0]) - pd.Timedelta(days=1)
