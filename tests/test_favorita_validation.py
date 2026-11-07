"""
Favorita real-data validation (validation/favorita/): no leakage across the
split date, no negative sales, holiday mapping and the results.json schema.
Tests that read the Favorita CSVs skip when data/favorita/ is missing (a
fresh clone); the rest run on small hand-made frames.
"""

import json

import numpy as np
import pandas as pd
import pytest

import favorita_data as fd
import run_validation as rv

needs_data = pytest.mark.skipif(not fd.data_available(), reason="Favorita CSVs not in data/favorita/")
needs_results = pytest.mark.skipif(not rv.RESULTS_PATH.exists(),
                                   reason="results.json not generated; run validation/favorita/run_validation.py")


def toy_frame():
    dates = pd.date_range("2016-01-01", fd.END, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"store_nbr": np.repeat([1, 2], len(dates)), "item_nbr": 7, "date": np.tile(dates, 2),
                         fd.TARGET: rng.poisson(5, 2 * len(dates)).astype("float32")})


def features(frame):
    return fd.add_lag_and_rolling_features(frame, group_cols=fd.SERIES_COLS, target_col=fd.TARGET)


def test_split_dates_do_not_overlap():
    train, test = rv.split(toy_frame())
    assert train["date"].max() == pd.Timestamp(fd.TRAIN_END)
    assert test["date"].min() == pd.Timestamp(fd.TEST[0]) > train["date"].max()
    assert pd.Timestamp(rv.TUNE_VALID[1]) <= pd.Timestamp(fd.TRAIN_END)
    assert (pd.Timestamp(rv.TUNE_VALID[0]) - pd.Timestamp(rv.TUNE_TRAIN_END)).days == 8  # a 7-day gap


def test_no_leakage_across_the_split_date():
    """Changing every test-window sale leaves all training rows (and the first test week) unchanged."""
    frame = toy_frame()
    changed = frame.copy()
    in_test = changed["date"] >= fd.TEST[0]
    changed.loc[in_test, fd.TARGET] *= 10
    before, after = features(frame), features(changed)
    known = before["date"] < pd.Timestamp(fd.TEST[0]) + pd.Timedelta(days=fd.LEAD_TIME_DAYS)
    pd.testing.assert_frame_equal(before.loc[known, fd.LAG_COLS], after.loc[known, fd.LAG_COLS])
    later = before["date"] >= pd.Timestamp(fd.TEST[0]) + pd.Timedelta(days=fd.LEAD_TIME_DAYS)
    assert not before.loc[later, "lag_7"].equals(after.loc[later, "lag_7"])  # the check can detect a change


def test_lag_7_is_the_sale_seven_days_earlier():
    out = features(toy_frame()).sort_values(fd.SERIES_COLS + ["date"])
    expected = out.groupby(fd.SERIES_COLS)[fd.TARGET].shift(7)
    pd.testing.assert_series_equal(out["lag_7"], expected, check_names=False)


def test_cleaning_clips_returns_and_fills_promotions():
    rows = pd.DataFrame({"date": pd.to_datetime(["2016-01-01"] * 3), "store_nbr": 1, "item_nbr": [1, 2, 3],
                         "unit_sales": np.array([-3, 0, 2.5], dtype="float32"), "onpromotion": [None, "True", "False"]})
    cleaned, stats = fd.clean(rows)
    assert cleaned["unit_sales"].tolist() == [0, 0, 2.5]
    assert cleaned["onpromotion"].tolist() == [False, True, False]
    assert stats["negative_sales_clipped"] == 1 and stats["onpromotion_missing_filled"] == 1


def test_grid_fills_zero_sale_days_from_the_first_sale_only():
    rows = pd.DataFrame({"date": pd.to_datetime(["2017-08-10", "2017-08-13", "2017-08-12"]), "store_nbr": 1,
                         "item_nbr": [5, 5, 6], "unit_sales": np.float32(2), "onpromotion": False})
    grid, stats = fd.build_grid(rows, rows[fd.SERIES_COLS].drop_duplicates())
    assert grid.groupby("item_nbr")["date"].min().tolist() == [pd.Timestamp("2017-08-10"), pd.Timestamp("2017-08-12")]
    assert len(grid) == 6 + 4 and (grid["unit_sales"] >= 0).all()
    assert stats["zero_sale_days_filled"] == len(grid) - 3


@needs_data
def test_no_negative_sales_in_the_modelling_data():
    path = fd.features_path(fd.select_stores()["store_nbr"].tolist())
    if not path.exists():
        pytest.skip("Favorita features not built; run validation/favorita/run_validation.py")
    frame = pd.read_parquet(path, columns=[fd.TARGET, "lag_7", "roll_mean_28_lag7", "store_open"])
    assert (frame[fd.TARGET] >= 0).all() and (frame[["lag_7", "roll_mean_28_lag7"]].min() >= 0).all()
    assert (frame["store_open"] == 1).all()


@needs_data
def test_holiday_mapping():
    stores = fd.select_stores()
    assert len(stores) == fd.N_STORES and stores["city"].is_unique
    assert {"Quito", "Guayaquil"} <= set(stores["city"])
    days = pd.to_datetime(["2017-08-10", "2017-08-11", "2016-11-04", "2016-11-12", "2017-04-13", "2016-04-16"])
    h = fd.holiday_features(stores, days).set_index(["date", "store_nbr"])
    on = lambda day, col: h.xs(pd.Timestamp(day), level="date")[col]
    # Primer Grito de Independencia 2017: 10 Aug is transferred=True, celebrated on 11 Aug (Transfer)
    assert (on("2017-08-10", "is_holiday") == 0).all()
    assert (on("2017-08-11", "is_holiday") == 1).all() and (on("2017-08-11", "is_national_holiday") == 1).all()
    assert (on("2017-08-10", "days_until_next_holiday") == 1).all()
    assert (on("2016-11-04", "is_holiday") == 1).all()  # Bridge (Puente Dia de Difuntos)
    # Work Day (Recupero Puente Dia de Difuntos): worked nationally, but a local holiday in Ambato
    ambato = int(stores.loc[stores["city"] == "Ambato", "store_nbr"].iloc[0])
    work_day = on("2016-11-12", "is_holiday")
    assert (on("2016-11-12", "is_work_day") == 1).all()
    assert work_day[ambato] == 1 and work_day.drop(ambato).eq(0).all()
    cuenca = int(stores.loc[stores["city"] == "Cuenca", "store_nbr"].iloc[0])  # local, Fundacion de Cuenca (Transfer)
    local = on("2017-04-13", "is_holiday")
    assert local[cuenca] == 1 and local.drop(cuenca).eq(0).all()
    assert (on("2016-04-16", "is_event") == 1).all() and (on("2016-04-16", "is_holiday") == 0).all()


@needs_results
def test_results_json_schema():
    results = json.loads(rv.RESULTS_PATH.read_text())
    for key in ("experiment", "dates", "stores", "items", "cleaning", "rows", "features", "parameters", "metrics",
                "holiday_vs_normal_wape_%", "wape_by_store_%", "run_time_seconds", "examples"):
        assert key in results, key
    assert results["dates"]["train"][1] < results["dates"]["test"][0]
    assert results["dates"]["tuning_valid"][1] <= results["dates"]["train"][1]
    assert len(results["stores"]) == fd.N_STORES and {"Quito", "Guayaquil"} <= {s["city"] for s in results["stores"]}
    assert 0 < results["items"]["sales_share_covered_%"] <= 100
    assert results["cleaning"]["negative_sales_after_cleaning"] == 0
    for segment in ("perishable", "nonperishable", "combined"):
        assert set(results["metrics"][segment]) == set(rv.METHODS)
        for block in results["metrics"][segment].values():
            assert {"n_rows", "MAE", "RMSE", "WAPE_%", "MASE", "Bias_%"} <= set(block)
        assert {"holiday_days", "normal_days"} == set(results["holiday_vs_normal_wape_%"][segment])
    for segment in ("perishable", "nonperishable"):
        assert results["parameters"][segment]["params"]["objective"] == "tweedie"
        assert results["parameters"][segment]["n_trials"] <= 30
    assert len(results["wape_by_store_%"]) == fd.N_STORES
    for name in ("fig_wape_vs_baselines.png", "fig_holiday_vs_normal.png", "fig_actual_vs_pred.png"):
        assert (rv.FAVORITA_FIGURES_DIR / name).exists(), name
