"""
Serving from SERVING_DATE without touching evaluation: the synthetic history
is extended past EVALUATION_END for serving only, and nothing the
evaluation produced may change. Data checks skip when the pipeline outputs
aren't there.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dashboard
import festival_uplift
import run_pipeline
import serving_risk_report
from config import DATA_DIR, EVALUATION_END, LEAD_TIME_DAYS, OUTPUTS_DIR, SERVING_DATE
from inventory import simulate_inventory

FIXTURES = Path(__file__).parent / "fixtures"
# sha256 of data/pos_synthetic.csv as it was before the extension (rows up to EVALUATION_END)
HISTORY_SHA256 = "926307ce63206b3cd2c532c4143235294ba01d9ba445a43907d1eeb74ec786cc"
# evaluation outputs: train_model.py, risk_confusion_matrix.py's input and the model comparison
EVALUATION_SHA256 = {
    "test_predictions.csv": "86fdb76abbc279f38b985369e28abdf48bc91f7887f007e5113d32d55652a3a3",
    "stockout_risk_report.csv": "12293d24f1b0f427039fa574fa2f38e27a6f74693d58d4d6ddded41bc9f6d9a0",
    "overstock_risk_report.csv": "45bf5425456ef17a955605d1fb402724e668412636882c88f5eeef5ad853bebe",
    "model_comparison.csv": "e138f0c73d5070482c39ff97c16e115a572e4cf190e3ad25dfa88a18e4e2c0a7",
}
SERVING_START = (pd.Timestamp(SERVING_DATE) + pd.Timedelta(days=1)).date().isoformat()


def require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not generated; run run_pipeline.py first")
    return path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_history_rows_are_byte_identical_and_the_extension_reaches_the_serving_date():
    data = require(DATA_DIR / "pos_synthetic.csv").read_bytes()
    first_new_day = (pd.Timestamp(EVALUATION_END) + pd.Timedelta(days=1)).date().isoformat()
    cut = data.index(f"\n{first_new_day},".encode()) + 1
    assert sha256(data[:cut]) == HISTORY_SHA256
    assert data.rstrip().rsplit(b"\n", 1)[1].startswith(SERVING_DATE.encode())


def test_metrics_json_is_unchanged():
    metrics = json.loads(require(OUTPUTS_DIR / "metrics.json").read_text())
    assert metrics == json.loads((FIXTURES / "evaluation_metrics.json").read_text())
    assert metrics["final_test_window"] == ["2026-04-01", EVALUATION_END]


@pytest.mark.parametrize("name", sorted(EVALUATION_SHA256))
def test_evaluation_outputs_are_unchanged(name):
    assert sha256(require(OUTPUTS_DIR / name).read_bytes()) == EVALUATION_SHA256[name]


def test_everything_served_counts_from_the_serving_date():
    for name in ["bigbasket_regional_forecast.csv", "bigbasket_tracked_daily_forecast.csv", "serving_opening_stock.csv"]:
        require(DATA_DIR / name)
    require(OUTPUTS_DIR / serving_risk_report.STOCKOUT_PATH.name)
    assert dashboard.data_date().isoformat() == SERVING_DATE
    window = pd.date_range(SERVING_START, periods=LEAD_TIME_DAYS).strftime("%Y-%m-%d").tolist()
    daily = pd.read_csv(DATA_DIR / "bigbasket_tracked_daily_forecast.csv")
    assert sorted(daily["date"].unique()) == window
    regional = pd.read_csv(DATA_DIR / "bigbasket_regional_forecast.csv", usecols=["forecast_start", "forecast_end"], nrows=5)
    assert (regional["forecast_start"] == window[0]).all() and (regional["forecast_end"] == window[-1]).all()
    assert set(pd.read_csv(DATA_DIR / "serving_opening_stock.csv")["date"]) == {SERVING_START}
    report = serving_risk_report.build_report()
    assert len(report) == daily.groupby(["city", "sku_id"]).ngroups and set(report["date"].dt.date.astype(str)) == {SERVING_START}

    calendar = dashboard.read_csv_from_path(DATA_DIR / "festival_calendar_5yr.csv")
    assert dashboard.next_festival(calendar, dashboard.data_date()) == {"name": "Diwali", "date": "2026-11-08", "days_until": 10}
    plan = festival_uplift.festival_planning("Delhi")
    assert (plan["festival"], plan["data_date"], plan["window_end"]) == ("Diwali", SERVING_DATE, window[-1])


def test_serving_report_has_the_evaluation_report_columns():
    from train_model import REPORT_COLS
    assert serving_risk_report.REPORT_COLS == REPORT_COLS


class NoRandom:
    """Stands in for the generator: fixed phase, never an over-order."""
    def __init__(self, phase):
        self.phase = phase

    def integers(self, low, high):
        return np.full(np.shape(high), self.phase)

    def random(self, shape):
        return np.ones(shape)

    def uniform(self, low, high, shape):
        return np.full(shape, low)


@pytest.mark.parametrize("category", ["dairy", "snacks"])
def test_continuing_the_inventory_matches_one_long_run(category):
    # one 60-day run vs 40 days + a continuation from the state on day 39
    demand = np.random.default_rng(1).uniform(5, 15, (1, 60))
    base = [10.0]
    opening, sold, until = simulate_inventory(demand, base, [category], NoRandom(phase=2))
    first_opening, first_sold, first_until = simulate_inventory(demand[:, :40], base, [category], NoRandom(phase=2))
    review_days = 3 if category == "dairy" else 7
    phase = (int(first_until[0, -1]) - 1) % review_days  # as generate_synthetic_data.extend does
    rest = simulate_inventory(demand[:, 40:], base, [category], NoRandom(phase=99),
                              initial_stock=[first_opening[0, -1] - first_sold[0, -1]], phase=[phase],
                              history=demand[:, 26:40])
    np.testing.assert_allclose(rest[0], opening[:, 40:])
    np.testing.assert_allclose(rest[1], sold[:, 40:])
    np.testing.assert_array_equal(rest[2], until[:, 40:])


def test_pipeline_skips_training_when_saved_models_exist(tmp_path, monkeypatch):
    for segment in ["perishable", "nonperishable"]:
        (tmp_path / f"lgbm_{segment}.txt").write_text("saved")
    monkeypatch.setattr(run_pipeline, "MODELS_DIR", tmp_path)
    ran = []
    monkeypatch.setattr(run_pipeline.subprocess, "run",
                        lambda command, cwd: ran.append(command[1]) or subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])
    run_pipeline.main()
    assert "train_model.py" not in ran and "forecast_regional.py" in ran and "serving_risk_report.py" in ran
    ran.clear()
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--retrain"])
    run_pipeline.main()
    assert "train_model.py" in ran
