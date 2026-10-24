"""
train_model.py
----------------
Trains SEPARATE LightGBM demand-forecasting models for:
    - perishables   (dairy, sweets)
    - non-perishables (frozen vegetables, dry fruits, beverages, snacks, household)

Then uses each model's forecast + current opening_stock to flag:
  - stockout risk   (forecast demand > available stock)
  - overstock risk  (available stock >> forecast demand for a sustained period)

Evaluation is TIME-BASED, never random -- random splits leak future
information into lag/rolling features. Models are compared with an
expanding-window cross-validation (CV_FOLDS, each with a LEAD_TIME_DAYS gap
between training and validation, folds chosen to cover festival seasons),
and the final model is scored once on FINAL_TEST, which no tuning step sees.

Outputs:
    models/lgbm_perishable.txt / lgbm_nonperishable.txt
  outputs/metrics.json
  outputs/feature_importance_perishable.png / _nonperishable.png
  outputs/stockout_risk_report.csv
  outputs/overstock_risk_report.csv
  outputs/test_predictions.csv
"""

import argparse
import json
import time

import optuna
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping
from config import (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, LEAD_TIME_DAYS, FEATURE_COLS,
                    CATEGORICAL_FEATURES, SHELF_LIFE_DAYS)
from inventory import classify_risk
from metrics import evaluate, mape, seasonal_naive_scale, wape
from sample_products import SAMPLED_PRODUCTS_PATH

FEATURES_PATH = DATA_DIR / "features.csv"
OUT_DIR = OUTPUTS_DIR
TARGET = "demand"
SERIES_COLS = ["city", "sku_id"]

# Expanding-window CV: (train_end, valid_start, valid_end), with a
# LEAD_TIME_DAYS gap between train_end and valid_start.
CV_FOLDS = [
    ("2025-03-24", "2025-04-01", "2025-06-30"),  # quiet quarter, no festivals
    ("2025-06-23", "2025-07-01", "2025-09-14"),  # Raksha Bandhan, Independence Day, Ganesh Chaturthi
    ("2025-09-07", "2025-09-15", "2025-12-31"),  # Navratri, Dussehra, Diwali, Christmas
    ("2025-12-24", "2026-01-01", "2026-03-31"),  # New Year, Makar Sankranti, Holi, Eid-ul-Fitr
]
# Held out from all tuning; scored once at the end.
FINAL_TEST = ("2026-04-01", "2026-08-25")

# The original hand-picked parameters, kept as the CV baseline.
BASELINE_PARAMS = {
    "objective": "poisson",
    "n_estimators": 400,
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}

# Tuning: each fold trains up to MAX_TREES and stops after EARLY_STOPPING
# rounds without improvement in validation L1 (equivalent to WAPE on a
# fixed validation set).
MAX_TREES = 3000
EARLY_STOPPING = 50

for train_end, valid_start, _ in CV_FOLDS:
    gap = (pd.Timestamp(valid_start) - pd.Timestamp(train_end)).days - 1
    assert gap == LEAD_TIME_DAYS, f"fold ending {train_end} has a {gap}-day gap"


def make_model(params):
    return LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)


def cv_folds(df):
    """Yields (fold_number, train, valid) for each expanding-window fold."""
    for number, (train_end, valid_start, valid_end) in enumerate(CV_FOLDS, start=1):
        train = df[df["date"] <= train_end]
        valid = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)]
        yield number, train, valid


def cross_validate(df, params, early_stop=False):
    """
    WAPE on each CV fold. With early_stop, each fold trains up to MAX_TREES
    and stops on its validation set; best_iteration is recorded per fold.
    """
    results = []
    for number, train, valid in cv_folds(df):
        if early_stop:
            model = make_model({**params, "n_estimators": MAX_TREES, "metric": "l1"})
            model.fit(
                train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES,
                eval_X=(valid[FEATURE_COLS],), eval_y=(valid[TARGET],),
                callbacks=[early_stopping(EARLY_STOPPING, verbose=False)],
            )
        else:
            model = make_model(params)
            model.fit(train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)
        preds = np.clip(model.predict(valid[FEATURE_COLS]), 0, None)
        result = {"fold": number, "WAPE_%": round(wape(valid[TARGET], preds), 2)}
        if early_stop:
            result["best_iteration"] = int(model.best_iteration_)
        results.append(result)
    return results


def suggest_params(trial):
    return {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "subsample_freq": 1,  # without this LightGBM ignores subsample
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def tune_segment(df, segment_name, n_trials):
    """
    Optuna TPE search minimising mean CV WAPE on `demand`. Returns the best
    parameters with n_estimators set to the mean best_iteration across folds,
    plus that trial's per-fold results.
    """
    def objective(trial):
        folds = cross_validate(df, suggest_params(trial), early_stop=True)
        trial.set_user_attr("folds", folds)
        return float(np.mean([fold["WAPE_%"] for fold in folds]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - started
    print(f"[{segment_name}] {n_trials} trials in {elapsed:.0f}s ({elapsed / n_trials:.1f}s/trial), "
          f"best mean CV WAPE {study.best_value:.2f}%")

    best = study.best_trial
    folds = best.user_attrs["folds"]
    params = suggest_params(optuna.trial.FixedTrial(best.params))
    params["n_estimators"] = int(round(np.mean([fold["best_iteration"] for fold in folds])))
    return params, folds, {"n_trials": n_trials, "seconds": round(elapsed, 1)}


def split_final(df):
    start, end = FINAL_TEST
    return df[df["date"] < start], df[(df["date"] >= start) & (df["date"] <= end)]


def train_segment(df, segment_name, n_trials):
    # Tuning and the baseline only ever see data before FINAL_TEST; the
    # final model is refit on all of that data with the tuned params.
    train, test = split_final(df)
    baseline_cv = cross_validate(train, BASELINE_PARAMS)
    print(f"[{segment_name}] baseline CV WAPE by fold:", [fold["WAPE_%"] for fold in baseline_cv])
    params_path = MODELS_DIR / f"params_{segment_name}.json"
    if n_trials > 0:
        params, tuned_cv, tuning = tune_segment(train, segment_name, n_trials)
        params_path.write_text(json.dumps(
            {"segment": segment_name, "params": params, "cv_folds": tuned_cv, "tuning": tuning}, indent=2))
    else:
        # --trials 0: reuse the last saved search instead of re-tuning, but
        # re-score it on the current data (the saved folds may be from older data)
        saved = json.loads(params_path.read_text())
        params, tuning = saved["params"], saved["tuning"]
        tuned_cv = cross_validate(train, params)
        print(f"[{segment_name}] reusing tuned params from {params_path.name}")
    print(f"[{segment_name}] tuned CV WAPE by fold:   ", [fold["WAPE_%"] for fold in tuned_cv])

    model = make_model(params)
    model.fit(train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)

    test = test.copy()
    test["predicted_units"] = np.clip(model.predict(test[FEATURE_COLS]), 0, None)
    scale = seasonal_naive_scale(train, TARGET, SERIES_COLS)
    slices = {
        "overall": test,
        "festival_weeks": test[test["is_festival_week"] == 1],
        "stockout_days": test[test["stockout_flag"] == 1],
    }
    final_test = {}
    for slice_name, rows in slices.items():
        final_test[slice_name] = {"n_rows": len(rows), **evaluate(rows[TARGET], rows["predicted_units"], scale)}
        if "true_demand" in rows.columns:  # synthetic data only
            final_test[slice_name]["WAPE_vs_true_demand_%"] = round(wape(rows["true_demand"], rows["predicted_units"]), 2)
    by_category = {category: round(wape(rows[TARGET], rows["predicted_units"]), 2)
                   for category, rows in test.groupby("category")}
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "final_test": final_test,
        "final_test_wape_by_category": by_category,
        "cv_wape_by_fold": {
            "baseline": [fold["WAPE_%"] for fold in baseline_cv],
            "tuned": [fold["WAPE_%"] for fold in tuned_cv],
            "baseline_mean": round(float(np.mean([fold["WAPE_%"] for fold in baseline_cv])), 2),
            "tuned_mean": round(float(np.mean([fold["WAPE_%"] for fold in tuned_cv])), 2),
        },
        "params": params,
        "tuning": tuning,
        "legacy_MAPE_%": round(mape(test[TARGET], test["predicted_units"]), 2),
    }

    model.booster_.save_model(str(MODELS_DIR / f"lgbm_{segment_name}.txt"))

    # feature importance chart
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    plt.figure(figsize=(7, 6))
    importances.tail(15).plot(kind="barh")
    plt.title(f"Feature importance - {segment_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance_{segment_name}.png", dpi=110)
    plt.close()

    return model, metrics, test


REPORT_COLS = ["date", "city", "sku_id", "product_name", "brand", "category", "predicted_units", "opening_stock", "days_of_cover",
               "horizon_demand", "avg_daily_pred", "days_until_delivery", "shelf_life_days", "reorder_qty"]
PREDICTION_COLS = ["date", "city", "sku_id", "category", "segment", "pred", "true_demand", "demand",
                   "opening_stock", "days_until_delivery", "shelf_life_days"]


def build_risk_report(scored):
    """
    Stockout / overstock status for EVERY day of the test window over a
    LEAD_TIME_DAYS planning horizon. Forecasts for today..today+H-1 only use
    sales up to yesterday (features lag >= H), so they are all known when
    the morning's opening_stock is counted. In the last H-1 days of the
    window the horizon is shorter, so its average is scaled up to H days.
    The stockout / overstock rule itself is inventory.classify_risk.
    """
    horizon = LEAD_TIME_DAYS
    df = scored.sort_values(SERIES_COLS + ["date"]).copy()
    df["avg_daily_pred"] = df.groupby(SERIES_COLS)["predicted_units"].transform(
        lambda s: s[::-1].rolling(horizon, min_periods=1).mean()[::-1])
    return classify_risk(df, horizon)


def risk_split(report):
    """Share of rows per status (%), per city, for the latest day and the whole window."""
    def shares(rows):
        counts = rows.groupby("city")["risk_status"].value_counts(normalize=True).unstack(fill_value=0) * 100
        return counts.reindex(columns=["stockout", "normal", "overstock"], fill_value=0).round(1).to_dict(orient="index")
    latest = report[report["date"] == report["date"].max()]
    return {"latest_day": shares(latest), "whole_test_window": shares(report)}


def main():
    parser = argparse.ArgumentParser(description="Tune, train and evaluate the demand models.")
    parser.add_argument("--trials", type=int, default=40,
                        help="Optuna trials per segment; 0 reuses models/params_<segment>.json")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    segments, scored, predictions = {}, [], []
    for name, is_perishable in [("perishable", 1), ("nonperishable", 0)]:
        model, metrics, test = train_segment(df[df["is_perishable"] == is_perishable], name, args.trials)
        segments[name] = metrics
        scored.append(test)
        predictions.append(test.assign(segment=name, pred=test["predicted_units"],
                                       shelf_life_days=test["category"].map(SHELF_LIFE_DAYS)))
        overall, festival = metrics["final_test"]["overall"], metrics["final_test"]["festival_weeks"]
        print(f"[{name}] FINAL TEST WAPE={overall['WAPE_%']}% (festival weeks {festival['WAPE_%']}%) "
              f"MAE={overall['MAE']} MASE={overall['MASE']} Bias={overall['Bias_%']}%")

    # scored final-test rows, for offline analysis (risk_confusion_matrix.py)
    pd.concat(predictions)[PREDICTION_COLS].sort_values(["city", "sku_id", "date"]).to_csv(
        OUT_DIR / "test_predictions.csv", index=False)

    report = build_risk_report(pd.concat(scored).drop(columns="brand", errors="ignore"))
    report = report.merge(pd.read_csv(SAMPLED_PRODUCTS_PATH, usecols=["sku_id", "product_name", "brand"]), on="sku_id", how="left")
    report[report["risk_status"] == "stockout"].sort_values(["date", "days_of_cover"])[REPORT_COLS].to_csv(
        OUT_DIR / "stockout_risk_report.csv", index=False)
    report[report["risk_status"] == "overstock"].sort_values(["date", "days_of_cover"], ascending=[True, False])[REPORT_COLS].to_csv(
        OUT_DIR / "overstock_risk_report.csv", index=False)
    split = risk_split(report)
    print("\nRisk split on the latest day (% of SKUs per city):")
    print(pd.DataFrame(split["latest_day"]).T.to_string())
    print("Risk split over the whole test window:")
    print(pd.DataFrame(split["whole_test_window"]).T.to_string())

    # MAPE is kept only for comparison with older runs, outside the main metrics
    legacy = [{"segment": name, "MAPE_%": m.pop("legacy_MAPE_%")} for name, m in segments.items()]
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "target": TARGET,
            "final_test_window": list(FINAL_TEST),
            "cv_folds": [{"train_end": a, "valid_start": b, "valid_end": c} for a, b, c in CV_FOLDS],
            "segments": segments,
            "risk_split_%": split,
            "legacy": legacy,
        }, f, indent=2)

    print(f"\nSaved metrics, risk reports, and feature-importance charts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
