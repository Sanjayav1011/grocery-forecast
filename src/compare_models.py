"""
compare_models.py
-------------------
Model comparison study. For each segment (perishable, non-perishable),
compares on the SAME features (config.FEATURE_COLS), the same
unconstrained-demand target, the same expanding-window CV folds
(train_model.CV_FOLDS, 7-day gap) and the same final test window
(train_model.FINAL_TEST):

  - Seasonal naive:   lag_7, the same weekday one week back (the most recent
                      value allowed by the 7-day lead time)
  - Moving average:   roll_mean_28_lag7, the 28-day mean ending 7 days back
  - XGBoost  (reg:tweedie)
  - CatBoost (Tweedie:variance_power=p)
  - LightGBM (tweedie)

Each ML model runs twice:
  default  library default parameters (only the Tweedie objective is set;
           CatBoost requires a variance power, so it gets 1.5, which is the
           LightGBM and XGBoost default)
  tuned    Optuna TPE, the same --trials budget, seed and objective (mean CV
           WAPE) for every model; each fold trains up to MAX_TREES with
           EARLY_STOPPING rounds on validation MAE, and the final number of
           trees is the mean best iteration (as in train_model.py). The
           Tweedie power is tuned in [1.1, 1.6] for all three.

The reported CV WAPE re-scores the chosen parameters on every fold with a
fixed number of trees (no early stopping on the fold being scored), so
tuned and default settings are scored the same way. Fold 3 covers Navratri,
Dussehra and Diwali.

This study never writes to models/ or to train_model.py's outputs.

Outputs:
  outputs/model_comparison.csv / .json
  outputs/comparison/wape_by_model.png, cv_wape_by_fold.png, training_time.png

--max-series N is a smoke test: it keeps only the first N city + SKU series
of each segment and writes everything to outputs/comparison_sample/ instead.

Usage: .venv/bin/python src/compare_models.py [--trials 40] [--max-series N]
"""

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import early_stopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBRegressor

from config import CATEGORICAL_FEATURES, FEATURE_COLS, OUTPUTS_DIR, PROJECT_DIR, RANDOM_SEED
from metrics import seasonal_naive_scale, wape
from train_model import (CV_FOLDS, EARLY_STOPPING, FEATURES_PATH, FINAL_TEST, MAX_TREES, SERIES_COLS, TARGET,
                         cv_folds, make_model, score_final_test, split_final, suggest_params)

sys.path.insert(0, str(PROJECT_DIR))
from risk_confusion_matrix import LABELS, forward_sum, risk_label  # noqa: E402

warnings.filterwarnings("ignore", message="Failed to import numba")  # only speeds up DemandMAE, which is cheap

OUT_DIR = OUTPUTS_DIR / "comparison"
SAMPLE_DIR = OUTPUTS_DIR / "comparison_sample"
DIWALI_FOLD = 3
SEGMENTS = [("perishable", 1), ("nonperishable", 0)]
EXTRA_COLS = ["date", "city", "sku_id", "category", "is_perishable", "stockout_flag", "true_demand",
              "opening_stock", "days_until_delivery", TARGET]


def split_signature(train, valid):
    """What a model was fitted and scored on: date bounds, row counts and a hash of the row ids."""
    def describe(rows):
        return {"start": f"{rows['date'].min():%Y-%m-%d}", "end": f"{rows['date'].max():%Y-%m-%d}",
                "n_rows": len(rows), "row_hash": int(pd.util.hash_array(rows.index.to_numpy()).sum() % 2**61)}
    return {"train": describe(train), "valid": describe(valid)}


# ---- models -----------------------------------------------------------------
# Every model exposes fit(train, params, valid=None) -> (model, best_n_trees or None),
# predict(model, frame) and save(model, path). With `valid`, it trains up to
# MAX_TREES and early-stops on validation MAE.

class Baseline:
    """A forecast read straight from a leak-free feature column; nothing to fit."""
    tunable = False
    extension = None

    def __init__(self, name, column):
        self.name, self.column = name, column

    def fit(self, train, params, valid=None):
        return None, None

    def predict(self, model, frame):
        return frame[self.column].fillna(0).to_numpy()


class LightGBM:
    name, tunable, extension = "LightGBM", True, ".txt"
    default_params = {"objective": "tweedie"}

    def __init__(self, categories):
        pass

    def suggest(self, trial):
        return suggest_params(trial)  # the production search space

    def fit(self, train, params, valid=None):
        if valid is None:
            model = make_model(params)
            model.fit(train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)
            return model, None
        model = make_model({**params, "n_estimators": MAX_TREES, "metric": "l1"})
        model.fit(train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES,
                  eval_X=(valid[FEATURE_COLS],), eval_y=(valid[TARGET],),
                  callbacks=[early_stopping(EARLY_STOPPING, verbose=False)])
        return model, int(model.best_iteration_)

    def predict(self, model, frame):
        return model.predict(frame[FEATURE_COLS])

    def save(self, model, path):
        model.booster_.save_model(str(path))


class XGBoost:
    name, tunable, extension = "XGBoost", True, ".ubj"
    default_params = {"objective": "reg:tweedie"}

    def __init__(self, categories):
        self.categories = categories  # fixed per segment, so codes mean the same in every split

    def features(self, frame):
        X = frame[FEATURE_COLS].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = pd.Categorical(X[col], categories=self.categories[col])
        return X

    def suggest(self, trial):
        return {
            "objective": "reg:tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }

    def make(self, params):
        return XGBRegressor(**params, enable_categorical=True, random_state=RANDOM_SEED, n_jobs=-1)

    def fit(self, train, params, valid=None):
        if valid is None:
            model = self.make(params)
            model.fit(self.features(train), train[TARGET])
            return model, None
        model = self.make({**params, "n_estimators": MAX_TREES, "eval_metric": "mae",
                           "early_stopping_rounds": EARLY_STOPPING})
        model.fit(self.features(train), train[TARGET], eval_set=[(self.features(valid), valid[TARGET])], verbose=False)
        return model, int(model.best_iteration) + 1

    def predict(self, model, frame):
        return model.predict(self.features(frame))

    def save(self, model, path):
        model.save_model(str(path))


class DemandMAE:
    """
    CatBoost eval metric: MAE of exp(raw prediction), i.e. on the demand scale.
    CatBoost's built-in MAE compares the raw log-scale value with the target
    under a Tweedie loss, which would early-stop on the wrong quantity.
    """
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        error = np.abs(np.exp(np.asarray(approxes[0])) - np.asarray(target)).sum()
        return float(error), float(len(target))

    def get_final_error(self, error, weight):
        return error / weight


class CatBoost:
    name, tunable, extension = "CatBoost", True, ".cbm"
    default_params = {"loss_function": "Tweedie:variance_power=1.5"}

    def __init__(self, categories):
        pass

    def suggest(self, trial):
        power = trial.suggest_float("tweedie_variance_power", 1.1, 1.6)
        return {
            "loss_function": f"Tweedie:variance_power={power}",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "rsm": trial.suggest_float("rsm", 0.6, 0.95),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        }

    def make(self, params):
        return CatBoostRegressor(**params, cat_features=CATEGORICAL_FEATURES, random_seed=RANDOM_SEED,
                                 thread_count=-1, verbose=False, allow_writing_files=False)

    def fit(self, train, params, valid=None):
        if valid is None:
            model = self.make(params)
            model.fit(train[FEATURE_COLS], train[TARGET])
            return model, None
        model = self.make({**params, "n_estimators": MAX_TREES, "eval_metric": DemandMAE(),
                           "early_stopping_rounds": EARLY_STOPPING})
        model.fit(train[FEATURE_COLS], train[TARGET], eval_set=(valid[FEATURE_COLS], valid[TARGET]))
        return model, int(model.get_best_iteration()) + 1

    def predict(self, model, frame):
        return model.predict(frame[FEATURE_COLS], prediction_type="Exponent")  # mean demand, not log

    def save(self, model, path):
        model.save_model(str(path))


def candidates(df):
    """(model, setting) pairs in report order; every one sees the same frame."""
    categories = {col: sorted(df[col].unique()) for col in CATEGORICAL_FEATURES}
    pairs = [(Baseline("Seasonal naive", "lag_7"), "baseline"),
             (Baseline("Moving average 28d", "roll_mean_28_lag7"), "baseline")]
    for cls in (XGBoost, CatBoost, LightGBM):
        model = cls(categories)
        pairs += [(model, "default"), (model, "tuned")]
    return pairs


# ---- evaluation ---------------------------------------------------------------

def cross_validate(df, model, params, early_stop=False):
    """
    WAPE on each train_model.CV_FOLDS fold, plus the split each fold used.
    With early_stop, each fold stops on its own validation set and records
    best_n_trees (tuning only).
    """
    results, splits = [], []
    for number, train, valid in cv_folds(df):
        fitted, best_n = model.fit(train, params, valid if early_stop else None)
        preds = np.clip(model.predict(fitted, valid), 0, None)
        result = {"fold": number, "WAPE_%": round(wape(valid[TARGET], preds), 2)}
        if best_n is not None:
            result["best_n_trees"] = best_n
        results.append(result)
        splits.append(split_signature(train, valid))
    return results, splits


def tune(df, model, n_trials, label):
    """Optuna TPE (seed RANDOM_SEED) minimising mean early-stopped CV WAPE."""
    def objective(trial):
        params = model.suggest(trial)
        folds, _ = cross_validate(df, model, params, early_stop=True)
        trial.set_user_attr("params", params)
        trial.set_user_attr("folds", folds)
        return float(np.mean([fold["WAPE_%"] for fold in folds]))

    def progress(study, trial):
        print(f"  {label} trial {trial.number + 1}/{n_trials}: {trial.value:.2f}% "
              f"(best {study.best_value:.2f}%, {time.perf_counter() - started:.0f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    best = study.best_trial
    params = {**best.user_attrs["params"],
              "n_estimators": int(round(np.mean([fold["best_n_trees"] for fold in best.user_attrs["folds"]])))}
    return params, {"n_trials": n_trials, "seconds": round(time.perf_counter() - started, 1),
                    "best_early_stopped_cv_wape_%": round(study.best_value, 2),
                    "early_stopped_folds": best.user_attrs["folds"]}


def risk_scores(test, preds):
    """
    Risk-level agreement with the label from true_demand, using the same
    delivery-aware rule and H-day forward window as risk_confusion_matrix.py.
    """
    df = test.assign(pred=preds).sort_values(SERIES_COLS + ["date"]).reset_index(drop=True)
    df["pred_sum"] = forward_sum(df, "pred")
    df["true_sum"] = forward_sum(df, "true_demand")
    df = df.dropna(subset=["pred_sum", "true_sum"])
    actual, predicted = risk_label(df, "true_sum"), risk_label(df, "pred_sum")
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, labels=LABELS, zero_division=0)
    stockout = LABELS.index("stockout")
    return {
        "n_rows": len(df),
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "stockout_recall": round(float(recall[stockout]), 4),
        "stockout_precision": round(float(precision[stockout]), 4),
        "stockout_f1": round(float(f1[stockout]), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": {"labels": LABELS,
                             "rows_actual_cols_predicted": pd.crosstab(actual, predicted).reindex(
                                 index=LABELS, columns=LABELS, fill_value=0).to_numpy().tolist()},
    }


def model_size_mb(model, fitted):
    if model.extension is None:
        return 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"model{model.extension}"
        model.save(fitted, path)
        return round(path.stat().st_size / 1e6, 3)


def evaluate_candidate(df, model, setting, n_trials, segment_name):
    """Tunes (if asked), cross-validates, refits on everything before FINAL_TEST and scores the test window."""
    pre, test = split_final(df)
    label = f"[{segment_name}] {model.name} {setting}"
    params, tuning = {}, None
    if setting == "default":
        params = dict(model.default_params)
    elif setting == "tuned":
        params, tuning = tune(pre, model, n_trials, label)
    cv, cv_splits = cross_validate(pre, model, params)

    started = time.perf_counter()
    fitted, _ = model.fit(pre, params)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    preds = np.clip(model.predict(fitted, test), 0, None)
    predict_ms_per_1k = (time.perf_counter() - started) * 1000 / len(test) * 1000

    final_test = score_final_test(test.assign(pred=preds), "pred", seasonal_naive_scale(pre, TARGET, SERIES_COLS))
    result = {
        "segment": segment_name,
        "model": model.name,
        "setting": setting,
        "final_test": final_test,
        "cv_wape_by_fold": [fold["WAPE_%"] for fold in cv],
        "cv_wape_mean": round(float(np.mean([fold["WAPE_%"] for fold in cv])), 2),
        "train_seconds": round(train_seconds, 2),
        "predict_ms_per_1k_rows": round(predict_ms_per_1k, 3),
        "model_size_mb": model_size_mb(model, fitted),
        "risk": risk_scores(test, preds),
        "params": params,
        "tuning": tuning,
        "splits": {"cv": cv_splits, "final_test": split_signature(pre, test)},
    }
    print(f"{label}: test WAPE {final_test['overall']['WAPE_%']}%, CV {result['cv_wape_by_fold']}, "
          f"train {train_seconds:.1f}s", flush=True)
    return result


def compare_segment(df, segment_name, n_trials):
    return [evaluate_candidate(df, model, setting, n_trials, segment_name) for model, setting in candidates(df)]


# ---- reporting -----------------------------------------------------------------

def flatten(result):
    overall, final = result["final_test"]["overall"], result["final_test"]
    risk = result["risk"]
    return {
        "segment": result["segment"], "model": result["model"], "setting": result["setting"],
        "WAPE_%": overall["WAPE_%"], "MAE": overall["MAE"], "RMSE": overall["RMSE"], "MASE": overall["MASE"],
        "Bias_%": overall["Bias_%"],
        "festival_week_WAPE_%": final["festival_weeks"]["WAPE_%"],
        "stockout_day_WAPE_%": final["stockout_days"]["WAPE_%"],
        "WAPE_vs_true_demand_%": overall.get("WAPE_vs_true_demand_%"),
        "cv_WAPE_mean_%": result["cv_wape_mean"],
        "diwali_fold_WAPE_%": result["cv_wape_by_fold"][DIWALI_FOLD - 1],
        "train_seconds": result["train_seconds"],
        "tuning_seconds": result["tuning"]["seconds"] if result["tuning"] else 0.0,
        "predict_ms_per_1k_rows": result["predict_ms_per_1k_rows"],
        "model_size_mb": result["model_size_mb"],
        "risk_accuracy": risk["accuracy"], "stockout_recall": risk["stockout_recall"],
        "stockout_precision": risk["stockout_precision"], "stockout_f1": risk["stockout_f1"],
        "risk_macro_f1": risk["macro_f1"],
    }


# Lower is better for these; higher for the risk scores. |Bias| is compared by size.
LOWER_BETTER = ["WAPE_%", "MAE", "RMSE", "MASE", "festival_week_WAPE_%", "stockout_day_WAPE_%",
                "WAPE_vs_true_demand_%", "cv_WAPE_mean_%", "diwali_fold_WAPE_%", "train_seconds",
                "predict_ms_per_1k_rows", "model_size_mb"]
HIGHER_BETTER = ["risk_accuracy", "stockout_recall", "stockout_precision", "stockout_f1", "risk_macro_f1"]


def beats_lightgbm(table):
    """Every metric on which some other entry beats tuned LightGBM, per segment."""
    lines = []
    for segment, rows in table.groupby("segment", sort=False):
        reference = rows[(rows["model"] == "LightGBM") & (rows["setting"] == "tuned")].iloc[0]
        others = rows[rows["model"] != "LightGBM"].assign(entry=lambda r: r["model"] + " " + r["setting"])
        metrics = [(m, -1) for m in LOWER_BETTER] + [(m, 1) for m in HIGHER_BETTER] + [("Bias_%", 0)]
        for metric, direction in metrics:
            if metric in ("train_seconds", "predict_ms_per_1k_rows", "model_size_mb"):
                others_metric = others[others["setting"] != "baseline"]  # baselines have no model
            else:
                others_metric = others
            values = others_metric[metric].abs() if direction == 0 else others_metric[metric]
            ref = abs(reference[metric]) if direction == 0 else reference[metric]
            better = values < ref if direction <= 0 else values > ref
            for _, row in others_metric[better].iterrows():
                lines.append(f"[{segment}] {row['entry']} beats tuned LightGBM on {metric}: "
                             f"{row[metric]} vs {reference[metric]}")
    return lines


def plot(results, table, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "axes.axisbelow": True})
    segment_colors = {"perishable": "#2a78d6", "nonperishable": "#eb6834"}
    model_colors = {"XGBoost": "#2a78d6", "CatBoost": "#eb6834", "LightGBM": "#1baf7a",
                    "Seasonal naive": "#8a8984", "Moving average 28d": "#52514e"}
    table = table.assign(entry=table["model"] + "\n" + table["setting"])
    entries = list(dict.fromkeys(table["entry"]))

    # 1. test WAPE: one group per model, one bar per segment
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    x = np.arange(len(entries))
    for i, (segment, color) in enumerate(segment_colors.items()):
        values = table[table["segment"] == segment].set_index("entry").loc[entries, "WAPE_%"]
        bars = ax.bar(x + (i - 0.5) * width, values, width * 0.95, color=color, label=segment)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, entries, fontsize=8)
    ax.set_ylabel("Final-test WAPE (%)")
    ax.set_title(f"Final-test WAPE by model ({FINAL_TEST[0]} to {FINAL_TEST[1]}), lower is better")
    ax.legend(title="Segment", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "wape_by_model.png", dpi=120)
    plt.close(fig)

    # 2. CV WAPE by fold: one panel per segment, a line per model (dashed = default params)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    folds = np.arange(1, len(CV_FOLDS) + 1)
    for ax, segment in zip(axes, segment_colors):
        for r in (r for r in results if r["segment"] == segment):
            style = {"baseline": ":", "default": "--", "tuned": "-"}[r["setting"]]
            ax.plot(folds, r["cv_wape_by_fold"], style, marker="o", markersize=5, linewidth=2,
                    color=model_colors[r["model"]], label=f"{r['model']} {r['setting']}")
        ax.set_xticks(folds, [f"Fold {f}" + (" (Diwali)" if f == DIWALI_FOLD else "") for f in folds])
        ax.set_title(segment)
    axes[0].set_ylabel("CV WAPE (%)")
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("CV WAPE by fold (solid = tuned, dashed = defaults, dotted = baselines)")
    fig.tight_layout()
    fig.savefig(out_dir / "cv_wape_by_fold.png", dpi=120)
    plt.close(fig)

    # 3. final-fit training time for the ML models
    ml = table[table["setting"] != "baseline"]
    ml_entries = list(dict.fromkeys(ml["entry"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ml_entries))
    for i, (segment, color) in enumerate(segment_colors.items()):
        values = ml[ml["segment"] == segment].set_index("entry").loc[ml_entries, "train_seconds"]
        bars = ax.bar(x + (i - 0.5) * width, values, width * 0.95, color=color, label=segment)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, ml_entries, fontsize=8)
    ax.set_ylabel("Seconds")
    ax.set_title("Final-fit training time on all pre-test data (excludes tuning)")
    ax.legend(title="Segment", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_time.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare baselines, XGBoost, CatBoost and LightGBM.")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials per tuned model and segment")
    parser.add_argument("--max-series", type=int, default=None,
                        help="smoke test: first N city + SKU series per segment, saved to outputs/comparison_sample/")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_series is None:
        table_dir, chart_dir = OUTPUTS_DIR, OUT_DIR
    else:
        table_dir = chart_dir = SAMPLE_DIR
        table_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"],
                     usecols=lambda col: col in set(FEATURE_COLS) | set(EXTRA_COLS))
    results = []
    for name, is_perishable in SEGMENTS:
        rows = df[df["is_perishable"] == is_perishable]
        if args.max_series is not None:
            keep = rows[SERIES_COLS].drop_duplicates().head(args.max_series)
            rows = rows[pd.MultiIndex.from_frame(rows[SERIES_COLS]).isin(pd.MultiIndex.from_frame(keep))]
        results += compare_segment(rows, name, args.trials)

    table = pd.DataFrame([flatten(r) for r in results])
    table.to_csv(table_dir / "model_comparison.csv", index=False)
    total_seconds = round(time.perf_counter() - started, 1)
    with open(table_dir / "model_comparison.json", "w") as f:
        json.dump({
            "target": TARGET,
            "features": FEATURE_COLS,
            "final_test_window": list(FINAL_TEST),
            "cv_folds": [{"train_end": a, "valid_start": b, "valid_end": c} for a, b, c in CV_FOLDS],
            "trials_per_model": args.trials,
            "max_series_per_segment": args.max_series,
            "max_trees": MAX_TREES,
            "early_stopping_rounds": EARLY_STOPPING,
            "seed": RANDOM_SEED,
            "total_seconds": total_seconds,
            "results": results,
        }, f, indent=2, default=str)
    plot(results, table, chart_dir)

    pd.set_option("display.width", 250)
    columns = ["model", "setting", "WAPE_%", "MAE", "RMSE", "MASE", "Bias_%", "festival_week_WAPE_%",
               "stockout_day_WAPE_%", "WAPE_vs_true_demand_%", "cv_WAPE_mean_%", "diwali_fold_WAPE_%"]
    cost = ["model", "setting", "train_seconds", "tuning_seconds", "predict_ms_per_1k_rows", "model_size_mb",
            "risk_accuracy", "stockout_recall", "stockout_precision", "stockout_f1", "risk_macro_f1"]
    for segment, rows in table.groupby("segment", sort=False):
        print(f"\n=== {segment}: accuracy (final test {FINAL_TEST[0]} to {FINAL_TEST[1]}) ===")
        print(rows[columns].to_string(index=False))
        print(f"=== {segment}: cost and risk-level agreement ===")
        print(rows[cost].to_string(index=False))

    print("\nWhere another model beats tuned LightGBM:")
    print("\n".join(beats_lightgbm(table)) or "  nowhere")
    print(f"\nSaved {table_dir / 'model_comparison.csv'}, .json and charts in {chart_dir}/")
    print(f"Total run time: {total_seconds:.0f}s ({total_seconds / 60:.1f} min) with --trials {args.trials}")


if __name__ == "__main__":
    main()
