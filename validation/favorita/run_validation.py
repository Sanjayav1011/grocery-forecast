"""
run_validation.py
-------------------
Real-data validation of the forecasting approach on Corporación Favorita
(10 Ecuadorian stores, 2016-01-01 .. 2017-08-15; see favorita_data.py).

Same approach as the main project: separate LightGBM Tweedie models for
perishable and non-perishable items (items.csv perishable flag), leak-free
features (every lag >= 7 days), the same search space and metrics
(train_model.suggest_params, metrics.evaluate) and the same baselines
(seasonal naive = lag_7, 7- and 28-day moving averages, all shifted 7 days).

  train     up to TRAIN_END (2017-03-31)
  test      2017-04-01 .. 2017-08-15, scored once; never used for tuning
  tuning    Optuna (TPE, --trials per model, default 30) fits on data up to
            TUNE_TRAIN_END and early-stops on TUNE_VALID, a window inside the
            training period with a 7-day gap; the final model is refit on all
            training data with the best trial's number of trees.
            --trials 0 uses the main project's tuned parameters instead
            (models/params_<segment>.json, read only); --note records why
            in results.json (parameters_note).

Outputs (validation/favorita/outputs/):
  reports/results.json
  figures/fig_wape_vs_baselines.png, fig_holiday_vs_normal.png, fig_actual_vs_pred.png

Usage: .venv/bin/python validation/favorita/run_validation.py [--trials 30]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import favorita_data as fd  # noqa: E402  (also puts src/ on the path)
from favorita_data import CATEGORICAL_FEATURES, FEATURE_COLS, SERIES_COLS, TARGET  # noqa: E402
from config import FAVORITA_FIGURES_DIR, FAVORITA_REPORTS_DIR, MODELS_DIR, RANDOM_SEED  # noqa: E402
from metrics import evaluate, seasonal_naive_scale, wape  # noqa: E402
from train_model import EARLY_STOPPING, MAX_TREES, make_model, suggest_params  # noqa: E402

TRAIN_END = fd.TRAIN_END
TEST = fd.TEST
TUNE_TRAIN_END = "2017-01-24"
TUNE_VALID = ("2017-02-01", "2017-03-31")
SEGMENTS = [("perishable", 1), ("nonperishable", 0)]
MODEL = "LightGBM"
BASELINES = {"Seasonal naive": "lag_7", "Moving average 7d": "roll_mean_7_lag7", "Moving average 28d": "roll_mean_28_lag7"}
METHODS = [MODEL] + list(BASELINES)
RESULTS_PATH = FAVORITA_REPORTS_DIR / "results.json"
DPI = 300

for end, start in [(TUNE_TRAIN_END, TUNE_VALID[0])]:
    assert (pd.Timestamp(start) - pd.Timestamp(end)).days - 1 == 7, "tuning needs a 7-day gap"
assert TUNE_VALID[1] <= TRAIN_END < TEST[0]


def split(df):
    """(train, test): train up to TRAIN_END, test the TEST window; nothing in between is dropped or shared."""
    return df[df["date"] <= TRAIN_END], df[(df["date"] >= TEST[0]) & (df["date"] <= TEST[1])]


def fit(train, params, valid=None):
    model = make_model({**params, "metric": "l1"} if valid is not None else params)
    kwargs = {}
    if valid is not None:
        from lightgbm import early_stopping
        kwargs = {"eval_X": (valid[FEATURE_COLS],), "eval_y": (valid[TARGET],),
                  "callbacks": [early_stopping(EARLY_STOPPING, verbose=False)]}
    model.fit(train[FEATURE_COLS], train[TARGET], categorical_feature=CATEGORICAL_FEATURES, **kwargs)
    return model


def predict(model, frame):
    return np.clip(model.predict(frame[FEATURE_COLS]), 0, None)


def tune(train, name, n_trials):
    """Optuna on the tuning split inside the training period; returns (params, details)."""
    import optuna

    fit_rows = train[train["date"] <= TUNE_TRAIN_END]
    valid = train[(train["date"] >= TUNE_VALID[0]) & (train["date"] <= TUNE_VALID[1])]

    def objective(trial):
        params = {**suggest_params(trial), "n_estimators": MAX_TREES}
        model = fit(fit_rows, params, valid)
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        return wape(valid[TARGET], predict(model, valid))

    def progress(study, trial):
        print(f"  [{name}] trial {trial.number + 1}/{n_trials}: {trial.value:.2f}% "
              f"(best {study.best_value:.2f}%, {time.perf_counter() - started:.0f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    params = suggest_params(optuna.trial.FixedTrial(study.best_params))
    params["n_estimators"] = study.best_trial.user_attrs["best_iteration"]
    return params, {"source": f"Optuna, {n_trials} trials, validated inside the training period",
                    "n_trials": n_trials, "tuning_valid_WAPE_%": round(study.best_value, 2),
                    "seconds": round(time.perf_counter() - started, 1)}


def main_project_params(name):
    saved = json.loads((MODELS_DIR / f"params_{name}.json").read_text())
    return saved["params"], {"source": f"main project's tuned parameters (models/params_{name}.json)", "n_trials": 0}


def score(test, scale):
    """Metric block per method (MAE, RMSE, WAPE, MASE, Bias) on `test` (needs a pred_<method> column per method)."""
    return {method: {"n_rows": len(test), **evaluate(test[TARGET], test[f"pred_{method}"], scale)} for method in METHODS}


def wape_block(rows):
    return {"n_rows": len(rows), **{method: round(wape(rows[TARGET], rows[f"pred_{method}"]), 2) for method in METHODS}}


def run(n_trials, note=""):
    started = time.perf_counter()
    df, info = fd.build_dataset()
    data_seconds = time.perf_counter() - started
    scored, scales, parameters, rows, train_all = [], {}, {}, {}, []
    for name, flag in SEGMENTS:
        train, test = split(df[df["perishable"] == flag])
        print(f"[{name}] train {len(train):,} rows, test {len(test):,} rows", flush=True)
        params, details = tune(train, name, n_trials) if n_trials > 0 else main_project_params(name)
        fit_started = time.perf_counter()
        model = fit(train, params)
        test = test.assign(segment=name, **{f"pred_{MODEL}": predict(model, test)},
                           **{f"pred_{method}": test[col].fillna(0).to_numpy() for method, col in BASELINES.items()})
        details["final_fit_seconds"] = round(time.perf_counter() - fit_started, 1)
        parameters[name] = {**details, "params": params}
        scales[name] = seasonal_naive_scale(train, TARGET, SERIES_COLS)
        rows[name] = {"train": len(train), "test": len(test), "series": int(test[SERIES_COLS].drop_duplicates().shape[0])}
        scored.append(test)
        train_all.append(train[SERIES_COLS + ["date", TARGET]])
        print(f"[{name}] test WAPE {wape(test[TARGET], test[f'pred_{MODEL}']):.2f}%", flush=True)
    test = pd.concat(scored, ignore_index=True)
    scales["combined"] = seasonal_naive_scale(pd.concat(train_all), TARGET, SERIES_COLS)
    rows["combined"] = {key: sum(rows[name][key] for name, _ in SEGMENTS) for key in ("train", "test", "series")}

    groups = {name: test[test["segment"] == name] for name, _ in SEGMENTS} | {"combined": test}
    metrics = {name: score(frame, scales[name]) for name, frame in groups.items()}
    holiday = {name: {"holiday_days": wape_block(frame[frame["is_holiday"] == 1]),
                      "normal_days": wape_block(frame[frame["is_holiday"] == 0])} for name, frame in groups.items()}
    stores = pd.DataFrame(info["stores"]).set_index("store_nbr")
    by_store = {int(store): {"city": stores.loc[store, "city"], **wape_block(frame)}
                for store, frame in test.groupby("store_nbr")}
    holiday_days = (test[test["is_holiday"] == 1].groupby("date")["store_nbr"].unique()
                    .map(lambda s: sorted(int(x) for x in s)))
    test_holidays = [{"date": day.date().isoformat(), "stores": s} for day, s in holiday_days.items()]

    results = {
        "experiment": "Real-data validation on Corporación Favorita (Ecuador): 7-day-ahead daily demand per store x item",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dates": {"subset": [fd.START, fd.END], "train": [fd.START, TRAIN_END], "test": list(TEST),
                  "tuning_fit": [fd.START, TUNE_TRAIN_END], "tuning_valid": list(TUNE_VALID),
                  "earthquake_flag": list(fd.EARTHQUAKE)},
        "stores": info["stores"],
        "items": {"top_items_per_store": fd.TOP_ITEMS_PER_STORE, "ranked_on": [fd.START, TRAIN_END], **info["items"]},
        "cleaning": info["cleaning"],
        "rows": rows,
        "target": f"{TARGET} = unit_sales clipped at 0 (no stock data, so not un-censored)",
        "features": FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURES,
        "parameters": parameters,
        "parameters_note": note,
        "mase_scale": {name: round(value, 4) for name, value in scales.items()},
        "metrics": metrics,
        "holiday_vs_normal_wape_%": holiday,
        "test_holidays": test_holidays,
        "wape_by_store_%": by_store,
        "run_time_seconds": {"data_and_features": round(data_seconds, 1), "total": None},
    }
    FAVORITA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FAVORITA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["examples"] = plot_actual_vs_pred(test, stores)
    plot_wape_vs_baselines(metrics)
    plot_holiday_vs_normal(holiday)
    results["run_time_seconds"]["total"] = round(time.perf_counter() - started, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {RESULTS_PATH} and figures in {FAVORITA_FIGURES_DIR}/ ({results['run_time_seconds']['total']:.0f}s)")
    print_summary(results)
    return results


# ---- figures -------------------------------------------------------------------

COLORS = {MODEL: "#2a78d6", "Seasonal naive": "#eb6834", "Moving average 7d": "#1baf7a", "Moving average 28d": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEGMENT_LABELS = {"perishable": "Perishable", "nonperishable": "Non-perishable", "combined": "Combined"}


def style(ax, title, subtitle, ylabel):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10 + 12 * (subtitle.count("\n") + 1))
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def save(fig, name):
    fig.savefig(FAVORITA_FIGURES_DIR / name, dpi=DPI, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def grouped_bars(ax, groups, series, values, colors, label_series):
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, name in enumerate(series):
        heights = [values[group][name] for group in groups]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, heights, width * 0.92, color=colors[name], label=name)
        if name in label_series:
            ax.bar_label(bars, labels=[f"{h:.1f}%" for h in heights], padding=2, fontsize=8, color=INK)
    ax.set_xticks(x, [SEGMENT_LABELS[group] for group in groups])


def plot_wape_vs_baselines(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(SEGMENT_LABELS)
    values = {group: {method: metrics[group][method]["WAPE_%"] for method in METHODS} for group in groups}
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    grouped_bars(ax, groups, METHODS, values, COLORS, label_series=METHODS)
    style(ax, "Forecast error on real store data: LightGBM vs baselines",
          f"Favorita, 10 stores, test window {TEST[0]} to {TEST[1]}, 7-day-ahead daily forecasts. Lower is better.",
          "WAPE (%)")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_wape_vs_baselines.png")


def plot_holiday_vs_normal(holiday):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(SEGMENT_LABELS)
    series = ["Normal days", "Holiday days"]
    colors = {"Normal days": COLORS[MODEL], "Holiday days": COLORS["Seasonal naive"]}
    values = {group: {"Normal days": holiday[group]["normal_days"][MODEL],
                      "Holiday days": holiday[group]["holiday_days"][MODEL]} for group in groups}
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    grouped_bars(ax, groups, series, values, colors, label_series=series)
    n_holiday = holiday["combined"]["holiday_days"]["n_rows"]
    n_normal = holiday["combined"]["normal_days"]["n_rows"]
    style(ax, "LightGBM error on holidays vs normal days",
          f"Favorita test window. Holiday = national, regional or local day off for that store\n"
          f"({n_holiday:,} holiday vs {n_normal:,} normal store-item-days).", "WAPE (%)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_holiday_vs_normal.png")


def example_series(test, stores):
    """Three store-items: top perishable in Quito, top non-perishable in Guayaquil, a median-volume item in Manta."""
    volume = test.groupby(SERIES_COLS + ["perishable"])[TARGET].sum().reset_index()
    city_store = {city: store for store, city in stores["city"].items()}
    picks = []
    top = lambda rows: rows.sort_values([TARGET, "item_nbr"], ascending=[False, True]).iloc[0]
    picks.append(("top perishable", top(volume[(volume["store_nbr"] == city_store["Quito"]) & (volume["perishable"] == 1)])))
    picks.append(("top non-perishable", top(volume[(volume["store_nbr"] == city_store["Guayaquil"]) & (volume["perishable"] == 0)])))
    manta = volume[volume["store_nbr"] == city_store["Manta"]].sort_values([TARGET, "item_nbr"]).reset_index(drop=True)
    picks.append(("median-volume item", manta.iloc[len(manta) // 2]))
    return picks


def plot_actual_vs_pred(test, stores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    items = pd.read_csv(fd.raw_path("items.csv")).set_index("item_nbr")
    picks = example_series(test, stores)
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 9), facecolor="white", sharex=True)
    examples = []
    for ax, (label, pick) in zip(axes, picks):
        store, item = int(pick["store_nbr"]), int(pick["item_nbr"])
        rows = test[(test["store_nbr"] == store) & (test["item_nbr"] == item)].sort_values("date")
        family = items.loc[item, "family"].title()
        for day in rows.loc[rows["is_holiday"] == 1, "date"]:
            ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color="#f1f0ec", linewidth=0)
        ax.plot(rows["date"], rows[TARGET], color=MUTED, linewidth=1.2, label="Actual units sold")
        ax.plot(rows["date"], rows[f"pred_{MODEL}"], color=COLORS[MODEL], linewidth=2, label="LightGBM forecast (7 days ahead)")
        item_wape = wape(rows[TARGET], rows[f"pred_{MODEL}"])
        style(ax, f"{stores.loc[store, 'city']} store {store} · item {item} ({family}) · {label}",
              f"WAPE {item_wape:.1f}% over the test window; shaded = holiday for this store", "Units per day")
        ax.set_ylim(bottom=0)
        examples.append({"label": label, "store_nbr": store, "city": stores.loc[store, "city"], "item_nbr": item,
                         "family": items.loc[item, "family"], "perishable": int(pick["perishable"]),
                         "WAPE_%": round(item_wape, 2)})
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Actual vs forecast daily sales, three example store-items (Favorita test window)",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(0.06, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
    save(fig, "fig_actual_vs_pred.png")
    return examples


def print_summary(results):
    lines = [f"{'segment':<15}{'method':<20}{'WAPE %':>8}{'MASE':>8}{'Bias %':>8}{'MAE':>8}"]
    for segment, methods in results["metrics"].items():
        for method, block in methods.items():
            lines.append(f"{segment:<15}{method:<20}{block['WAPE_%']:>8.2f}{block['MASE']:>8.3f}"
                         f"{block['Bias_%']:>8.2f}{block['MAE']:>8.2f}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Favorita real-data validation.")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per model (max 30); 0 uses the main project's tuned parameters")
    parser.add_argument("--note", default="", help="why these parameters were used (saved in results.json)")
    args = parser.parse_args()
    if not 0 <= args.trials <= 30:
        parser.error("--trials must be between 0 and 30")
    if not fd.data_available():
        sys.exit(f"Favorita CSVs not found in {fd.FAVORITA_DATA_DIR}")
    run(args.trials, args.note)


if __name__ == "__main__":
    main()
