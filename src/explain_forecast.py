"""
explain_forecast.py
---------------------
Why the saved LightGBM models forecast what they do, from LightGBM's own
SHAP values (Booster.predict(X, pred_contrib=True); no extra dependency).
The models and the forecasts are not changed: the same feature rows are
scored, only with pred_contrib.

SHAP in the Tweedie model's raw (log) space: for one row,
    raw prediction = base value + sum of the feature contributions
    forecast       = exp(raw prediction)
so contributions multiply the forecast rather than add to it. To read them
as units/day, each one is scaled by the logarithmic mean of the baseline
b = exp(base value) and the forecast f:
    units_i = contribution_i x (f - b) / (ln f - ln b)
The units then add up exactly: b + sum(units_i) = f, every driver keeps its
sign, and a driver's share of the gap f - b equals its share of the log gap.
It is an approximation of "what this driver adds" (a multiplicative effect
is worth more units on a big forecast than a small one), not a second model.
Per product the 7 forecast days are converted day by day, then averaged.

Drivers group the features (DRIVERS covers every model feature once).

Outputs:
  data/serving_explanations.csv   per tracked product x city at the serving
                                  date (forecast SERVING_DATE + 1..7): baseline,
                                  forecast, units per driver and label context
  outputs/shap_importance.json    global importance per segment: mean |SHAP|
                                  per feature on the evaluation test window
                                  (train_model.FINAL_TEST, 1 Apr - 25 Aug 2026)
  outputs/shap_importance_<segment>.png   the same, top 10 features
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DATA_DIR, FEATURE_COLS, LEAD_TIME_DAYS, MODELS_DIR, OUTPUTS_DIR, SERVING_DATE
from festival_calendar import festival_df

SEGMENTS = {"perishable": 1, "nonperishable": 0}
SERVING_PATH = DATA_DIR / "serving_explanations.csv"
IMPORTANCE_PATH = OUTPUTS_DIR / "shap_importance.json"
FEATURES_PATH = DATA_DIR / "features.csv"
TEST_WINDOW = ("2026-04-01", "2026-08-25")  # train_model.FINAL_TEST (importing train_model pulls in optuna)
TOP_FEATURES = 10
TOP_DRIVERS = 6

DRIVERS = {
    "festival": ["days_until_next_festival", "days_since_last_festival", "next_festival_weight", "is_festival_week"],
    "day_of_week": ["day_of_week", "is_weekend"],
    "recent_demand": ["lag_7", "lag_14", "lag_21", "roll_mean_7_lag7", "roll_mean_14_lag7", "roll_mean_28_lag7",
                      "roll_std_14_lag7"],
    "last_year": ["lag_364"],
    "price": ["unit_price", "log_mrp"],
    "city": ["city_code"],
    "product": ["category_code", "brand_code", "sku_id_code", "pack_size_value"],
    "other": ["on_promotion", "month", "week_of_year", "is_month_start", "is_month_end"],
}
FEATURE_LABELS = {
    "unit_price": "Selling price", "on_promotion": "On promotion",
    "days_until_next_festival": "Days until next festival", "days_since_last_festival": "Days since last festival",
    "next_festival_weight": "Size of next festival", "is_festival_week": "Festival week",
    "day_of_week": "Day of week", "is_weekend": "Weekend", "month": "Month", "week_of_year": "Week of year",
    "is_month_start": "Month start", "is_month_end": "Month end",
    "lag_7": "Sales 7 days before", "lag_14": "Sales 14 days before", "lag_21": "Sales 21 days before",
    "lag_364": "Sales same day last year", "roll_mean_7_lag7": "Recent 7-day average",
    "roll_mean_14_lag7": "Recent 14-day average", "roll_mean_28_lag7": "Recent 28-day average",
    "roll_std_14_lag7": "Recent sales volatility", "log_mrp": "MRP", "pack_size_value": "Pack size",
    "city_code": "City", "category_code": "Category", "sku_id_code": "Product", "brand_code": "Brand",
}
FEATURE_GROUP = {feature: group for group, features in DRIVERS.items() for feature in features}
_BOOSTERS, _CACHE = {}, {}


class ExplanationNotFound(LookupError):
    pass


def booster(segment):
    if segment not in _BOOSTERS:
        _BOOSTERS[segment] = lgb.Booster(model_file=str(MODELS_DIR / f"lgbm_{segment}.txt"))
    return _BOOSTERS[segment]


def contributions(model, X):
    """(per-feature SHAP in raw log space, DataFrame like X; base value per row) from LightGBM's pred_contrib."""
    values = model.predict(X[FEATURE_COLS], pred_contrib=True)
    return pd.DataFrame(values[:, :-1], index=X.index, columns=FEATURE_COLS), values[:, -1]


def to_units(contrib, base):
    """Contributions in units/day: each scaled by the log-mean of exp(base) and the forecast (see the module doc)."""
    baseline = np.exp(base)
    log_gap = contrib.sum(axis=1).to_numpy()
    forecast = baseline * np.exp(log_gap)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(log_gap) > 1e-12, (forecast - baseline) / log_gap, baseline)
    return contrib.mul(scale, axis=0), baseline, forecast


def by_driver(frame):
    """Feature columns summed into DRIVERS columns."""
    return pd.DataFrame({group: frame[features].sum(axis=1) for group, features in DRIVERS.items()}, index=frame.index)


def serving_rows(as_of=None):
    """The serving feature rows of every tracked series (the rows forecast_regional scores)."""
    from build_features import RAW_POS_PATH
    from forecast_regional import HISTORY_DAYS, serving_features

    as_of = pd.Timestamp(as_of or SERVING_DATE)
    pos = pd.read_csv(RAW_POS_PATH, parse_dates=["date"])
    history = pos[(pos["date"] > as_of - pd.Timedelta(days=HISTORY_DAYS)) & (pos["date"] <= as_of)]
    rows, _ = serving_features(history, as_of)
    return rows, as_of


def serving_explanations(rows, as_of):
    """Per series: baseline, forecast (mean of the 7 days) and mean units per driver, plus label context."""
    out = []
    for segment, is_perishable in SEGMENTS.items():
        part = rows[rows["is_perishable"] == is_perishable]
        if part.empty:
            continue
        contrib, base = contributions(booster(segment), part)
        units, baseline, forecast = to_units(contrib, base)
        day = by_driver(units).add_prefix("units_").assign(
            city=part["city"].to_numpy(), sku_id=part["sku_id"].to_numpy(), segment=segment,
            baseline=baseline, forecast=forecast, raw_base=base,
            recent_28=part["roll_mean_28_lag7"].to_numpy(), last_year=part["lag_364"].to_numpy(),
            unit_price=part["unit_price"].to_numpy(), category=part["category"].to_numpy())
        out.append(day.groupby(["city", "sku_id", "segment", "category"], as_index=False).mean(numeric_only=True))
    table = pd.concat(out, ignore_index=True)
    upcoming = festival_df()
    upcoming = upcoming[upcoming["date"] > as_of].sort_values("date").head(1)
    table["festival"] = upcoming["festival_name"].iloc[0] if not upcoming.empty else ""
    table["festival_days"] = int((upcoming["date"].iloc[0] - as_of).days) if not upcoming.empty else -1
    table["forecast_start"] = (as_of + pd.Timedelta(days=1)).date().isoformat()
    table["forecast_end"] = (as_of + pd.Timedelta(days=LEAD_TIME_DAYS)).date().isoformat()
    return table


def test_window_importance(features_path=FEATURES_PATH, window=TEST_WINDOW):
    """Per segment, every feature's mean |SHAP| (raw log space) on the evaluation test window, largest first."""
    df = pd.read_csv(features_path, usecols=["date", "is_perishable"] + FEATURE_COLS, parse_dates=["date"])
    df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])]
    result = {}
    for segment, is_perishable in SEGMENTS.items():
        part = df[df["is_perishable"] == is_perishable]
        contrib, _ = contributions(booster(segment), part)
        mean_abs = contrib.abs().mean().sort_values(ascending=False)
        result[segment] = {"rows": len(part), "features": [
            {"feature": feature, "label": FEATURE_LABELS[feature], "driver": FEATURE_GROUP[feature],
             "mean_abs_shap": round(float(value), 5), "typical_effect_pct": round(float(np.expm1(value) * 100), 1)}
            for feature, value in mean_abs.items()]}
    return {"window": list(window), "note": "mean |SHAP| in the Tweedie model's log space on the evaluation test "
                                            "window (simulated sales history); typical_effect_pct = exp(mean |SHAP|) - 1",
            "segments": result}


def save_importance_charts(importance, outputs_dir=OUTPUTS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for segment, block in importance["segments"].items():
        top = block["features"][:TOP_FEATURES][::-1]
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.barh([row["label"] for row in top], [row["mean_abs_shap"] for row in top], color="#1f8f4d", height=0.6)
        for i, row in enumerate(top):
            ax.text(row["mean_abs_shap"], i, f"  {row['mean_abs_shap']:.3f} (~{row['typical_effect_pct']:.0f}%)",
                    va="center", fontsize=8, color="#173b28")
        ax.set_xlabel("mean |SHAP| (log scale; ~% = typical effect on the forecast)")
        ax.set_title(f"What drives demand - {segment} model\nmean |SHAP| on the test window {importance['window'][0]} "
                     f"to {importance['window'][1]} (simulated sales history)", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.25)
        fig.tight_layout()
        fig.savefig(outputs_dir / f"shap_importance_{segment}.png", dpi=110)
        plt.close(fig)


# ---- the dashboard's /api/explain ------------------------------------------------

def load_serving():
    if "serving" not in _CACHE:
        _CACHE["serving"] = pd.read_csv(SERVING_PATH).set_index(["city", "sku_id"])
    return _CACHE["serving"]


def load_importance():
    if "importance" not in _CACHE:
        _CACHE["importance"] = json.loads(IMPORTANCE_PATH.read_text())
    return _CACHE["importance"]


def clear_cache():
    _CACHE.clear()


def _day(iso):
    return pd.Timestamp(iso).strftime("%-d %b")


def driver_label(driver, row, city):
    """Plain-English name of a driver for one product's forecast window."""
    if driver == "festival":
        days = int(row["festival_days"])
        return f"{row['festival']} in {days} day{'' if days == 1 else 's'}" if days >= 0 else "Festivals"
    if driver == "day_of_week":
        days = pd.date_range(row["forecast_start"], row["forecast_end"])
        weekend = days[days.day_of_week >= 5]
        return f"Day of week (weekend {_day(weekend[0])}–{_day(weekend[-1])})" if len(weekend) else "Day of week"
    if driver == "recent_demand":
        return f"Recent sales (~{row['recent_28']:,.0f}/day over 4 weeks)"
    if driver == "last_year":
        return "Same time last year" + ("" if pd.isna(row["last_year"]) else f" (~{row['last_year']:,.0f}/day)")
    if driver == "price":
        return f"Price ₹{row['unit_price']:,.0f}"
    if driver == "city":
        return f"City: {city}"
    if driver == "product":
        return "Product, brand & category"
    return "Season & other (month, week; no promotion)"


def explain_series(city, sku_id):
    """One tracked product x city's explanation; ExplanationNotFound when it isn't tracked."""
    table = load_serving()
    if (city, sku_id) not in table.index:
        raise ExplanationNotFound(sku_id)
    row = table.loc[(city, sku_id)]
    drivers = sorted(({"driver": driver, "label": driver_label(driver, row, city),
                       "units": round(float(row[f"units_{driver}"]), 2)} for driver in DRIVERS),
                     key=lambda item: -abs(item["units"]))
    rest = drivers[TOP_DRIVERS:]
    return {
        "city": city, "sku_id": sku_id, "segment": row["segment"], "category": row["category"],
        "window": {"start": row["forecast_start"], "end": row["forecast_end"]},
        "baseline": round(float(row["baseline"]), 2), "forecast": round(float(row["forecast"]), 2),
        "drivers": drivers[:TOP_DRIVERS],
        "rest": {"count": len(rest), "units": round(sum(item["units"] for item in rest), 2)},
        "all_drivers": drivers,
    }


def explain(city, sku_id):
    """
    /api/explain: tracked products get their own explanation; analog products
    the explanation of their top-weighted analog in the same city.
    ExplanationNotFound for an unknown city / SKU.
    """
    import stockout_view

    regional, _ = stockout_view.raw_rows("regional")
    match = regional[(regional["city"].str.casefold() == city.strip().casefold()) & (regional["sku_id"] == sku_id.strip())]
    if match.empty:
        raise ExplanationNotFound(sku_id)
    row = match.iloc[0]
    payload = {"city": row["city"], "sku_id": row["sku_id"], "product_name": str(row["product_name"]),
               "forecast_daily": round(float(row["forecast_daily"]), 2), "analog": None,
               "notes": ["SHAP values from the saved LightGBM model (trained on simulated sales), converted from the "
                         "model's log scale to approximate units/day; the 7 forecast days are averaged."]}
    if row["forecast_source"] == stockout_view.DIRECT:
        payload["source"] = "direct"
        payload["explanation"] = explain_series(row["city"], row["sku_id"])
        return payload
    ids = str(row["analog_ids"]).split(";")
    weights = [float(value) for value in str(row["analog_weights"]).split(";")]
    price_adj = [float(value) for value in str(row["analog_price_adj"]).split(";")]
    top = int(np.argmax(weights))
    names = regional[(regional["city"] == row["city"]) & (regional["sku_id"] == ids[top])]["product_name"]
    payload["source"] = "analog"
    payload["analog"] = {"count": len(ids), "sku_id": ids[top], "product_name": str(names.iloc[0]) if len(names) else "",
                         "weight": weights[top], "price_adjustment": price_adj[top]}
    payload["explanation"] = explain_series(row["city"], ids[top])
    return payload


def importance_payload():
    """/api/explain/importance: the top TOP_FEATURES per segment."""
    importance = load_importance()
    return {"window": importance["window"], "note": importance["note"],
            "segments": {segment: {"rows": block["rows"], "features": block["features"][:TOP_FEATURES]}
                         for segment, block in importance["segments"].items()}}


def main():
    rows, as_of = serving_rows()
    table = serving_explanations(rows, as_of)
    table.to_csv(SERVING_PATH, index=False)
    print(f"Saved {len(table):,} serving explanations (forecast {table['forecast_start'].iloc[0]}.."
          f"{table['forecast_end'].iloc[0]}) to {SERVING_PATH}")
    importance = test_window_importance()
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2))
    save_importance_charts(importance)
    for segment, block in importance["segments"].items():
        top = ", ".join(f"{row['label']} {row['mean_abs_shap']:.3f}" for row in block["features"][:5])
        print(f"[{segment}] {block['rows']:,} test-window rows; top: {top}")
    print(f"Saved {IMPORTANCE_PATH} and shap_importance_<segment>.png")


if __name__ == "__main__":
    main()
