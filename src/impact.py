"""
impact.py
-----------
Estimated business impact of the risk alerts, per city: rupee KPIs for the
admin page and the report. Estimates on simulated sales and stock; the
models, forecasts, risk rule and evaluation outputs are not changed.

Evaluation window (outputs/test_predictions.csv, tracked products,
1 Apr - 25 Aug 2026). Labels come from risk_confusion_matrix.py's rule:
the next-7-day sum of the forecast (flagged) and of true demand (actual),
each through inventory.classify_risk with the product's stock and days until
its next delivery. A stockout lasts until the next delivery, so every
day of a delivery cycle would repeat the same lost sales; each cycle
(product x city x next-delivery date) is therefore counted ONCE, on its
first day in the window (the morning stock is counted after a delivery),
and only when the window reaches its next delivery.

  shortfall            max(0, true demand from that day until the next
                       delivery - opening stock): the sales lost if nothing
                       is done (sales are capped by stock; no delivery in
                       between)
  sales protected      stockout flagged and real: shortfall x MRP
  sales still missed   real stockout not flagged: shortfall x MRP
  spoilage avoided     overstock flagged and real: units that expire unsold,
                       max(0, stock - true daily demand x shelf life) x MRP
  false-alarm cost     stockout flagged, not real: the extra units ordered
                       early, max(0, reorder qty on the forecast - reorder qty
                       on true demand), x MRP x CARRYING_COST_PER_WEEK
                       (carried for one week)

Serving date (config.SERVING_DATE, stock counted the next morning), planning
estimates:
  at risk this week    today's stockout products (serving_risk_report):
                       max(0, 7-day forecast - stock) x MRP
  Diwali value at risk the Diwali-readiness "Order now" products
                       (festival_readiness): shortfall x MRP

MRP: data/sampled_products.csv, missing ones filled with the category median
(as generate_synthetic_data does).

Outputs: outputs/impact_summary.json, outputs/impact_by_city.png.
"""

import json
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, LEAD_TIME_DAYS, OUTPUTS_DIR, PROJECT_DIR
from inventory import classify_risk

HORIZON = LEAD_TIME_DAYS
CARRYING_COST_PER_WEEK = 0.02  # share of the value of stock held one week (capital, space, handling)
SUMMARY_PATH = OUTPUTS_DIR / "impact_summary.json"
CHART_PATH = OUTPUTS_DIR / "impact_by_city.png"
EVALUATION_KPIS = {
    "sales_protected": "Sales protected",
    "sales_missed": "Sales still missed",
    "spoilage_avoided": "Spoilage avoided",
    "false_alarm_cost": "False-alarm cost",
}
SERVING_KPIS = {"at_risk_this_week": "At risk this week", "festival_value_at_risk": "Diwali value at risk"}
FESTIVAL = "Diwali"


def prices(data_dir=DATA_DIR):
    """MRP per sku_id, missing ones filled with the category median."""
    products = pd.read_csv(data_dir / "sampled_products.csv", usecols=["sku_id", "category", "mrp"])
    mrp = products["mrp"].where(products["mrp"] > 0)
    return mrp.fillna(mrp.groupby(products["category"]).transform("median")).set_axis(products["sku_id"])


def reorder_qty(rows, sum_col):
    """classify_risk's reorder quantity with the 7-day sum in sum_col as the demand."""
    frame = rows[["category", "opening_stock", "days_until_delivery"]].assign(avg_daily_pred=rows[sum_col] / HORIZON)
    return classify_risk(frame, HORIZON)["reorder_qty"]


def decision_rows(predictions):
    """
    One row per delivery cycle (first day in the window, full 7-day window,
    window reaching the next delivery) with flagged / actual labels and
    cycle_demand (true demand from that day until the next delivery).
    """
    sys.path.insert(0, str(PROJECT_DIR))
    from risk_confusion_matrix import SERIES_COLS, forward_sum, risk_label  # the evaluation rule (imports sklearn)

    df = predictions.sort_values(SERIES_COLS + ["date"]).reset_index(drop=True)
    gaps = df.groupby(SERIES_COLS)["date"].diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "series have missing days; forward windows would be wrong"
    df["pred_sum"] = forward_sum(df, "pred")
    df["true_sum"] = forward_sum(df, "true_demand")
    df["next_delivery"] = df["date"] + pd.to_timedelta(df["days_until_delivery"], unit="D")
    cycle = SERIES_COLS + ["next_delivery"]
    # true demand from each day to the end of its cycle (rows of a cycle are consecutive days)
    df["cycle_demand"] = df[::-1].groupby(cycle)["true_demand"].cumsum()[::-1]
    series_end = df.groupby(SERIES_COLS)["date"].transform("max")
    reaches_delivery = df["next_delivery"] <= series_end + pd.Timedelta(days=1)
    df = df[df["pred_sum"].notna() & df["true_sum"].notna() & reaches_delivery]
    first = df.groupby(cycle)["date"].transform("min") == df["date"]
    df = df[first].copy()
    df["flagged"] = risk_label(df, "pred_sum")
    df["actual"] = risk_label(df, "true_sum")
    return df


def evaluation_impact(predictions, price):
    """Per decision row: shortfall, expiring and extra units and the four KPI values (rupees)."""
    df = decision_rows(predictions)
    df["mrp"] = df["sku_id"].map(price)
    df["shortfall"] = (df["cycle_demand"] - df["opening_stock"]).clip(lower=0)
    df["expiring"] = (df["opening_stock"] - df["true_sum"] / HORIZON * df["shelf_life_days"]).clip(lower=0)
    df["extra_units"] = (reorder_qty(df, "pred_sum") - reorder_qty(df, "true_sum")).clip(lower=0)
    real_stockout, flagged_stockout = df["actual"] == "stockout", df["flagged"] == "stockout"
    real_overstock, flagged_overstock = df["actual"] == "overstock", df["flagged"] == "overstock"
    df["sales_protected"] = np.where(real_stockout & flagged_stockout, df["shortfall"] * df["mrp"], 0.0)
    df["sales_missed"] = np.where(real_stockout & ~flagged_stockout, df["shortfall"] * df["mrp"], 0.0)
    df["spoilage_avoided"] = np.where(real_overstock & flagged_overstock, df["expiring"] * df["mrp"], 0.0)
    df["false_alarm_cost"] = np.where(flagged_stockout & ~real_stockout,
                                      df["extra_units"] * df["mrp"] * CARRYING_COST_PER_WEEK, 0.0)
    df["outcome"] = np.select(
        [real_stockout & flagged_stockout, real_stockout, flagged_stockout,
         real_overstock & flagged_overstock],
        ["stockout_caught", "stockout_missed", "false_alarm", "overstock_caught"], default="")
    return df


def summarize_evaluation(df):
    """Rupees and cycle counts per city and in total."""
    def block(rows):
        counts = rows["outcome"].value_counts()
        return {**{kpi: round(float(rows[kpi].sum())) for kpi in EVALUATION_KPIS},
                "cycles": int(len(rows)),
                **{f"{name}_cycles": int(counts.get(name, 0))
                   for name in ["stockout_caught", "stockout_missed", "false_alarm", "overstock_caught"]}}

    return {"by_city": {city: block(rows) for city, rows in df.groupby("city")}, "total": block(df)}


def serving_impact(cities):
    """At risk this week (serving model risk report) and Diwali "Order now" value at risk, per city."""
    import festival_readiness
    import serving_risk_report

    report = serving_risk_report.build_report()
    price = prices()
    stockout = report[report["risk_status"] == "stockout"].copy()
    stockout["value"] = (stockout["horizon_demand"] - stockout["opening_stock"]).clip(lower=0) * stockout["sku_id"].map(price)
    by_city, meta = {}, None
    for city in cities:
        meta, table = festival_readiness.city_readiness(city, FESTIVAL, next_after_data_date=True)
        order_now = table[table["status"] == "Order now"]
        in_city = stockout[stockout["city"] == city]
        by_city[city] = {
            "at_risk_this_week": round(float(in_city["value"].sum())), "stockout_products": int(len(in_city)),
            "festival_value_at_risk": round(float(order_now["value_at_risk"].sum())),
            "order_now_products": int(len(order_now)),
            "order_now_without_mrp": int(order_now["value_at_risk"].isna().sum()),
        }
    total = {key: sum(block[key] for block in by_city.values()) for key in next(iter(by_city.values()))}
    return {"date": str(report["date"].iloc[0].date()), "festival": meta["festival"],
            "festival_date": meta["date"].date().isoformat(), "by_city": by_city, "total": total}


def rupees(value):
    """₹ in Indian units: crore (1e7), lakh (1e5), else rupees."""
    short = lambda number: f"{number:,.2f}".rstrip("0").rstrip(".")
    if abs(value) >= 1e7:
        return f"₹{short(value / 1e7)} Cr"
    if abs(value) >= 1e5:
        return f"₹{short(value / 1e5)} L"
    return f"₹{value:,.0f}"


def save_chart(summary, path=CHART_PATH):
    """Small multiples: one panel per KPI, one bar per city (one value axis per panel, in rupees)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("evaluation", key, label, "#2a78d6") for key, label in EVALUATION_KPIS.items()]
    panels += [("serving", key, label, "#eb6834") for key, label in SERVING_KPIS.items()]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (section, key, label, color) in zip(axes.flat, panels):
        by_city = summary[section]["by_city"]
        cities = list(by_city)
        values = [by_city[city][key] for city in cities]
        ax.bar(cities, values, color=color, width=0.6)
        for i, value in enumerate(values):
            ax.text(i, value, rupees(value), ha="center",
                    va="bottom", fontsize=8, color="#173b28")
        ax.set_title(f"{label}" + (" (evaluation)" if section == "evaluation" else " (planning)"), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: rupees(v)))
        ax.margins(y=0.15)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    window = summary["evaluation"]["window"]
    fig.suptitle(f"Estimated business impact by city (simulated sales and stock). Evaluation: {window[0]} to "
                 f"{window[1]}; planning: stock on {summary['serving']['stock_date']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_summary():
    predictions = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv", parse_dates=["date"])
    evaluation = evaluation_impact(predictions, prices())
    serving = serving_impact(sorted(predictions["city"].unique()))
    stock_date = (pd.Timestamp(serving["date"])).date().isoformat()
    return {
        "evaluation": {
            "label": "Estimate · simulated sales and stock · evaluation window",
            "window": [predictions["date"].min().date().isoformat(), predictions["date"].max().date().isoformat()],
            "carrying_cost_per_week": CARRYING_COST_PER_WEEK,
            **summarize_evaluation(evaluation),
        },
        "serving": {"label": "Planning estimate · simulated stock · serving date", "stock_date": stock_date, **serving},
        "currency": "INR",
    }


def main():
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    save_chart(summary)
    rows = {city: {**summary["evaluation"]["by_city"][city], **summary["serving"]["by_city"][city]}
            for city in summary["evaluation"]["by_city"]}
    rows["Total"] = {**summary["evaluation"]["total"], **summary["serving"]["total"]}
    table = pd.DataFrame(rows).T[list(EVALUATION_KPIS) + list(SERVING_KPIS)]
    print(f"Estimated impact (₹; evaluation {summary['evaluation']['window'][0]}..{summary['evaluation']['window'][1]}, "
          f"planning on stock of {summary['serving']['stock_date']}):")
    print(table.to_string())
    print(f"Saved {SUMMARY_PATH} and {CHART_PATH}")


if __name__ == "__main__":
    main()
