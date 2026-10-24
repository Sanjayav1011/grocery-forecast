"""
risk_confusion_matrix.py
--------------------------
How well the forecast reproduces the risk label we would assign if we knew
the true demand. For each city + sku and day of the final test window, the
next-H-day sum (H = LEAD_TIME_DAYS, today included) of pred and of
true_demand is turned into a risk label with inventory.classify_risk, the
same rule and horizon as the risk report in train_model.py: cover = stock /
(sum / H); stockout if cover < days_until_delivery, overstock if
cover > H + 2 + shelf_life_days.

Rows in the last H-1 days of each series have no full H-day window and are
left out. Reads outputs/test_predictions.csv (written by src/train_model.py).

Usage: .venv/bin/python risk_confusion_matrix.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import LEAD_TIME_DAYS, OUTPUTS_DIR  # noqa: E402
from inventory import classify_risk  # noqa: E402

HORIZON = LEAD_TIME_DAYS
LABELS = ["stockout", "normal", "overstock"]
NAMES = ["Stockout", "Normal", "Overstock"]
SERIES_COLS = ["city", "sku_id"]


def forward_sum(df, col):
    """Sum of col over today..today+HORIZON-1 per series; NaN without a full window."""
    return df.groupby(SERIES_COLS)[col].transform(
        lambda s: s[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1])


def risk_label(df, sum_col):
    rows = df[["category", "opening_stock", "days_until_delivery"]].assign(avg_daily_pred=df[sum_col] / HORIZON)
    return classify_risk(rows, HORIZON)["risk_status"]


def print_matrix(actual, predicted, title):
    matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=LABELS),
                          index=[f"Actual {n}" for n in NAMES], columns=[f"Pred {n}" for n in NAMES])
    matrix["Total"] = matrix.sum(axis=1)
    matrix.loc["Total"] = matrix.sum()
    print(f"\n{title} (n={len(actual):,})")
    print(matrix.to_string())


def main():
    df = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv", parse_dates=["date"])
    df = df.sort_values(SERIES_COLS + ["date"]).reset_index(drop=True)
    gaps = df.groupby(SERIES_COLS)["date"].diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "series have missing days; forward windows would be wrong"

    df["pred_sum"] = forward_sum(df, "pred")
    df["true_sum"] = forward_sum(df, "true_demand")
    n_all = len(df)
    df = df.dropna(subset=["pred_sum", "true_sum"]).copy()
    print(f"Horizon H={HORIZON} days. {n_all:,} test rows; {n_all - len(df):,} without a full "
          f"{HORIZON}-day window dropped; {len(df):,} scored "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}).")

    df["predicted"] = risk_label(df, "pred_sum")
    df["actual"] = risk_label(df, "true_sum")

    print_matrix(df["actual"], df["predicted"], "All SKUs: rows = actual (true_demand), columns = predicted (pred)")
    print("\nClassification report:")
    print(classification_report(df["actual"], df["predicted"], labels=LABELS, target_names=NAMES,
                                digits=3, zero_division=0))
    print(f"Overall accuracy: {accuracy_score(df['actual'], df['predicted']):.3f}")

    balance = pd.DataFrame({
        "actual_%": df["actual"].value_counts(normalize=True),
        "predicted_%": df["predicted"].value_counts(normalize=True),
    }).reindex(LABELS, fill_value=0).mul(100).round(2)
    balance.index = NAMES
    print("\nClass balance:")
    print(balance.to_string())

    for segment, title in [("perishable", "Perishable"), ("nonperishable", "Non-perishable")]:
        rows = df[df["segment"] == segment]
        print_matrix(rows["actual"], rows["predicted"], title)
        print(f"Accuracy: {accuracy_score(rows['actual'], rows['predicted']):.3f}")


if __name__ == "__main__":
    main()
