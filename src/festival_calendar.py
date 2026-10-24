"""
festival_calendar.py
---------------------
Builds an Indian festival reference calendar (2021-01-01 to 2027-12-31)
and exposes a reusable function to compute a "festival demand uplift" signal
for any date + product category.

IMPORTANT: Lunar-calendar festivals (Diwali, Holi, Eid, Raksha Bandhan,
Ganesh Chaturthi, Navratri/Dussehra, Karva Chauth) shift every year and can
vary by a day depending on regional Panchang conventions. The dates below
were checked against public calendars at the time this script was written.
Re-verify each year's dates against an authoritative source (e.g. your
state government holiday calendar or a Panchang site) before relying on
this for a live forecasting system -- do not treat these as permanently
fixed.

This module is imported by BOTH generate_synthetic_data.py (to create demo
sales with realistic festival spikes) and build_features.py (to turn the
same calendar into model features), so the synthetic data and the feature
pipeline always stay consistent.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# name, date, "weight" (relative importance / how much buying behaviour shifts)
FESTIVALS = [
    # 2021
    ("New Year",        "2021-01-01", 0.3),
    ("Makar Sankranti",  "2021-01-14", 0.4),
    ("Republic Day",     "2021-01-26", 0.3),
    ("Holi",              "2021-03-29", 0.7),
    ("Eid-ul-Fitr",       "2021-05-14", 0.6),
    ("Independence Day",  "2021-08-15", 0.3),
    ("Raksha Bandhan",    "2021-08-22", 0.5),
    ("Ganesh Chaturthi",  "2021-09-10", 0.8),
    ("Navratri Start",    "2021-10-07", 0.5),
    ("Dussehra",          "2021-10-15", 0.7),
    ("Diwali",            "2021-11-04", 1.0),
    ("Christmas",         "2021-12-25", 0.4),
    # 2022
    ("New Year",        "2022-01-01", 0.3),
    ("Makar Sankranti",  "2022-01-14", 0.4),
    ("Republic Day",     "2022-01-26", 0.3),
    ("Holi",              "2022-03-18", 0.7),
    ("Eid-ul-Fitr",       "2022-05-03", 0.6),
    ("Independence Day",  "2022-08-15", 0.3),
    ("Raksha Bandhan",    "2022-08-11", 0.5),
    ("Ganesh Chaturthi",  "2022-08-31", 0.8),
    ("Navratri Start",    "2022-09-26", 0.5),
    ("Dussehra",          "2022-10-05", 0.7),
    ("Diwali",            "2022-10-24", 1.0),
    ("Christmas",         "2022-12-25", 0.4),
    # 2023
    ("New Year",        "2023-01-01", 0.3),
    ("Makar Sankranti",  "2023-01-15", 0.4),
    ("Republic Day",     "2023-01-26", 0.3),
    ("Holi",              "2023-03-08", 0.7),
    ("Eid-ul-Fitr",       "2023-04-22", 0.6),
    ("Independence Day",  "2023-08-15", 0.3),
    ("Raksha Bandhan",    "2023-08-30", 0.5),
    ("Ganesh Chaturthi",  "2023-09-19", 0.8),
    ("Navratri Start",    "2023-10-15", 0.5),
    ("Dussehra",          "2023-10-24", 0.7),
    ("Diwali",            "2023-11-12", 1.0),
    ("Christmas",         "2023-12-25", 0.4),
    # 2024
    ("New Year",         "2024-01-01", 0.3),
    ("Makar Sankranti",   "2024-01-15", 0.4),
    ("Republic Day",      "2024-01-26", 0.3),
    ("Holi",               "2024-03-25", 0.7),
    ("Eid-ul-Fitr",        "2024-04-11", 0.6),
    ("Independence Day",   "2024-08-15", 0.3),
    ("Raksha Bandhan",     "2024-08-19", 0.5),
    ("Ganesh Chaturthi",   "2024-09-07", 0.8),
    ("Navratri Start",     "2024-10-03", 0.5),
    ("Dussehra",           "2024-10-12", 0.7),
    ("Diwali",             "2024-10-31", 1.0),
    ("Christmas",          "2024-12-25", 0.4),
    # 2025
    ("New Year",          "2025-01-01", 0.3),
    ("Makar Sankranti",    "2025-01-14", 0.4),
    ("Republic Day",       "2025-01-26", 0.3),
    ("Holi",                "2025-03-14", 0.7),
    ("Eid-ul-Fitr",         "2025-03-31", 0.6),
    ("Independence Day",    "2025-08-15", 0.3),
    ("Raksha Bandhan",      "2025-08-09", 0.5),
    ("Ganesh Chaturthi",    "2025-08-27", 0.8),
    ("Navratri Start",      "2025-09-22", 0.5),
    ("Dussehra",            "2025-10-02", 0.7),
    ("Diwali",              "2025-10-20", 1.0),
    ("Christmas",           "2025-12-25", 0.4),
    # 2026
    ("New Year",           "2026-01-01", 0.3),
    ("Makar Sankranti",     "2026-01-14", 0.4),
    ("Republic Day",        "2026-01-26", 0.3),
    ("Holi",                 "2026-03-04", 0.7),
    ("Eid-ul-Fitr",          "2026-03-20", 0.6),
    ("Independence Day",     "2026-08-15", 0.3),
    ("Raksha Bandhan",       "2026-08-28", 0.5),
    ("Ganesh Chaturthi",     "2026-09-14", 0.8),
    ("Mahalaya Amavasya",    "2026-10-10", 0.8),
    ("Navratri Start",       "2026-10-12", 0.5),
    ("Dussehra",             "2026-10-20", 0.7),
    ("Diwali",               "2026-11-08", 1.0),
    ("Christmas",            "2026-12-25", 0.4),
    # 2027
    ("New Year",            "2027-01-01", 0.3),
    ("Makar Sankranti",      "2027-01-14", 0.4),
    ("Republic Day",         "2027-01-26", 0.3),
    ("Holi",                  "2027-03-23", 0.7),
    ("Eid-ul-Fitr",           "2027-03-09", 0.6),
    ("Independence Day",      "2027-08-15", 0.3),
    ("Raksha Bandhan",        "2027-08-17", 0.5),
    ("Ganesh Chaturthi",      "2027-09-04", 0.8),
    ("Navratri Start",        "2027-10-01", 0.5),
    ("Dussehra",              "2027-10-09", 0.7),
    ("Diwali",                "2027-10-29", 1.0),
    ("Christmas",             "2027-12-25", 0.4),
]

# How strongly each category reacts to festivals (relative sensitivity).
# Sweets/dairy spike hardest before festivals; household goods barely move.
CATEGORY_FESTIVAL_SENSITIVITY = {
    "sweets": 3.0,
    "dairy": 1.8,
    "vegetables": 1.5,
    "fruits": 1.4,
    "beverages": 1.2,
    "snacks": 1.3,
    "household": 0.6,
}

# Categories whose festival response isn't proportional to festival weight:
# peak uplift per festival name ("*" = every other festival).
# Personal care: Diwali gifting ~1.3x, other festivals ~1.05x.
FESTIVAL_PEAK_OVERRIDES = {
    "personal_care": {"Diwali": 0.30, "*": 0.05},
}

# Frozen vegetables and dry fruits (the catalog's "vegetables"/"fruits") are not perishable.
PERISHABLE_CATEGORIES = {"dairy", "sweets"}

# Directional festival distances are capped so far-away festivals look alike.
FESTIVAL_DAYS_CAP = 60


def festival_df():
    df = pd.DataFrame(FESTIVALS, columns=["festival_name", "date", "weight"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_calendar_table(start="2021-01-01", end="2027-12-31"):
    """
    Daily calendar with directional festival features: days until the next
    festival (0 on the festival day) and days since the last one, both capped
    at FESTIVAL_DAYS_CAP, plus the next festival's name and weight.
    """
    dates = pd.date_range(start, end, freq="D")
    fdf = festival_df()
    cal = pd.DataFrame({"date": dates})

    nxt = fdf.rename(columns={"festival_name": "next_festival", "weight": "next_festival_weight"})
    nxt["next_festival_date"] = nxt["date"]
    cal = pd.merge_asof(cal, nxt, on="date", direction="forward")

    last = fdf[["date"]].assign(last_festival_date=fdf["date"])
    cal = pd.merge_asof(cal, last, on="date", direction="backward")

    # No festival within the calendar on that side -> treat as "far away".
    cal["days_until_next_festival"] = (
        (cal["next_festival_date"] - cal["date"]).dt.days
        .fillna(FESTIVAL_DAYS_CAP).clip(upper=FESTIVAL_DAYS_CAP).astype(int)
    )
    cal["days_since_last_festival"] = (
        (cal["date"] - cal["last_festival_date"]).dt.days
        .fillna(FESTIVAL_DAYS_CAP).clip(upper=FESTIVAL_DAYS_CAP).astype(int)
    )
    cal["next_festival"] = cal["next_festival"].fillna("No festival")
    cal["next_festival_weight"] = cal["next_festival_weight"].fillna(0.0)
    cal = cal.drop(columns=["next_festival_date", "last_festival_date"])
    cal["is_festival_week"] = (cal["days_until_next_festival"] <= 7).astype(int)
    cal["day_of_week"] = cal["date"].dt.day_of_week
    cal["is_weekend"] = cal["day_of_week"].isin([5, 6]).astype(int)
    cal["month"] = cal["date"].dt.month
    return cal


def festival_uplift(dates: pd.Series, category: str, sigma_days: float = 2.5, peak_offset: int = -1):
    """
    Returns a multiplicative demand uplift factor (1.0 = no effect) for each
    date, for the given product category, based on proximity to festivals.
    peak_offset=-1 means demand peaks 1 day BEFORE the festival (typical for
    grocery -- people stock up in advance), not on the festival day itself.
    """
    sensitivity = CATEGORY_FESTIVAL_SENSITIVITY.get(category, 0.5)
    overrides = FESTIVAL_PEAK_OVERRIDES.get(category)
    fdf = festival_df()
    dates = pd.to_datetime(dates)
    uplift = np.ones(len(dates), dtype=float)

    for _, row in fdf.iterrows():
        peak_date = row["date"] + pd.Timedelta(days=peak_offset)
        days_from_peak = (dates - peak_date).dt.days.values
        peak = (overrides.get(row["festival_name"], overrides["*"]) if overrides
                else row["weight"] * sensitivity)
        bump = peak * np.exp(-0.5 * (days_from_peak / sigma_days) ** 2)
        uplift += bump
    return uplift


if __name__ == "__main__":
    cal = build_calendar_table()
    out_path = Path(__file__).resolve().parents[1] / "data" / "festival_calendar_5yr.csv"
    cal.to_csv(out_path, index=False)
    print(f"Saved {len(cal)} rows to {out_path}")
    print(cal.head(10))
