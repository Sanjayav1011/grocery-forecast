"""
"Highest demand — next 7 days" and "Festival planning" on the admin page:
festival uplifts measured from past festivals only (src/festival_uplift.py),
every admin date counted from the data date (never the computer's clock),
and the planning list / product-panel note. Uses the hand-made data set from
test_dashboard_filters.py (data date 2026-08-25).
"""

import datetime
import inspect
import json
import re
import urllib.error

import pandas as pd
import pytest

import dashboard
import festival_uplift
import product_detail
from festival_calendar import build_calendar_table, festival_df
from test_dashboard_filters import fake_data, get, server  # noqa: F401  (pytest fixtures)

DATA_DATE = pd.Timestamp("2026-08-25")


def flat_demand(start="2022-01-01", end="2026-12-31", level=10.0):
    days = pd.date_range(start, end)
    return pd.DataFrame({"date": days, "city": "Delhi", "category": "sweets", "demand": level})


def diwali_dates():
    festivals = festival_df()
    return festivals.loc[festivals["festival_name"] == "Diwali", "date"].tolist()


def test_uplift_uses_only_past_festivals():
    daily = flat_demand()
    for diwali in diwali_dates():
        week = daily["date"].between(diwali - pd.Timedelta(days=6), diwali)
        # past Diwali weeks double; Diwali 2026 is after the data date, so its x100 must be ignored
        daily.loc[week, "demand"] *= 100 if diwali > DATA_DATE else 2
    daily.loc[daily["date"] > DATA_DATE + pd.Timedelta(days=60), "demand"] = 1000  # later history must not count either
    ratios = festival_uplift.occurrence_ratios(daily, DATA_DATE)
    diwali = ratios[ratios["festival"] == "Diwali"]
    assert sorted(pd.to_datetime(diwali["date"]).dt.year) == [2022, 2023, 2024, 2025]  # 2021 is before the data starts
    assert (pd.to_datetime(ratios["date"]) <= DATA_DATE).all()
    summary = festival_uplift.summarise(ratios).set_index(["festival", "city", "category"])
    row = summary.loc[("Diwali", "Delhi", "sweets")]
    assert row["uplift"] == pytest.approx(2.0)
    assert (row["occurrences"], row["first_year"], row["last_year"]) == (4, 2022, 2025)


def test_uplift_moves_with_the_data_date():
    daily = flat_demand()
    for diwali in diwali_dates():
        daily.loc[daily["date"].between(diwali - pd.Timedelta(days=6), diwali), "demand"] *= 2 + diwali.year - 2022
    # data up to 25 Oct 2024: Diwali 2024 (31 Oct) is still ahead, even though its week has
    # started, so only 2022 (x2) and 2023 (x3) count
    ratios = festival_uplift.occurrence_ratios(daily, pd.Timestamp("2024-10-25"))
    diwali = ratios[ratios["festival"] == "Diwali"]
    assert sorted(diwali["ratio"].round(6)) == [2.0, 3.0]


def test_baseline_is_the_normal_weeks_before_the_festival_week():
    daily = flat_demand()
    diwali = pd.Timestamp("2025-10-20")
    daily.loc[daily["date"].between(diwali - pd.Timedelta(days=6), diwali), "demand"] = 30
    # a busy Dussehra / Navratri run-up is not a normal week, so it can't dilute the baseline
    daily.loc[daily["date"].between("2025-09-20", "2025-10-03"), "demand"] = 50
    ratio = festival_uplift.occurrence_ratios(daily, DATA_DATE).query("festival == 'Diwali' and date == @diwali")
    assert ratio["ratio"].iloc[0] == pytest.approx(3.0)
    assert ratio["normal_days"].iloc[0] == festival_uplift.NORMAL_DAYS


class NoClock(datetime.date):
    @classmethod
    def today(cls):
        raise AssertionError("the admin page must not use the computer's date")


def write_admin_outputs(tmp_path):
    report = [{"date": "2026-08-25", "city": "Delhi", "sku_id": "T1", "product_name": "Amul Lassi 200 ml", "brand": "Amul",
               "category": "dairy", "predicted_units": 12, "opening_stock": 18, "days_of_cover": 1.5, "horizon_demand": 84,
               "avg_daily_pred": 12, "days_until_delivery": 2, "shelf_life_days": 3, "reorder_qty": 90}]
    pd.DataFrame(report).to_csv(tmp_path / "stockout_risk_report.csv", index=False)
    pd.DataFrame(columns=list(report[0])).to_csv(tmp_path / "overstock_risk_report.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"segments": {}}))
    calendar = build_calendar_table("2026-01-01", "2027-03-31")
    calendar.assign(date=calendar["date"].dt.date.astype(str)).to_csv(tmp_path / "festival_calendar_5yr.csv", index=False)


def test_every_admin_date_counts_from_the_data_date(fake_data, monkeypatch):
    write_admin_outputs(fake_data)
    monkeypatch.setattr(dashboard, "date", NoClock)  # any call to today() fails the test
    assert dashboard.data_date() == datetime.date(2026, 8, 25)

    payload = dashboard.dashboard_payload("Delhi")
    assert payload["data_date"] == payload["latest"] == "2026-08-25"
    raksha = {"name": "Raksha Bandhan", "date": "2026-08-28", "days_until": 3}
    assert payload["next_festival"] == raksha

    delhi = next(region for region in dashboard.regional_intelligence() if region["region"] == "Delhi")
    assert (delhi["data_date"], delhi["window_start"], delhi["window_end"]) == ("2026-08-25", "2026-08-26", "2026-09-01")
    assert delhi["next_festival"] == raksha

    plan = festival_uplift.festival_planning("Delhi")  # default: the first festival after the 7-day window
    assert (plan["festival"], plan["date"], plan["days_until"], plan["data_date"]) == ("Ganesh Chaturthi", "2026-09-14", 20, "2026-08-25")
    assert all(option["date"] > "2026-09-01" for option in plan["options"])
    assert product_detail.product_detail("Delhi", "T1")["today"] == "2026-08-25"


def test_admin_page_never_reads_the_clock():
    source = inspect.getsource(dashboard)
    assert "date.today" not in source and "datetime.now" not in source
    script = dashboard.ADMIN_PAGE[dashboard.ADMIN_PAGE.index("<script>"):]
    assert not re.search(r"new Date\(\s*\)", script) and "Date.now" not in script
    assert "Data as of" in script and "UPDATED" not in script


def test_festival_planning_list(server):
    plan = get(server, "/api/festival-planning", festival="diwali")
    assert (plan["festival"], plan["date"], plan["days_until"]) == ("Diwali", "2026-11-08", 75)
    assert plan["uplifts"][0] == {"category": "sweets", "uplift": 3.0, "occurrences": 5, "first_year": 2021, "last_year": 2025}
    rows = {row["sku_id"]: row for row in plan["rows"]}
    # festival-week units = 7-day forecast x the category's uplift; A1 (beverages) has no uplift, A4 duplicates A1
    assert [row["sku_id"] for row in plan["rows"]] == ["T3", "T2", "T1", "A2", "A3", "T4"]
    assert (rows["T3"]["forecast_7d"], rows["T3"]["uplift"], rows["T3"]["festival_week_units"]) == (98.0, 3.0, 294.0)
    assert rows["A2"]["revenue_estimate"] == 126 * 900
    by_revenue = get(server, "/api/festival-planning", festival="Diwali", rank="revenue")
    assert [row["sku_id"] for row in by_revenue["rows"]] == ["A2", "T3", "T2", "A3", "T1", "T4"]  # A3 / T1 tie at ₹4,200
    assert get(server, "/api/festival-planning")["rows"] == []  # Ganesh Chaturthi: no measured uplift in this data
    for params, code in [({"festival": "Onam"}, 400), ({"rank": "profit"}, 400), ({"city": "Atlantis"}, 404)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, "/api/festival-planning", **params)
        assert error.value.code == code


def test_product_panel_shows_the_uplift_calculation(server):
    note = get(server, "/api/product", sku_id="T3", festival="Diwali")["festival_planning"]
    assert note == {"festival": "Diwali", "date": "2026-11-08", "category": "sweets", "uplift": 3.0, "occurrences": 5,
                    "first_year": 2021, "last_year": 2025, "seven_day_units": 98.0, "festival_week_units": 294.0}
    assert get(server, "/api/product", sku_id="T3")["festival_planning"] is None
    assert get(server, "/api/product", sku_id="A1", festival="Diwali")["festival_planning"] is None  # no beverages uplift
