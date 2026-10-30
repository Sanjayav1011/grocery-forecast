"""
Festival readiness (src/festival_readiness.py): the shortfall / status math
on hand-made frames, the /api/readiness route and the optional "Festival
readiness" section of the vendor email.

Serving date 29 Oct 2026 (stock counted 30 Oct), Diwali 8 Nov 2026: the
festival week is 2-8 Nov, so 3 days of demand (30 Oct - 1 Nov) come before it.
"""

import csv
import io
import urllib.error

import pandas as pd
import pytest

import alerts
import festival_readiness
from festival_readiness import readiness
from test_dashboard_filters import fake_data, get, server  # noqa: F401  (pytest fixtures)

SERVING, DIWALI = "2026-10-29", "2026-11-08"


def frame(**columns):
    defaults = {"model_category": "snacks", "forecast_daily": 10.0, "festival_week_units": 140.0,
                "opening_stock": 50, "days_until_delivery": 2}
    return pd.DataFrame([{**defaults, **columns}])


def test_shortfall_math_for_a_non_perishable():
    # snacks: weekly deliveries (cycle 7), 10 a day. Stock 50; deliveries on 1 Nov (offset 2) and 8 Nov.
    # Only 1 Nov arrives before the festival week: 7 x 10 = 70. The 30 units sold before the week
    # come out of the oldest stock first: 50 -> 20. Available 20 + 70 = 90; demand 140 -> short 50.
    row = readiness(frame(), DIWALI, SERVING).iloc[0]
    assert row["available"] == pytest.approx(90)
    assert row["shortfall"] == pytest.approx(50)
    assert row["last_delivery"] == pd.Timestamp("2026-11-01")  # the last delivery before the festival
    assert row["order_by"] == pd.Timestamp("2026-10-25")  # 7-day order lead time
    assert row["status"] == "Order now"  # 1 Nov is within 7 days of 29 Oct
    assert row["suggested_order"] == 90  # 50 + 2 days x 140 / 7


def test_demand_before_the_week_uses_the_oldest_stock_first():
    # 10 on hand covers 1 of the 3 days before the week; the other 20 come out of the 1 Nov delivery (70 -> 50)
    row = readiness(frame(opening_stock=10), DIWALI, SERVING).iloc[0]
    assert row["available"] == pytest.approx(50)
    assert row["shortfall"] == pytest.approx(90)


def test_perishables_dont_count_stock_that_will_have_expired():
    # the same numbers for dairy (3-day cycle and shelf life): stock on hand (delivered 28 Oct) and the
    # 31 Oct delivery expire before Diwali, so nothing counts and the whole festival week is short
    dairy, snacks = (readiness(frame(model_category=category, opening_stock=40, days_until_delivery=1),
                               DIWALI, SERVING).iloc[0] for category in ["dairy", "snacks"])
    assert snacks["available"] == pytest.approx(80)  # 40 - 30 + the 31 Oct delivery (70), which keeps for months
    assert dairy["available"] == 0 and dairy["shortfall"] == pytest.approx(140)
    # dairy's last delivery before the festival is 6 Nov, more than 7 days out: at risk, order by 30 Oct
    assert (dairy["status"], dairy["last_delivery"], dairy["order_by"]) == (
        "At risk", pd.Timestamp("2026-11-06"), pd.Timestamp("2026-10-30"))


def test_stock_within_shelf_life_on_the_festival_day_counts():
    # festival 1 Nov (offset 2): sweets keep 4 days. Stock delivered 30 Oct keeps to 2 Nov and counts;
    # stock delivered 27 Oct keeps to 30 Oct and doesn't
    fresh, stale = (readiness(frame(model_category="sweets", opening_stock=100, days_until_delivery=until,
                                    festival_week_units=70.0), "2026-11-01", SERVING).iloc[0] for until in [4, 1])
    assert fresh["available"] == pytest.approx(100) and fresh["status"] == "Ready"
    assert stale["available"] == 0 and stale["shortfall"] == pytest.approx(70)


def test_no_shortfall_is_ready():
    row = readiness(frame(opening_stock=1000), DIWALI, SERVING).iloc[0]
    assert row["available"] == pytest.approx(1040)  # 1000 - 30 + 70
    assert (row["shortfall"], row["status"], row["suggested_order"]) == (0, "Ready", 0)


def test_readiness_route_filters_sorts_and_downloads(server):
    payload = get(server, "/api/readiness", festival="Diwali")
    assert (payload["festival"], payload["date"], payload["city"]) == ("Diwali", "2026-11-08", "Delhi")
    # Delhi in the fixture: 7 products after de-duplication; beverages (A1) has no Diwali uplift
    assert payload["products"] == 6 and sum(payload["counts"].values()) == 6
    assert set(payload["counts"]) == {"Order now", "At risk", "Ready"}
    statuses = [row["status"] for row in payload["rows"]]
    assert statuses == sorted(statuses, key=["Order now", "At risk", "Ready"].index)  # default order
    sweets = get(server, "/api/readiness", festival="Diwali", category="sweets")
    assert {row["sku_id"] for row in sweets["rows"]} == {"T3", "A2"}
    by_stock = get(server, "/api/readiness", festival="Diwali", sort="stock", order="asc")
    stock = [row["opening_stock"] for row in by_stock["rows"]]
    assert stock == sorted(stock)
    rows = list(csv.DictReader(io.StringIO(get(server, "/api/readiness", festival="Diwali", format="csv", status="Order now"))))
    assert len(rows) == payload["counts"]["Order now"] and all(row["status"] == "Order now" for row in rows)
    for params, code in [({"city": "Atlantis", "festival": "Diwali"}, 404), ({"festival": "Diwali", "sort": "name"}, 400)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, "/api/readiness", **params)
        assert error.value.code == code


def test_alert_section_only_for_a_festival_within_21_days(fake_data):
    # data date 25 Aug 2026 in the fixture: Diwali (75 days) is too far, so no section
    assert festival_readiness.alert_rows("Delhi", alerts.READINESS_WINDOW_DAYS) is None
    uplift = pd.read_csv(fake_data / "festival_uplift.csv")
    pd.concat([uplift, uplift.assign(festival="Raksha Bandhan")]).to_csv(fake_data / "festival_uplift.csv", index=False)
    import festival_uplift
    festival_uplift._UPLIFTS.clear()
    section = festival_readiness.alert_rows("Delhi", alerts.READINESS_WINDOW_DAYS)  # Raksha Bandhan, 28 Aug
    assert (section["festival"], section["days_until"]) == ("Raksha Bandhan", 3)
    assert set(section["rows"]["status"]) <= {"Order now"}
    shortfall = section["rows"]["shortfall"].tolist()
    assert shortfall == sorted(shortfall, reverse=True)


def test_vendor_email_has_the_order_now_section(tmp_path, monkeypatch):
    from test_alerts import REPORT_COLS, report_rows
    pd.DataFrame(report_rows(1, "Delhi", "2026-10-30", "snacks", 1.0), columns=REPORT_COLS).to_csv(
        tmp_path / alerts.REPORT_FILES["stockout"], index=False)
    pd.DataFrame(columns=REPORT_COLS).to_csv(tmp_path / alerts.REPORT_FILES["overstock"], index=False)
    monkeypatch.setattr(alerts, "REPORT_DIR", tmp_path)
    rows = pd.DataFrame([{"sku_id": f"S{i}", "product_name": f"Sweet {i}", "model_category": "sweets" if i % 2 else "dairy",
                          "opening_stock": 10, "festival_week_units": 100.0, "shortfall": 90.0 - i,
                          "order_by": pd.Timestamp("2026-10-25"), "suggested_order": 120} for i in range(30)])
    monkeypatch.setattr(alerts, "load_readiness",
                        lambda city: {"festival": "Diwali", "date": "2026-11-08", "days_until": 10, "rows": rows})
    alert = alerts.build_alert("Delhi")
    assert "STOCKOUT" in alert["text"] and "OVERSTOCK" in alert["text"]  # existing sections stay
    assert "FESTIVAL READINESS: DIWALI 2026-11-08 – ORDER NOW (30)" in alert["text"]
    assert alert["text"].count("Sweet ") == alerts.READINESS_CAP and "+5 more" in alert["text"]
    assert "Planning estimate, stock simulated" in alert["html"] and alert["readiness"]["order_now"] == 30
    dairy_only = alerts.build_alert("Delhi", categories=["dairy"])  # category routing applies
    assert dairy_only["readiness"]["order_now"] == 15
