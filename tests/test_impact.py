"""
Business-impact estimates (src/impact.py) on small hand-made series, MRP
₹100: sales protected / missed, spoilage avoided and false-alarm cost, one
decision per delivery cycle, and no impact when nothing is at risk.

Snacks: delivered every 7 days (days until delivery 7, 6, ..., 1), shelf life
180. Dairy: every 3 days, shelf life 3. 14 days per series, so a 7-day
cycle has 2 decision days (day 0 and day 7) with a full 7-day window.
"""

import json

import pandas as pd
import pytest

import impact

MRP = 100.0
SHELF = {"snacks": 180, "dairy": 3}
CYCLE = {"snacks": 7, "dairy": 3}


def series(sku, category, pred, true, stock, city="Delhi", days=14):
    """A series with constant pred / true demand; stock: {day: units}, 1,000 days of cover otherwise."""
    return pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=days), "city": city, "sku_id": sku, "category": category,
        "segment": "x", "pred": float(pred), "true_demand": float(true), "demand": float(true),
        "opening_stock": [stock.get(day, true * 30) for day in range(days)],
        "days_until_delivery": [CYCLE[category] - day % CYCLE[category] for day in range(days)],
        "shelf_life_days": SHELF[category],
    })


def run(*frames):
    predictions = pd.concat(frames, ignore_index=True)
    price = pd.Series(MRP, index=predictions["sku_id"].unique())
    return impact.evaluation_impact(predictions, price)


def total(rows, kpi):
    return impact.summarize_evaluation(rows)["total"][kpi]


def test_stockout_caught_is_sales_protected():
    # day 0: 30 units, 10/day forecast and true demand until the delivery on day 7 -> 40 units short
    rows = run(series("S1", "snacks", pred=10, true=10, stock={0: 30, 7: 300}))
    assert len(rows) == 2  # one decision per cycle: days 0 and 7
    assert total(rows, "sales_protected") == 40 * MRP
    assert total(rows, "sales_missed") == total(rows, "false_alarm_cost") == 0
    assert impact.summarize_evaluation(rows)["total"]["stockout_caught_cycles"] == 1


def test_stockout_not_flagged_is_sales_missed():
    # the forecast says 2/day (cover 15 days, normal); true demand 10/day runs out
    rows = run(series("S2", "snacks", pred=2, true=10, stock={0: 30, 7: 300}))
    assert total(rows, "sales_missed") == 40 * MRP
    assert total(rows, "sales_protected") == 0


def test_false_alarm_costs_the_extra_reorder_carried_one_week():
    # forecast 10/day flags a stockout; true 2/day lasts (cover 15). Reorder on the forecast
    # ceil(7x10 + 2x10 - 30) = 60, on true demand max(0, 7x2 + 2x2 - 30) = 0 -> 60 extra units
    rows = run(series("S3", "snacks", pred=10, true=2, stock={0: 30, 7: 300}))
    assert total(rows, "false_alarm_cost") == pytest.approx(60 * MRP * impact.CARRYING_COST_PER_WEEK)
    assert total(rows, "sales_protected") == total(rows, "sales_missed") == 0
    assert impact.summarize_evaluation(rows)["total"]["false_alarm_cycles"] == 1


def test_overstock_caught_is_spoilage_avoided():
    # dairy, shelf life 3: 500 units at 10/day -> 500 - 10 x 3 = 470 expire. Other cycles: 30 units (normal)
    rows = run(series("D1", "dairy", pred=10, true=10, stock={day: 500 if day == 0 else 30 for day in range(14)}))
    assert list(rows["date"].dt.day) == [1, 4, 7]  # cycles starting days 0, 3, 6 have a full 7-day window
    assert total(rows, "spoilage_avoided") == 470 * MRP
    assert total(rows, "sales_protected") == total(rows, "sales_missed") == total(rows, "false_alarm_cost") == 0


def test_no_risk_means_no_impact():
    rows = run(series("N1", "snacks", pred=10, true=10, stock={0: 100, 7: 100}),  # cover 10 days
               series("N2", "dairy", pred=5, true=5, stock={day: 20 for day in range(14)}))  # cover 4 days
    summary = impact.summarize_evaluation(rows)["total"]
    assert {kpi: summary[kpi] for kpi in impact.EVALUATION_KPIS} == dict.fromkeys(impact.EVALUATION_KPIS, 0)
    assert (rows["outcome"] == "").all()


def test_per_city_blocks_add_up_to_the_total():
    rows = run(series("S1", "snacks", 10, 10, {0: 30, 7: 300}, city="Delhi"),
               series("S2", "snacks", 2, 10, {0: 30, 7: 300}, city="Mumbai"),
               series("D1", "dairy", 10, 10, {0: 500}, city="Mumbai"))
    summary = impact.summarize_evaluation(rows)
    assert list(summary["by_city"]) == ["Delhi", "Mumbai"]
    for kpi in impact.EVALUATION_KPIS:
        assert sum(block[kpi] for block in summary["by_city"].values()) == summary["total"][kpi]
    assert summary["by_city"]["Mumbai"]["sales_missed"] == 40 * MRP


def test_generated_summary_has_every_kpi_per_city():
    if not impact.SUMMARY_PATH.exists():
        pytest.skip("impact summary not generated; run run_pipeline.py first")
    summary = json.loads(impact.SUMMARY_PATH.read_text())
    assert summary["evaluation"]["window"] == ["2026-04-01", "2026-08-25"]
    assert "simulated" in summary["evaluation"]["label"] and "Planning estimate" in summary["serving"]["label"]
    for section, kpis in [("evaluation", impact.EVALUATION_KPIS), ("serving", impact.SERVING_KPIS)]:
        block = summary[section]
        assert set(block["by_city"]) == {"Bengaluru", "Chennai", "Delhi", "Mumbai"}
        for kpi in kpis:
            assert sum(city[kpi] for city in block["by_city"].values()) == pytest.approx(block["total"][kpi], abs=4)
